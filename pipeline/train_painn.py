"""
Train PaiNN on Transition1x DFT data using SchNetPack.

Pipeline:
  1. Load SchNetPack .db (created by convert_transition1x.py)
  2. Build PaiNN model (energy + forces)
  3. Train with PyTorch Lightning
  4. Save best checkpoint + training curves

Usage:
    python train_painn.py \
        --db      ~/data/transition1x_train.db \
        --output  ~/painn_results \
        --epochs  500 \
        --gpu

    # Smoke test (CPU, few steps):
    python train_painn.py \
        --db     ~/data/transition1x_test50.db \
        --output ~/painn_smoke \
        --epochs 5 \
        --batch-size 8
"""
import argparse
import os

import torch
import torch.serialization
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import AtomsDataModule

ENERGY_KEY = 'energy'
FORCES_KEY = 'forces'


def build_model(cutoff, n_rbf, n_atom_basis, n_interactions):
    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

    representation = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )

    pred_energy = spk.atomistic.Atomwise(
        n_in=n_atom_basis,
        output_key=ENERGY_KEY,
    )
    pred_forces = spk.atomistic.Forces(energy_key=ENERGY_KEY)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets(ENERGY_KEY, add_mean=True, add_atomrefs=False),
        ],
    )
    return nnpot


def main(args):
    os.makedirs(args.output, exist_ok=True)

    # Data
    datamodule = AtomsDataModule(
        datapath=args.db,
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        transforms=[
            trn.ASENeighborList(cutoff=args.cutoff),
            trn.RemoveOffsets(ENERGY_KEY, remove_mean=True, remove_atomrefs=False),
            trn.CastTo32(),
        ],
        property_units={ENERGY_KEY: 'eV', FORCES_KEY: 'eV/Angstrom'},
        num_workers=args.num_workers,
        pin_memory=args.gpu,
        load_properties=[ENERGY_KEY, FORCES_KEY],
    )
    datamodule.prepare_data()
    datamodule.setup()

    # Model
    nnpot = build_model(
        cutoff=args.cutoff,
        n_rbf=args.n_rbf,
        n_atom_basis=args.n_atom_basis,
        n_interactions=args.n_interactions,
    )

    # Task
    output_energy = spk.task.ModelOutput(
        name=ENERGY_KEY,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=args.energy_weight,
        metrics={'MAE': torchmetrics.MeanAbsoluteError()},
    )
    output_forces = spk.task.ModelOutput(
        name=FORCES_KEY,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=args.forces_weight,
        metrics={'MAE': torchmetrics.MeanAbsoluteError()},
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={'lr': args.lr},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={'mode': 'min', 'factor': 0.5, 'patience': 50},
        scheduler_monitor='val_loss',
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output, 'checkpoints'),
            filename='best',
            monitor='val_loss',
            save_top_k=1,
            mode='min',
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=150,
            mode='min',
        ),
    ]

    logger = TensorBoardLogger(args.output, name='painn')

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpu else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=args.output,
        inference_mode=False,
    )

    # PyTorch 2.6 changed weights_only default to True, which breaks Lightning checkpoint loading
    torch.serialization.add_safe_globals([])  # trigger import; actual fix below
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

    trainer.fit(task, datamodule=datamodule, ckpt_path=getattr(args, 'resume_from', None))

    torch.load = _orig_load  # restore

    print(f"\nDone. Best model: {args.output}/checkpoints/best.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',            required=True, help='SchNetPack .db file')
    parser.add_argument('--output',        required=True, help='Output directory')
    parser.add_argument('--epochs',        type=int,   default=500)
    parser.add_argument('--batch-size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--cutoff',        type=float, default=5.0,  help='Angstrom')
    parser.add_argument('--n-rbf',         type=int,   default=20)
    parser.add_argument('--n-atom-basis',  type=int,   default=128)
    parser.add_argument('--n-interactions',type=int,   default=3)
    parser.add_argument('--energy-weight', type=float, default=0.01)
    parser.add_argument('--forces-weight', type=float, default=0.99)
    parser.add_argument('--num-train',     type=int,   default=None)
    parser.add_argument('--num-val',       type=int,   default=None)
    parser.add_argument('--num-workers',   type=int,   default=4)
    parser.add_argument('--gpu',           action='store_true')
    parser.add_argument('--resume-from',   default=None,  help='Resume from checkpoint path')
    args = parser.parse_args()
    main(args)
