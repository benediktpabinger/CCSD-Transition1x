"""
Train delta correction head on top of frozen MACE.

Loads pre-processed train/val data from delta_cache/, runs the frozen MACE
backbone each batch to get node_feats, then trains a NonLinearReadoutBlock
head to predict delta_energy and delta_forces.

    delta(geometry) = E_wB97M-V/def2-TZVP - E_wB97X-D3/6-31G(d)

Output: ~/delta_head/delta_head.pt  (head state_dict only)

Usage:
    python train_delta_head.py [--epochs 200] [--batch-size 32] [--lr 1e-3]
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
torch.serialization.add_safe_globals([slice])  # needed for e3nn constants.pt under PyTorch 2.6+
from ase import Atoms
from e3nn import o3
from mace.calculators import MACECalculator
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.blocks import NonLinearReadoutBlock
from mace.tools import torch_geometric
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

HOME      = '/home/energy/s242862'
MODEL     = f'{HOME}/mace_t1x_p10_compiled.model'
CACHE_DIR = f'{HOME}/delta_cache'
OUT_DIR   = f'{HOME}/delta_head'

# Delta head irreps — match readouts[-1] of mace_t1x_p10
HIDDEN_IRREPS = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS    = o3.Irreps("64x0e")
NODE_FEATS_OFFSET = 1024   # skip first interaction's 1024x0e output


def build_batch(structs, z_table, r_max, device):
    atoms_list = [Atoms(numbers=s['atomic_numbers'].tolist(), positions=s['positions']) for s in structs]
    configs    = [config_from_atoms(a) for a in atoms_list]
    data_list  = [AtomicData.from_config(c, z_table=z_table, cutoff=r_max, heads=['Default'])
                  for c in configs]
    batch = torch_geometric.Batch.from_data_list(data_list).to(device)
    return batch


def forward_delta(batch, model, delta_head, training=True):
    # enable_grad needed even during val: autograd.grad requires a live graph
    with torch.enable_grad():
        batch.positions.requires_grad_(True)
        # TorchScript compiled model requires Dict[str, Tensor], not Batch
        batch_dict = {key: batch[key] for key in batch.keys}
        out = model(batch_dict, training=False, compute_force=False,
                    compute_virials=False, compute_stress=False)
        node_feats = out['node_feats'][:, NODE_FEATS_OFFSET:]      # [N_atoms, 16384]
        per_atom   = delta_head(node_feats).squeeze(-1)             # [N_atoms]
        per_struct = torch.zeros(batch.num_graphs, device=per_atom.device)
        per_struct.scatter_add_(0, batch.batch, per_atom)           # [B]
        forces = -torch.autograd.grad(
            per_struct.sum(), batch_dict['positions'],
            create_graph=training,
        )[0]                                                        # [N_atoms, 3]
    return per_struct, forces


def compute_loss(pred_e, pred_f, structs, batch_obj, huber_delta, force_weight):
    target_e = torch.tensor([s['delta_eV'] for s in structs],
                            dtype=torch.float32, device=pred_e.device)
    loss_e = F.huber_loss(pred_e, target_e, delta=huber_delta, reduction='mean')

    # Build per-atom force targets and mask
    target_f_list, mask_list = [], []
    for s in structs:
        n = len(s['atomic_numbers'])
        if s['delta_forces'] is not None:
            target_f_list.append(torch.tensor(s['delta_forces'], dtype=torch.float32))
            mask_list.extend([True] * n)
        else:
            target_f_list.append(torch.zeros(n, 3, dtype=torch.float32))
            mask_list.extend([False] * n)

    mask     = torch.tensor(mask_list, device=pred_e.device)
    target_f = torch.cat(target_f_list, dim=0).to(pred_e.device)

    if mask.any():
        loss_f = F.huber_loss(pred_f[mask], target_f[mask], delta=huber_delta, reduction='mean')
    else:
        loss_f = torch.tensor(0.0, device=pred_e.device)

    loss = loss_e + force_weight * loss_f
    return loss, loss_e.item(), loss_f.item()


def run_epoch(data, model, delta_head, z_table, r_max, device, optimizer,
              batch_size, huber_delta, force_weight, training):
    if training:
        delta_head.train()
        random.shuffle(data)
    else:
        delta_head.eval()

    total_loss = total_e = total_f = 0.0
    n_batches = 0

    for i in range(0, len(data), batch_size):
        structs = data[i:i + batch_size]
        batch   = build_batch(structs, z_table, r_max, device)

        with torch.set_grad_enabled(training):
            pred_e, pred_f = forward_delta(batch, model, delta_head, training=training)
            loss, le, lf   = compute_loss(pred_e, pred_f, structs, batch,
                                          huber_delta, force_weight)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_e    += le
        total_f    += lf
        n_batches  += 1

    return total_loss / n_batches, total_e / n_batches, total_f / n_batches


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    train_data = torch.load(f'{CACHE_DIR}/train.pt')
    val_data   = torch.load(f'{CACHE_DIR}/val.pt')
    print(f'Train: {len(train_data)} geoms | Val pool: {len(val_data)} geoms')

    # Load frozen MACE
    print(f'Loading MACE from {MODEL}...')
    calc    = MACECalculator(model_paths=MODEL, device=str(device), default_dtype='float32')
    model   = calc.models[0]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    z_table = calc.z_table
    r_max   = float(model.r_max)
    print(f'MACE loaded. r_max={r_max}, z_table={z_table.zs}')

    # Delta head
    delta_head = NonLinearReadoutBlock(
        irreps_in  = HIDDEN_IRREPS,
        MLP_irreps = MLP_IRREPS,
        gate       = F.silu,
    ).to(device)
    n_params = sum(p.numel() for p in delta_head.parameters())
    print(f'Delta head: {n_params} parameters')

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        delta_head.load_state_dict(state)
        print(f'Resumed from {args.resume}')

    optimizer = Adam(delta_head.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.5)

    os.makedirs(OUT_DIR, exist_ok=True)

    wandb.init(
        project='transition1x-delta',
        entity='s242862-danmarks-tekniske-universitet-dtu',
        config={
            'lr':            args.lr,
            'batch_size':    args.batch_size,
            'val_samples':   args.val_samples,
            'max_epochs':    args.epochs,
            'huber_delta':   args.huber_delta,
            'force_weight':  args.force_weight,
            'scheduler_patience': args.patience,
            'n_train':       len(train_data),
            'train_samples': args.train_samples or len(train_data),
            'n_val_pool':    len(val_data),
            'hidden_irreps': str(HIDDEN_IRREPS),
            'MLP_irreps':    str(MLP_IRREPS),
            'resume':        args.resume,
        },
    )

    val_sample   = random.sample(val_data, min(args.val_samples, len(val_data)))
    val_f_sample = [s for s in val_data if s['delta_forces'] is not None]
    print(f'Val sample fixed: {len(val_sample)} geoms | Force val: {len(val_f_sample)} geoms')

    best_force_loss = float('inf')

    for epoch in range(1, args.epochs + 1):

        epoch_data = (random.sample(train_data, args.train_samples)
                      if args.train_samples and args.train_samples < len(train_data)
                      else train_data)

        train_loss, train_e, train_f = run_epoch(
            epoch_data, model, delta_head, z_table, r_max, device,
            optimizer, args.batch_size, args.huber_delta, args.force_weight,
            training=True,
        )
        val_loss, val_e, val_f = run_epoch(
            val_sample, model, delta_head, z_table, r_max, device,
            None, args.batch_size, args.huber_delta, args.force_weight,
            training=False,
        )
        _, val_f_e, val_f_f = run_epoch(
            val_f_sample, model, delta_head, z_table, r_max, device,
            None, args.batch_size, args.huber_delta, args.force_weight,
            training=False,
        )

        lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        wandb.log({
            'epoch':      epoch,
            'train_loss': train_loss,
            'train_e':    train_e,
            'train_f':    train_f,
            'val_loss':   val_loss,
            'val_e':      val_e,
            'val_f':      val_f,
            'val_f_e':    val_f_e,
            'val_f_f':    val_f_f,
            'lr':         lr,
        })

        print(f'Epoch {epoch:3d} | train={train_loss:.4f} (e={train_e:.4f} f={train_f:.4f}) '
              f'| val={val_loss:.4f} (e={val_e:.4f} f={val_f:.4f}) '
              f'| fval_f={val_f_f:.4f} | lr={lr:.2e}')

        if val_f_f < best_force_loss:
            best_force_loss = val_f_f
            torch.save(delta_head.state_dict(), f'{OUT_DIR}/delta_head_fw{args.force_weight:.2f}.pt')
            print(f'  -> Saved best model (val_force_loss={val_f_f:.4f})')

        if lr < 1e-6:
            print('Learning rate below 1e-6, stopping.')
            break

    # Final eval on full val set
    print('\nFinal evaluation on full val set...')
    full_val_loss, full_val_e, full_val_f = run_epoch(
        val_data, model, delta_head, z_table, r_max, device,
        None, args.batch_size, args.huber_delta, args.force_weight,
        training=False,
    )
    print(f'Full val: loss={full_val_loss:.4f} (e={full_val_e:.4f} f={full_val_f:.4f})')
    wandb.log({'final_val_loss': full_val_loss, 'final_val_e': full_val_e, 'final_val_f': full_val_f})
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',        type=int,   default=200)
    parser.add_argument('--batch-size',    type=int,   default=64)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--val-samples',   type=int,   default=1024)
    parser.add_argument('--huber-delta',   type=float, default=0.1)
    parser.add_argument('--force-weight',  type=float, default=1.0)
    parser.add_argument('--patience',      type=int,   default=10)
    parser.add_argument('--train-samples', type=int,   default=0,    help='Geoms per epoch (0 = full dataset)')
    parser.add_argument('--resume',        type=str,   default=None, help='Path to checkpoint .pt to fine-tune from')
    args = parser.parse_args()
    main(args)
