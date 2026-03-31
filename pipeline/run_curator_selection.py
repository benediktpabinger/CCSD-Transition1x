"""
LCMD+GRAD(RP) active learning selection using CURATOR's lcmd_greedy on SchNetPack PaiNN.

Uses AtomsDataModule (same pipeline as training) for efficient batched GPU inference.
Extracts 64-dim last-layer features, runs LCMD to select n_select configs,
writes geometries + metadata to selected_configs.json.

Usage:
    python run_curator_selection.py \
        --checkpoint ~/painn_results/checkpoints/best.ckpt \
        --db         ~/data/transition1x_train.db \
        --n-select   50 \
        --output     ~/curator_results/round1 \
        --batch-size 512 \
        --num-workers 4 \
        --gpu

    # Smoke test (login node, no GPU):
    python run_curator_selection.py ... --max-configs 500 --num-workers 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import ase.db
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
from schnetpack import properties

from curator.select.select import lcmd_greedy
from curator.select.kernel import FeatureKernelMatrix


def load_model(ckpt_path, device):
    """Load SchNetPack Lightning checkpoint, return NeuralNetworkPotential (no Forces)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    model_state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=5.0)
    representation = spk.representation.PaiNN(
        n_atom_basis=128,
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(5.0),
    )
    pred_energy = spk.atomistic.Atomwise(n_in=128, output_key='energy')
    model = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets('energy', add_mean=True, add_atomrefs=False),
        ],
    )
    current = model.state_dict()
    for k, v in model_state.items():
        if k in current and v.shape != current[k].shape:
            model_state[k] = v.reshape(current[k].shape)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


def extract_features(model, db_path, batch_size, device, max_configs=None, num_workers=4):
    """
    Extract 64-dim per-molecule features using AtomsDataModule (same as training).

    Returns:
        features:   (N_configs, 64) float32 tensor
        db_ids:     list of db row ids (1-based)
    """
    first_linear = model.output_modules[0].outnet[0]  # Linear(128 → 64)

    from schnetpack.data import AtomsLoader
    full_dataset = ASEAtomsData(
        datapath=db_path,
        transforms=[
            trn.ASENeighborList(cutoff=5.0),
            trn.CastTo32(),
        ],
    )
    if max_configs:
        full_dataset = torch.utils.data.Subset(full_dataset, list(range(min(max_configs, len(full_dataset)))))

    loader = AtomsLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    total = len(full_dataset)
    print(f"Extracting features: {total} configs, batch_size={batch_size}, device={device}")

    all_features = []
    n_processed = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            for m in model.input_modules:
                batch = m(batch)
            batch = model.representation(batch)

            scalar = batch['scalar_representation']     # (N_atoms_total, 128)
            hidden = first_linear(scalar)               # (N_atoms_total, 64)
            image_idx = batch[properties.idx_m]         # (N_atoms_total,)

            n_mols = image_idx.max().item() + 1
            mol_feat = torch.zeros(n_mols, 64, dtype=hidden.dtype, device=device)
            mol_feat.scatter_add_(0, image_idx.unsqueeze(1).expand_as(hidden), hidden)

            all_features.append(mol_feat.cpu())
            n_processed += n_mols
            if n_processed % 50000 == 0:
                print(f"  {n_processed}/{total} ...")

    features = torch.cat(all_features, dim=0).float()
    print(f"Done. Feature matrix: {features.shape}")

    # db_ids: 1-based row indices
    db_ids = list(range(1, len(features) + 1))
    return features, db_ids


def run_lcmd(features, n_select):
    print(f"Running LCMD: {features.shape[0]} configs → select {n_select} ...")
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)
    features = features.double()
    matrix = FeatureKernelMatrix(features.unsqueeze(0))
    selected = lcmd_greedy(matrix=matrix, batch_size=n_select, n_train=0)
    return selected.tolist()


def save_results(selected_indices, db_ids, db_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    selected_db_ids = [db_ids[i] for i in selected_indices]
    results = []
    selected_atoms_list = []

    with ase.db.connect(db_path) as db:
        for rank, (pool_idx, db_id) in enumerate(zip(selected_indices, selected_db_ids)):
            row = db.get(db_id)
            atoms = row.toatoms()
            e_raw = row.data.get('energy', float('nan')) if row.data else float('nan')
            e_dft = float(np.asarray(e_raw).flat[0])
            results.append({
                'rank':      rank,
                'pool_idx':  pool_idx,
                'db_id':     db_id,
                'n_atoms':   len(atoms),
                'e_dft_eV':  e_dft,
                'symbols':   atoms.get_chemical_formula(),
            })
            selected_atoms_list.append(atoms)

    json_path = os.path.join(output_dir, 'selected_configs.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    from ase.io import write as ase_write
    ase_write(os.path.join(output_dir, 'selected_configs.xyz'), selected_atoms_list)

    print(f"\nSaved {len(results)} configs to {output_dir}/")
    print(f"  selected_configs.json")
    print(f"  selected_configs.xyz")


def main(args):
    if args.gpu and torch.cuda.is_available():
        try:
            torch.cuda.init()
            device = torch.device('cuda')
        except RuntimeError as e:
            print(f"CUDA init failed ({e}), falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    features_path = os.path.join(args.output, 'features.pt')
    ids_path = os.path.join(args.output, 'db_ids.pt')

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)

    if os.path.exists(features_path) and not args.recompute:
        print(f"Loading cached features ...")
        features = torch.load(features_path)
        db_ids = torch.load(ids_path)
    else:
        features, db_ids = extract_features(
            model, args.db, args.batch_size, device,
            max_configs=args.max_configs, num_workers=args.num_workers)
        torch.save(features, features_path)
        torch.save(db_ids, ids_path)
        print(f"Cached to {features_path}")

    selected_indices = run_lcmd(features.cpu(), args.n_select)
    print(f"Selected: {selected_indices}")
    save_results(selected_indices, db_ids, args.db, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--db',          required=True)
    parser.add_argument('--n-select',    type=int, default=50)
    parser.add_argument('--output',      required=True)
    parser.add_argument('--batch-size',  type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-configs', type=int, default=None)
    parser.add_argument('--recompute',   action='store_true')
    parser.add_argument('--gpu',         action='store_true')
    args = parser.parse_args()
    main(args)
