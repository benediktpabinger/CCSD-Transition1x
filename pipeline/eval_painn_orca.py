"""
Evaluate PaiNN barrier predictions against ωB97M-V/def2-TZVP ORCA NEB reference.

For each converged ORCA NEB reaction:
  1. Load NEB images from neb.db (ωB97M-V/def2-TZVP energies)
  2. Run PaiNN on each image -> predicted energy
  3. Compare predicted barrier vs ORCA reference barrier

Usage:
    python eval_painn_orca.py \
        --orca-dir   ~/orca_neb_results \
        --checkpoint ~/painn_results_v2/checkpoints/best.ckpt \
        --n-images   10 \
        --output     ~/eval_painn_orca.json
"""
import argparse
import json
import os

import numpy as np
import torch
import ase.db

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack import properties

EV_TO_KCAL = 23.0609


def load_painn(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    model_state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}

    model = spk.model.NeuralNetworkPotential(
        representation=spk.representation.PaiNN(
            n_atom_basis=128, n_interactions=3,
            radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=5.0),
            cutoff_fn=spk.nn.CosineCutoff(5.0)),
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[spk.atomistic.Atomwise(n_in=128, output_key='energy')],
        postprocessors=[trn.CastTo64(),
                        trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)],
    )
    current = model.state_dict()
    for k, v in model_state.items():
        if k in current and v.shape != current[k].shape:
            model_state[k] = v.reshape(current[k].shape)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    model.to(device)
    return model


def predict_energy(model, atoms, converter, device):
    batch = converter(atoms)
    batch = {k: v.unsqueeze(0) if v.dim() == 0 else v for k, v in batch.items()}
    batch[properties.idx_m] = torch.zeros(batch[properties.n_atoms].item(), dtype=torch.long)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.enable_grad():
        result = model(batch)
    return float(result['energy'].cpu().detach().item())


def compute_barrier(energies):
    return (np.max(energies) - energies[0]) * EV_TO_KCAL


def main(args):
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            device = torch.device('cuda')
        except RuntimeError:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    print(f"Loading PaiNN from {args.checkpoint} ...")
    model = load_painn(args.checkpoint, device)

    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0),
        dtype=torch.float32,
        device=device,
    )

    # Get all converged reactions
    rxns = sorted([
        d for d in os.listdir(args.orca_dir)
        if os.path.isfile(os.path.join(args.orca_dir, d, 'converged'))
    ])
    print(f"Found {len(rxns)} converged reactions\n")

    results = []
    for i, rxn in enumerate(rxns):
        db_path = os.path.join(args.orca_dir, rxn, 'neb.db')
        db = ase.db.connect(db_path)
        total = db.count()
        if total < args.n_images:
            continue

        images = [db.get_atoms(id=total - args.n_images + j + 1) for j in range(args.n_images)]
        orca_energies = np.array([a.get_potential_energy() for a in images])
        painn_energies = np.array([predict_energy(model, a, converter, device) for a in images])

        barrier_orca  = compute_barrier(orca_energies)
        barrier_painn = compute_barrier(painn_energies)
        err = barrier_painn - barrier_orca

        results.append({
            'rxn': rxn,
            'barrier_orca':  barrier_orca,
            'barrier_painn': barrier_painn,
            'error':         err,
        })
        print(f"  [{i+1}/{len(rxns)}] {rxn}: ORCA={barrier_orca:.2f}  PaiNN={barrier_painn:.2f}  err={err:+.2f} kcal/mol")

    errs = np.array([r['error'] for r in results])
    print(f"\n{'='*55}")
    print(f"  N   = {len(results)} reactions")
    print(f"  MAE = {np.abs(errs).mean():.2f} kcal/mol")
    print(f"  RMSE= {np.sqrt((errs**2).mean()):.2f} kcal/mol")
    print(f"  Bias= {errs.mean():+.2f} kcal/mol")
    print(f"{'='*55}")

    with open(args.output, 'w') as f:
        json.dump({
            'n': len(results),
            'mae':  float(np.abs(errs).mean()),
            'rmse': float(np.sqrt((errs**2).mean())),
            'bias': float(errs.mean()),
            'results': results,
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orca-dir',   required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n-images',   type=int, default=10)
    parser.add_argument('--output',     required=True)
    args = parser.parse_args()
    main(args)
