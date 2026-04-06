"""
Train and evaluate delta model using vault NEB data.

For each vault reaction:
  1. Load CCSD NEB images from ccsd_pyscf_results/{rxn}/neb.db
  2. Run PaiNN on each image -> E_PaiNN + 64-dim features
  3. Delta = E_CCSD - E_PaiNN  (per image)

Split reactions: 200 train / 87 test
Train ridge regression: features -> delta
Evaluate barrier MAE on test set: PaiNN alone vs PaiNN + delta

Usage:
    python train_eval_delta_vault.py \
        --ccsd-dir   ~/ccsd_pyscf_results \
        --checkpoint ~/painn_results_v2/checkpoints/best.ckpt \
        --n-images   10 \
        --n-train    200 \
        --output     ~/delta_vault_results
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
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


def get_features_and_energy(model, batch, device):
    features_out = {}
    linear_128_64 = model.output_modules[0].outnet[0]  # Linear(128->64)

    def hook(module, inp, out):
        features_out['hidden'] = out

    h = linear_128_64.register_forward_hook(hook)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.enable_grad():
        result = model(batch)

    h.remove()

    energy = float(result['energy'].cpu().detach().item())
    hidden = features_out['hidden'].detach()
    idx_m = batch[properties.idx_m]
    mol_features = torch.zeros(1, hidden.shape[1], device='cpu')
    mol_features.scatter_add_(0, idx_m.unsqueeze(1).expand_as(hidden).cpu(), hidden.cpu())
    features = mol_features[0].numpy()

    return energy, features


def process_reaction(rxn, ccsd_dir, model, converter, device, n_images):
    db_path = os.path.join(ccsd_dir, rxn, 'neb.db')
    if not os.path.exists(db_path):
        return None

    db = ase.db.connect(db_path)
    total = db.count()
    if total < n_images:
        return None

    images = [db.get_atoms(id=total - n_images + i + 1) for i in range(n_images)]

    ccsd_energies = np.array([atoms.get_potential_energy() for atoms in images])

    painn_energies = []
    all_features = []

    for atoms in images:
        batch = converter(atoms)
        batch = {k: v.unsqueeze(0) if v.dim() == 0 else v for k, v in batch.items()}
        batch[properties.idx_m] = torch.zeros(batch[properties.n_atoms].item(), dtype=torch.long)

        e_painn, feats = get_features_and_energy(model, batch, device)
        painn_energies.append(e_painn)
        all_features.append(feats)

    painn_energies = np.array(painn_energies)
    all_features = np.array(all_features)  # (n_images, 64)
    deltas = ccsd_energies - painn_energies  # (n_images,)

    return {
        'rxn': rxn,
        'ccsd_energies': ccsd_energies,
        'painn_energies': painn_energies,
        'features': all_features,
        'deltas': deltas,
    }


def compute_barrier(energies):
    return (energies.max() - energies[0]) * EV_TO_KCAL


def main(args):
    os.makedirs(args.output, exist_ok=True)

    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            device = torch.device('cuda')
        except RuntimeError as e:
            print(f"CUDA init failed ({e}), using CPU")
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

    # Collect all vault reactions
    rxns = sorted([d for d in os.listdir(args.ccsd_dir)
                   if os.path.isdir(os.path.join(args.ccsd_dir, d))])
    print(f"Found {len(rxns)} vault reactions")

    # Process all reactions
    data = []
    for i, rxn in enumerate(rxns):
        print(f"  [{i+1}/{len(rxns)}] {rxn} ...", end=' ', flush=True)
        r = process_reaction(rxn, args.ccsd_dir, model, converter, device, args.n_images)
        if r is None:
            print("SKIP")
            continue
        data.append(r)
        print(f"delta_mean={r['deltas'].mean():.3f} eV")

    print(f"\nProcessed {len(data)} reactions successfully")

    # Split train/test
    np.random.seed(42)
    idx = np.random.permutation(len(data))
    train_idx = idx[:args.n_train]
    test_idx  = idx[args.n_train:]

    train_data = [data[i] for i in train_idx]
    test_data  = [data[i] for i in test_idx]
    print(f"Split: {len(train_data)} train, {len(test_data)} test reactions")

    # Build training arrays (all images from all train reactions)
    X_train = np.vstack([r['features'] for r in train_data])
    y_train = np.concatenate([r['deltas'] for r in train_data])
    print(f"Training on {len(X_train)} images, delta range: {y_train.min():.3f} to {y_train.max():.3f} eV")

    # Train ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    train_pred = ridge.predict(X_train_scaled)
    train_mae = np.abs(train_pred - y_train).mean() * 1000
    print(f"Train MAE: {train_mae:.1f} meV")

    # Save model
    joblib.dump({'ridge': ridge, 'scaler': scaler},
                os.path.join(args.output, 'delta_model_vault.pkl'))

    # Evaluate on test reactions
    print(f"\nEvaluating on {len(test_data)} test reactions ...")
    results = []
    for r in test_data:
        X_test = scaler.transform(r['features'])
        delta_pred = ridge.predict(X_test)

        corrected_energies = r['painn_energies'] + delta_pred

        barrier_ccsd   = compute_barrier(r['ccsd_energies'])
        barrier_painn  = compute_barrier(r['painn_energies'])
        barrier_delta  = compute_barrier(corrected_energies)

        results.append({
            'rxn':           r['rxn'],
            'barrier_ccsd':  barrier_ccsd,
            'barrier_painn': barrier_painn,
            'barrier_delta': barrier_delta,
            'err_painn':     barrier_painn - barrier_ccsd,
            'err_delta':     barrier_delta  - barrier_ccsd,
        })
        print(f"  {r['rxn']}: CCSD={barrier_ccsd:.2f}  PaiNN={barrier_painn:.2f}  "
              f"Delta={barrier_delta:.2f}  err_painn={barrier_painn-barrier_ccsd:+.2f}  "
              f"err_delta={barrier_delta-barrier_ccsd:+.2f} kcal/mol")

    errs_painn = np.array([r['err_painn'] for r in results])
    errs_delta = np.array([r['err_delta']  for r in results])

    print(f"\n{'='*55}")
    print(f"  N = {len(results)} test reactions")
    print(f"  MAE  PaiNN only : {np.abs(errs_painn).mean():.2f} kcal/mol")
    print(f"  MAE  +delta     : {np.abs(errs_delta).mean():.2f} kcal/mol")
    print(f"  RMSE PaiNN only : {np.sqrt((errs_painn**2).mean()):.2f} kcal/mol")
    print(f"  RMSE +delta     : {np.sqrt((errs_delta**2).mean()):.2f} kcal/mol")
    print(f"{'='*55}")

    out_path = os.path.join(args.output, 'evaluation_vault.json')
    with open(out_path, 'w') as f:
        json.dump({
            'n_train_reactions': len(train_data),
            'n_test_reactions':  len(test_data),
            'train_mae_meV':     float(train_mae),
            'mae_painn':         float(np.abs(errs_painn).mean()),
            'mae_delta':         float(np.abs(errs_delta).mean()),
            'rmse_painn':        float(np.sqrt((errs_painn**2).mean())),
            'rmse_delta':        float(np.sqrt((errs_delta**2).mean())),
            'results':           results,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ccsd-dir',   required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n-images',   type=int, default=10)
    parser.add_argument('--n-train',    type=int, default=200)
    parser.add_argument('--output',     required=True)
    args = parser.parse_args()
    main(args)
