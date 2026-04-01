"""
Evaluate the delta model on the 50 val reactions.

For each val reaction:
  1. Load the converged CCSD NEB images from ccsd_pyscf_results/{rxn}/neb.db
  2. Run PaiNN on each image → E_DFT_predicted + 64-dim features
  3. Apply delta model → delta_predicted
  4. E_final = E_DFT_predicted + delta_predicted
  5. Compute barrier = max(E) - E[reactant]
  6. Compare predicted barrier to CCSD reference barrier

Reports MAE in kcal/mol for:
  - Plain PaiNN (DFT-level)
  - PaiNN + delta model (CCSD-level)

Usage:
    python evaluate_delta_model.py \
        --val-list    ~/ccsd_dataset/val_reactions.txt \
        --ccsd-dir    ~/ccsd_pyscf_results \
        --checkpoint  ~/painn_results/checkpoints/best.ckpt \
        --delta-model ~/curator_results/round1/delta_model.json \
        --n-images    10
"""
import argparse
import json
import os

import numpy as np
import torch
import ase.db
from ase import units

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack import properties
from schnetpack.data import ASEAtomsData, AtomsLoader


EV_TO_KCAL = 23.0609   # 1 eV = 23.0609 kcal/mol


# ── model loading (same as run_curator_selection.py) ──────────────────────────

def load_painn(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    model_state = {k[len('model.'):]: v for k, v in state.items()
                   if k.startswith('model.')}

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


def load_delta_model(delta_json_path):
    with open(delta_json_path) as f:
        d = json.load(f)
    coef      = np.array(d['coef'])       # (64,)
    intercept = float(d['intercept'])
    X_mean    = np.array(d['X_mean'])     # (64,)
    X_std     = np.array(d['X_std'])      # (64,)
    return coef, intercept, X_mean, X_std


# ── feature extraction hook ───────────────────────────────────────────────────

def get_features_and_energy(model, batch, device):
    """Returns (energy_eV, features_64) for a batch of one molecule."""
    features_out = {}

    first_linear = model.output_modules[0].outnet[0]   # Linear(128→64)

    def hook(module, inp, out):
        features_out['hidden'] = out      # (N_atoms, 64) output of Linear(128→64)

    h = first_linear.register_forward_hook(hook)

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        result = model(batch)

    h.remove()

    energy = float(result['energy'].cpu().item())   # eV

    hidden = features_out['hidden']                 # (N_atoms, 64)
    idx_m  = batch[properties.idx_m]
    n_mols = int(idx_m.max().item()) + 1
    mol_features = torch.zeros(n_mols, hidden.shape[1], device='cpu')
    mol_features.scatter_add_(0, idx_m.unsqueeze(1).expand_as(hidden).cpu(), hidden.cpu())
    features = mol_features[0].numpy()              # (64,)

    return energy, features


def predict_delta(features, coef, intercept, X_mean, X_std):
    f_norm = (features - X_mean) / X_std
    return float(np.dot(coef, f_norm) + intercept)


# ── per-reaction evaluation ───────────────────────────────────────────────────

def evaluate_reaction(rxn, ccsd_dir, model, delta_params, device, n_images, converter):
    rxn_dir = os.path.join(ccsd_dir, rxn)
    db_path = os.path.join(rxn_dir, 'neb.db')
    if not os.path.exists(db_path):
        return None

    # Read final n_images from neb.db (last rows = converged NEB)
    db = ase.db.connect(db_path)
    total = db.count()
    if total < n_images:
        return None

    images = [db.get_atoms(id=total - n_images + i + 1) for i in range(n_images)]

    # CCSD reference energies (stored as calc.results['energy'])
    ccsd_energies = []
    for atoms in images:
        e = atoms.get_potential_energy()
        ccsd_energies.append(e)
    ccsd_energies = np.array(ccsd_energies)

    # PaiNN predictions + delta
    painn_energies  = []
    final_energies  = []
    coef, intercept, X_mean, X_std = delta_params

    for atoms in images:
        batch = converter(atoms)
        # add batch dim
        batch = {k: v.unsqueeze(0) if v.dim() == 0 else v for k, v in batch.items()}
        batch[properties.idx_m] = torch.zeros(batch[properties.n_atoms].item(), dtype=torch.long)

        e_dft, feats = get_features_and_energy(model, batch, device)
        delta         = predict_delta(feats, coef, intercept, X_mean, X_std)
        e_final       = e_dft + delta

        painn_energies.append(e_dft)
        final_energies.append(e_final)

    painn_energies = np.array(painn_energies)
    final_energies = np.array(final_energies)

    # Barriers (relative to first image = reactant)
    barrier_ccsd    = (ccsd_energies.max()  - ccsd_energies[0])  * EV_TO_KCAL
    barrier_painn   = (painn_energies.max() - painn_energies[0]) * EV_TO_KCAL
    barrier_delta   = (final_energies.max() - final_energies[0]) * EV_TO_KCAL

    return {
        'rxn':            rxn,
        'barrier_ccsd':   barrier_ccsd,
        'barrier_painn':  barrier_painn,
        'barrier_delta':  barrier_delta,
        'err_painn':      barrier_painn - barrier_ccsd,
        'err_delta':      barrier_delta  - barrier_ccsd,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            device = torch.device('cuda')
        except RuntimeError as e:
            print(f"CUDA init failed ({e}), falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    print(f"Loading PaiNN from {args.checkpoint} ...")
    model = load_painn(args.checkpoint, device)

    print(f"Loading delta model from {args.delta_model} ...")
    delta_params = load_delta_model(args.delta_model)

    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0),
        dtype=torch.float32,
        device=device,
    )

    with open(args.val_list) as f:
        val_rxns = [l.strip() for l in f if l.strip()]

    print(f"\nEvaluating {len(val_rxns)} val reactions ...")
    results = []
    for rxn in val_rxns:
        r = evaluate_reaction(rxn, args.ccsd_dir, model, delta_params,
                              device, args.n_images, converter)
        if r is None:
            print(f"  {rxn}: SKIP (missing or incomplete)")
            continue
        results.append(r)
        print(f"  {rxn}: CCSD={r['barrier_ccsd']:.2f}  PaiNN={r['barrier_painn']:.2f}  "
              f"Delta={r['barrier_delta']:.2f}  err_painn={r['err_painn']:+.2f}  "
              f"err_delta={r['err_delta']:+.2f} kcal/mol")

    if not results:
        print("No results.")
        return

    errs_painn = np.array([r['err_painn'] for r in results])
    errs_delta = np.array([r['err_delta']  for r in results])

    print(f"\n{'='*55}")
    print(f"  N = {len(results)} reactions")
    print(f"  MAE  PaiNN only : {np.abs(errs_painn).mean():.2f} kcal/mol")
    print(f"  MAE  +delta     : {np.abs(errs_delta).mean():.2f} kcal/mol")
    print(f"  RMSE PaiNN only : {np.sqrt((errs_painn**2).mean()):.2f} kcal/mol")
    print(f"  RMSE +delta     : {np.sqrt((errs_delta**2).mean()):.2f} kcal/mol")
    print(f"{'='*55}")

    out_path = os.path.join(os.path.dirname(args.delta_model), 'evaluation.json')
    with open(out_path, 'w') as f:
        json.dump({'results': results,
                   'mae_painn':  float(np.abs(errs_painn).mean()),
                   'mae_delta':  float(np.abs(errs_delta).mean()),
                   'rmse_painn': float(np.sqrt((errs_painn**2).mean())),
                   'rmse_delta': float(np.sqrt((errs_delta**2).mean()))}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-list',   required=True)
    parser.add_argument('--ccsd-dir',   required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--delta-model',required=True)
    parser.add_argument('--n-images',   type=int, default=10)
    args = parser.parse_args()
    main(args)
