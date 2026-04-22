"""
Evaluate MACE model on Transition1x test set and converged NEB geometries.

Usage:
    python eval_mace.py \
        --model  ~/mace_t1x_p5.model \
        --test   ~/data/transition1x_test.xyz \
        --neb-dir ~/orca_neb_results \
        --output ~/eval_mace_p5.json \
        --n-test 5000
"""
import argparse
import json
import os
import random

import numpy as np
import ase.db
from ase.io import read
from mace.calculators import MACECalculator


def rmse(pred, ref):
    return float(np.sqrt(np.mean((np.array(pred) - np.array(ref)) ** 2)))


def eval_structures(structures, calc, label):
    e_pred, e_ref, f_pred, f_ref = [], [], [], []
    errors = 0
    for atoms in structures:
        ref_e = atoms.get_potential_energy()
        ref_f = atoms.get_forces()
        n = len(atoms)
        try:
            atoms.calc = calc
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            e_pred.append(e / n)
            e_ref.append(ref_e / n)
            f_pred.extend(f.flatten().tolist())
            f_ref.extend(ref_f.flatten().tolist())
        except Exception as ex:
            errors += 1
            continue

    rmse_e = rmse(e_pred, e_ref) * 1000  # meV/atom
    rmse_f = rmse(f_pred, f_ref) * 1000  # meV/Å
    print(f"[{label}] n={len(e_pred)} errors={errors}  "
          f"RMSE_E={rmse_e:.1f} meV/atom  RMSE_F={rmse_f:.1f} meV/Å")
    return {"n": len(e_pred), "errors": errors,
            "rmse_e_meV_per_atom": rmse_e, "rmse_f_meV_per_A": rmse_f}


def load_neb_ts(neb_dir, max_rxn=200):
    structs = []
    for rxn in sorted(os.listdir(neb_dir))[:max_rxn]:
        d = os.path.join(neb_dir, rxn)
        ts = os.path.join(d, 'transition_state.xyz')
        if os.path.exists(os.path.join(d, 'converged')) and os.path.exists(ts):
            try:
                structs.append(read(ts))
            except Exception:
                pass
    return structs


def eval_barriers(neb_dir, calc):
    """Compare MACE barrier heights vs ORCA NEB reference."""
    barrier_pred, barrier_ref, rxn_names = [], [], []
    for rxn in sorted(os.listdir(neb_dir)):
        d = os.path.join(neb_dir, rxn)
        db_path = os.path.join(d, 'neb.db')
        if not os.path.exists(os.path.join(d, 'converged')) or not os.path.exists(db_path):
            continue
        try:
            with ase.db.connect(db_path) as db:
                rows = list(db.select())
            if len(rows) < 10:
                continue
            images = [row.toatoms() for row in rows[-10:]]
            # ORCA reference barrier
            orca_energies = [img.get_potential_energy() for img in images]
            ref_barrier = (max(orca_energies) - orca_energies[0]) * 1000  # meV

            # MACE predicted barrier
            mace_energies = []
            for img in images:
                img.calc = calc
                mace_energies.append(img.get_potential_energy())
            pred_barrier = (max(mace_energies) - mace_energies[0]) * 1000  # meV

            barrier_pred.append(pred_barrier)
            barrier_ref.append(ref_barrier)
            rxn_names.append(rxn)
        except Exception:
            continue

    errors = np.array(barrier_pred) - np.array(barrier_ref)
    rmse_b = float(np.sqrt(np.mean(errors ** 2)))
    mae_b = float(np.mean(np.abs(errors)))
    print(f"[barriers] n={len(barrier_pred)}  RMSE={rmse_b:.1f} meV  MAE={mae_b:.1f} meV")
    return {"n": len(barrier_pred), "rmse_meV": rmse_b, "mae_meV": mae_b,
            "per_rxn": [{"rxn": r, "ref": ref, "pred": pred, "error": pred - ref}
                        for r, ref, pred in zip(rxn_names, barrier_ref, barrier_pred)]}


def main(args):
    print(f"Loading model: {args.model}")
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    calc = MACECalculator(model_paths=args.model, device=device, default_dtype='float32')

    results = {}

    # Test set
    print(f"Reading test set (first {args.n_test} structures)...")
    structs = read(args.test, index=f':{args.n_test}')
    results['test_set'] = eval_structures(structs, calc, 'test_set')

    # NEB transition states + barrier heights
    if args.neb_dir and os.path.isdir(args.neb_dir):
        print("Loading NEB transition states...")
        neb_structs = load_neb_ts(args.neb_dir)
        print(f"  Found {len(neb_structs)} converged TS geometries")
        if neb_structs:
            results['neb_ts'] = eval_structures(neb_structs, calc, 'neb_ts')

        print("Evaluating barrier heights vs ORCA reference...")
        results['barriers'] = eval_barriers(args.neb_dir, calc)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   required=True)
    parser.add_argument('--test',    required=True)
    parser.add_argument('--neb-dir', default=None)
    parser.add_argument('--output',  required=True)
    parser.add_argument('--n-test',  type=int, default=5000)
    args = parser.parse_args()
    main(args)
