"""
Compare barrier heights from three sources for all converged NEB reactions:
  1. Transition1x reference  (wB97X-D3/6-31G(d), optimised TS/reactant in HDF5)
  2. ORCA NEB reference      (wB97M-V/def2-TZVP, last 10 images of neb.db)
  3. MACE prediction         (trained on Transition1x, applied to NEB images)

Usage:
    python compare_barriers.py \
        --neb-dir  ~/orca_neb_results \
        --t1x-h5   ~/data/Transition1x.h5 \
        --model    ~/mace_t1x_p10_ep194.model \
        --output   ~/barrier_comparison_full.json \
        --plot     ~/barrier_comparison_full.png
"""
import argparse
import json
import os

import h5py
import numpy as np
import ase.db
from ase.io import read
from mace.calculators import MACECalculator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_t1x_barrier(h5, rxn_id):
    """Forward barrier (meV) from HDF5 optimised TS and reactant energies."""
    for split in ('test', 'val', 'train', 'data'):
        if split not in h5:
            continue
        for formula in h5[split].keys():
            if rxn_id in h5[split][formula]:
                grp = h5[split][formula][rxn_id]
                ts_e = float(grp['transition_state']['wB97x_6-31G(d).energy'][0])
                r_e  = float(grp['reactant']['wB97x_6-31G(d).energy'][0])
                return (ts_e - r_e) * 1000  # eV → meV
    return None


def orca_and_images_from_db(db_path):
    """ORCA forward barrier (meV) and last-10 NEB images from neb.db."""
    with ase.db.connect(db_path) as db:
        rows = list(db.select())
    if len(rows) < 10:
        return None, None
    images = [row.toatoms() for row in rows[-10:]]
    energies = [img.get_potential_energy() for img in images]
    return (max(energies) - energies[0]) * 1000, images


def mace_barrier_from_images(images, calc):
    """MACE forward barrier (meV) evaluated on the NEB images."""
    energies = []
    for img in images:
        atoms = img.copy()
        atoms.calc = calc
        energies.append(atoms.get_potential_energy())
    return (max(energies) - energies[0]) * 1000


def stats(pred, ref):
    err = np.array(pred) - np.array(ref)
    return {
        'n':           len(err),
        'rmse_meV':    float(np.sqrt(np.mean(err**2))),
        'mae_meV':     float(np.mean(np.abs(err))),
        'bias_meV':    float(np.mean(err)),
        'max_err_meV': float(np.max(np.abs(err))),
        'r2':          float(1 - np.sum(err**2) / np.sum((np.array(ref) - np.mean(ref))**2)),
    }


def main(args):
    print(f"Loading MACE model: {args.model}")
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    calc = MACECalculator(model_paths=args.model, device=device, default_dtype='float32')

    t1x_barriers, orca_barriers, mace_barriers, rxn_ids = [], [], [], []
    skipped = []

    with h5py.File(args.t1x_h5, 'r') as h5:
        for rxn in sorted(os.listdir(args.neb_dir)):
            d = os.path.join(args.neb_dir, rxn)
            db_path = os.path.join(d, 'neb.db')
            if not os.path.exists(os.path.join(d, 'converged')) or not os.path.exists(db_path):
                continue

            t1x_b = find_t1x_barrier(h5, rxn)
            if t1x_b is None:
                skipped.append((rxn, 'not found in T1x HDF5'))
                continue

            orca_b, images = orca_and_images_from_db(db_path)
            if orca_b is None:
                skipped.append((rxn, 'too few NEB images in db'))
                continue

            print(f"{rxn}  computing MACE...", flush=True)
            try:
                mace_b = mace_barrier_from_images(images, calc)
            except Exception as e:
                skipped.append((rxn, f'MACE error: {e}'))
                continue

            t1x_barriers.append(t1x_b)
            orca_barriers.append(orca_b)
            mace_barriers.append(mace_b)
            rxn_ids.append(rxn)
            print(f"{rxn}  T1x={t1x_b:7.0f}  ORCA={orca_b:7.0f}  MACE={mace_b:7.0f}  meV", flush=True)

    if not rxn_ids:
        print("No results — check paths.")
        return

    t1x  = np.array(t1x_barriers)
    orca = np.array(orca_barriers)
    mace = np.array(mace_barriers)

    s_orca_vs_t1x  = stats(orca, t1x)
    s_mace_vs_t1x  = stats(mace, t1x)
    s_mace_vs_orca = stats(mace, orca)

    print("\n=== Summary ===")
    print(f"Reactions included : {len(rxn_ids)}")
    print(f"Reactions skipped  : {len(skipped)}")
    header = f"{'Comparison':<25}  {'RMSE':>8}  {'MAE':>8}  {'Bias':>8}  {'MaxErr':>8}  {'R²':>6}"
    print(header)
    print("-" * len(header))
    for label, s in [
        ("ORCA vs T1x",  s_orca_vs_t1x),
        ("MACE vs T1x",  s_mace_vs_t1x),
        ("MACE vs ORCA", s_mace_vs_orca),
    ]:
        print(f"{label:<25}  {s['rmse_meV']:>7.1f}  {s['mae_meV']:>7.1f}  "
              f"{s['bias_meV']:>+7.1f}  {s['max_err_meV']:>7.1f}  {s['r2']:>6.4f}  meV")

    output = {
        'n_included': len(rxn_ids),
        'n_skipped':  len(skipped),
        'skipped':    skipped,
        'orca_vs_t1x':  s_orca_vs_t1x,
        'mace_vs_t1x':  s_mace_vs_t1x,
        'mace_vs_orca': s_mace_vs_orca,
        'reactions': [
            {'rxn': r, 't1x_meV': float(a), 'orca_meV': float(b), 'mace_meV': float(c)}
            for r, a, b, c in zip(rxn_ids, t1x, orca, mace)
        ],
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")

    # ── Plot ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Barrier Height Comparison — all converged NEB reactions', fontsize=13, fontweight='bold')

    plot_pairs = [
        (t1x,  orca, 'T1x  wB97X-D3/6-31G(d)',       'ORCA  wB97M-V/def2-TZVP',   'steelblue',  s_orca_vs_t1x),
        (t1x,  mace, 'T1x  wB97X-D3/6-31G(d)',       'MACE p10 (epoch 291)',        'darkorange', s_mace_vs_t1x),
        (orca, mace, 'ORCA  wB97M-V/def2-TZVP', 'MACE p10 (epoch 291)',        'seagreen',   s_mace_vs_orca),
    ]

    for ax, (x, y, xlabel, ylabel, color, s) in zip(axes, plot_pairs):
        ax.scatter(x, y, alpha=0.5, s=18, color=color, edgecolors='none')
        lo = min(x.min(), y.min()) - 300
        hi = max(x.max(), y.max()) + 300
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            f"RMSE={s['rmse_meV']:.0f} meV  MAE={s['mae_meV']:.0f} meV\n"
            f"bias={s['bias_meV']:+.0f} meV  R²={s['r2']:.3f}",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(args.plot, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {args.plot}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neb-dir', default='/home/energy/s242862/orca_neb_results')
    parser.add_argument('--t1x-h5',  default='/home/energy/s242862/data/Transition1x.h5')
    parser.add_argument('--model',   default='/home/energy/s242862/mace_t1x_p10_ep194.model')
    parser.add_argument('--output',  default='/home/energy/s242862/barrier_comparison_full.json')
    parser.add_argument('--plot',    default='/home/energy/s242862/barrier_comparison_full.png')
    main(parser.parse_args())
