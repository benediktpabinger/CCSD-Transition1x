"""
Compare ωB97M-V/def2-TZVP (ORCA NEB) barrier heights vs wB97x/6-31G(d) (Transition1x).

For each converged reaction in orca_neb_results/ and orca_neb_val_results/:
  - ORCA barrier: E(transition_state.xyz) - E(reactant.xyz)
  - T1x barrier:  max(wB97x energies along IRC) - energies[0]

Usage:
    python compare_barriers.py \
        --h5file ~/data/Transition1x.h5 \
        --test-dir ~/orca_neb_results \
        --val-dir  ~/orca_neb_val_results \
        --output   ~/barrier_comparison.png
"""
import argparse
import os
import json

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read


def read_orca_barrier(rxn_dir):
    """Returns (E_TS - E_reactant) in eV from ORCA NEB output."""
    ts_file = os.path.join(rxn_dir, 'transition_state.xyz')
    r_file  = os.path.join(rxn_dir, 'reactant.xyz')
    if not os.path.exists(ts_file) or not os.path.exists(r_file):
        return None
    ts  = read(ts_file)
    r   = read(r_file)
    if ts.calc is None or r.calc is None:
        # Read energy from comment line manually
        with open(ts_file) as f:
            lines = f.readlines()
        # extxyz: second line has energy=...
        e_ts = None
        for tok in lines[1].split():
            if tok.startswith('energy='):
                e_ts = float(tok.split('=')[1])
        with open(r_file) as f:
            lines = f.readlines()
        e_r = None
        for tok in lines[1].split():
            if tok.startswith('energy='):
                e_r = float(tok.split('=')[1])
        if e_ts is None or e_r is None:
            return None
        return e_ts - e_r
    return ts.get_potential_energy() - r.get_potential_energy()


def get_t1x_barrier(h5file, split, rxn_id):
    """Returns E_TS - E_reactant in eV using the optimized TS/reactant subgroups."""
    with h5py.File(h5file, 'r') as f:
        grp = f[split]
        for formula in grp:
            if rxn_id in grp[formula]:
                rxn = grp[formula][rxn_id]
                e_ts  = float(rxn['transition_state']['wB97x_6-31G(d).energy'][0])
                e_r   = float(rxn['reactant']['wB97x_6-31G(d).energy'][0])
                return e_ts - e_r
    return None


def collect(neb_dir, h5file, split):
    rxns = [d for d in os.listdir(neb_dir)
            if os.path.isfile(os.path.join(neb_dir, d, 'converged'))]
    orca, t1x, ids = [], [], []
    for rxn in rxns:
        b_orca = read_orca_barrier(os.path.join(neb_dir, rxn))
        b_t1x  = get_t1x_barrier(h5file, split, rxn)
        if b_orca is None or b_t1x is None:
            continue
        orca.append(b_orca)
        t1x.append(b_t1x)
        ids.append(rxn)
        if len(ids) % 50 == 0:
            print(f"  {split}: {len(ids)} reactions processed...")
    return np.array(orca), np.array(t1x), ids


def main(args):
    print("Collecting test split...")
    orca_test, t1x_test, ids_test = collect(args.test_dir, args.h5file, 'test')
    print(f"  {len(ids_test)} converged test reactions")

    print("Collecting val split...")
    orca_val, t1x_val, ids_val = collect(args.val_dir, args.h5file, 'val')
    print(f"  {len(ids_val)} converged val reactions")

    orca_all = np.concatenate([orca_test, orca_val])
    t1x_all  = np.concatenate([t1x_test,  t1x_val])
    ids_all    = ids_test + ids_val
    splits_all = ['test']*len(ids_test) + ['val']*len(ids_val)

    # Stats
    delta = orca_all - t1x_all
    mae   = np.mean(np.abs(delta))
    rmse  = np.sqrt(np.mean(delta**2))
    bias  = np.mean(delta)
    ss_res = np.sum(delta**2)
    ss_tot = np.sum((orca_all - np.mean(orca_all))**2)
    r2 = 1 - ss_res / ss_tot
    pearson_r = np.corrcoef(t1x_all, orca_all)[0, 1]
    print(f"\n=== Barrier comparison (ωB97M-V vs wB97x) ===")
    print(f"N reactions:  {len(orca_all)}")
    print(f"MAE:          {mae*1000:.1f} meV  ({mae*23.06:.2f} kcal/mol)")
    print(f"RMSE:         {rmse*1000:.1f} meV  ({rmse*23.06:.2f} kcal/mol)")
    print(f"Bias (ωB97M-V - wB97x): {bias*1000:.1f} meV  ({bias*23.06:.2f} kcal/mol)")
    print(f"Pearson r:    {pearson_r:.4f}")
    print(f"R²:           {r2:.4f}")
    print(f"T1x barrier range: {t1x_all.min():.2f} – {t1x_all.max():.2f} eV")
    print(f"ORCA barrier range: {orca_all.min():.2f} – {orca_all.max():.2f} eV")

    # Save raw data
    out_json = os.path.splitext(args.output)[0] + '.json'
    with open(out_json, 'w') as f:
        json.dump({
            'rxn_ids': ids_all,
            'splits':  splits_all,
            'orca_barriers_eV': orca_all.tolist(),
            't1x_barriers_eV':  t1x_all.tolist(),
        }, f, indent=2)
    print(f"Raw data saved to {out_json}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity plot
    ax = axes[0]
    lim = [min(t1x_all.min(), orca_all.min()) - 0.1,
           max(t1x_all.max(), orca_all.max()) + 0.1]
    ax.scatter(t1x_all, orca_all, alpha=0.4, s=10, color='steelblue')
    ax.plot(lim, lim, 'k--', lw=1, label='y=x')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('wB97x/6-31G(d) barrier [eV]')
    ax.set_ylabel('ωB97M-V/def2-TZVP barrier [eV]')
    ax.set_title(f'Barrier parity  (N={len(orca_all)}, MAE={mae*1000:.0f} meV)')
    ax.legend()

    # Difference histogram
    ax = axes[1]
    ax.hist(delta * 1000, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='k', lw=1, ls='--')
    ax.axvline(bias * 1000, color='red', lw=1.5, ls='-', label=f'bias={bias*1000:.0f} meV')
    ax.set_xlabel('Δ barrier [meV]  (ωB97M-V − wB97x)')
    ax.set_ylabel('Count')
    ax.set_title('Barrier difference distribution')
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file',   required=True)
    parser.add_argument('--test-dir', default='/home/energy/s242862/orca_neb_results')
    parser.add_argument('--val-dir',  default='/home/energy/s242862/orca_neb_val_results')
    parser.add_argument('--output',   default='/home/energy/s242862/barrier_comparison.png')
    args = parser.parse_args()
    main(args)
