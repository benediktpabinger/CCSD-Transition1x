"""
Plot MACE vs wB97X-D3 vs wB97M-V on delta-experiment geometries.

Produces two figures:
  1. Energy profiles along the converged NEB path (last 10 images per reaction)
  2. Parity plots + error histograms across all 270 geometries

Usage:
    python plot_delta_eval.py \
        --input ~/delta_experiment/mace_eval.json \
        --out-dir ~/delta_experiment/plots
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS = {
    't1x':   ('steelblue',  'wB97X-D3 / 6-31G(d)'),
    'silver': ('seagreen',   'wB97M-V / def2-TZVP'),
    'mace':  ('darkorange', 'MACE p10 (epoch 291)'),
}


def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.array(a) - np.array(b))))


def bias(a, b):
    return float(np.mean(np.array(a) - np.array(b)))


# ── Figure 1: energy profiles ─────────────────────────────────────────────────

def plot_profiles(data, out_path):
    rxns = list(data.keys())
    fig, axes = plt.subplots(1, len(rxns), figsize=(5 * len(rxns), 4.5), sharey=False)
    if len(rxns) == 1:
        axes = [axes]
    fig.suptitle('Energy Profile — converged NEB path  (last 10 images)', fontsize=12, fontweight='bold')

    for ax, rxn in zip(axes, rxns):
        r = data[rxn]
        geoms = r['geometries']
        # last 10 images = last NEB iteration
        last = geoms[-10:]
        x = list(range(10))

        t1x    = [g['de_t1x_meV']    for g in last]
        silver = [g['de_silver_meV'] for g in last]
        mace   = [g['de_mace_meV']   for g in last]

        ax.plot(x, t1x,    '-o', color=COLORS['t1x'][0],    ms=5, lw=1.5, label=COLORS['t1x'][1])
        ax.plot(x, silver, '-s', color=COLORS['silver'][0], ms=5, lw=1.5, label=COLORS['silver'][1])
        ax.plot(x, mace,   '-^', color=COLORS['mace'][0],   ms=5, lw=1.5, label=COLORS['mace'][1])

        ax.set_title(rxn, fontsize=11)
        ax.set_xlabel('NEB image', fontsize=9)
        ax.set_ylabel('Relative energy (meV)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # annotate barrier
        ts = int(np.argmax(t1x))
        ax.annotate(
            f"TS: t1x={t1x[ts]:.0f}\nwB97M={silver[ts]:.0f}\nMACE={mace[ts]:.0f} meV",
            xy=(ts, t1x[ts]), xytext=(ts + 0.3, t1x[ts] * 0.85),
            fontsize=7, ha='left',
            arrowprops=dict(arrowstyle='->', color='grey', lw=0.8),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


# ── Figure 2: parity + error histograms ──────────────────────────────────────

def plot_parity_and_errors(data, out_path):
    # Pool all geometries across all reactions
    de_t1x_all    = []
    de_silver_all = []
    de_mace_all   = []

    for r in data.values():
        for g in r['geometries']:
            de_t1x_all.append(g['de_t1x_meV'])
            de_silver_all.append(g['de_silver_meV'])
            de_mace_all.append(g['de_mace_meV'])

    de_t1x    = np.array(de_t1x_all)
    de_silver = np.array(de_silver_all)
    de_mace   = np.array(de_mace_all)
    n = len(de_mace)

    pairs = [
        (de_t1x,    de_mace, COLORS['t1x'][1],    'MACE p10 (epoch 291)', COLORS['mace'][0]),
        (de_silver, de_mace, COLORS['silver'][1],  'MACE p10 (epoch 291)', COLORS['mace'][0]),
    ]
    err_pairs = [
        (de_mace - de_t1x,    f"MACE vs wB97X-D3\n(wB97X-D3/6-31G(d))",   COLORS['mace'][0]),
        (de_mace - de_silver, f"MACE vs wB97M-V\n(wB97M-V/def2-TZVP)",     COLORS['silver'][0]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f'MACE vs DFT — delta experiment geometries  (n={n}, 3 reactions)',
        fontsize=12, fontweight='bold',
    )

    # Row 0: parity plots
    for ax, (x, y, xlabel, ylabel, color) in zip(axes[0], pairs):
        ax.scatter(x, y, alpha=0.4, s=12, color=color, edgecolors='none')
        lo = min(x.min(), y.min()) - 200
        hi = max(x.max(), y.max()) + 200
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        r = rmse(y, x); m = mae(y, x); b = bias(y, x)
        r2 = float(1 - np.sum((y - x)**2) / np.sum((x - np.mean(x))**2))
        ax.set_title(f'RMSE={r:.0f} meV  MAE={m:.0f} meV\nbias={b:+.0f} meV  R²={r2:.3f}', fontsize=9)

    # Row 1: error histograms
    for ax, (err, label, color) in zip(axes[1], err_pairs):
        r = rmse(err, np.zeros_like(err)); m = mae(err, np.zeros_like(err)); b = bias(err, np.zeros_like(err))
        ax.hist(err, bins=40, color=color, alpha=0.8, edgecolor='white', linewidth=0.4)
        ax.axvline(0, color='black', lw=1.2, ls='--', label='zero')
        ax.axvline(b, color='red',   lw=1.5, ls='-',  label=f'bias={b:+.0f} meV')
        ax.set_xlabel('Error (meV)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.text(0.97, 0.95, f'RMSE={r:.0f} meV\nMAE={m:.0f} meV',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_profiles(data,            os.path.join(args.out_dir, 'delta_profiles.png'))
    plot_parity_and_errors(data,   os.path.join(args.out_dir, 'delta_parity.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default='/home/energy/s242862/delta_experiment/mace_eval.json')
    parser.add_argument('--out-dir', default='/home/energy/s242862/delta_experiment/plots')
    args = parser.parse_args()
    main(args)
