"""Version 6: N_FOD and spin class at the transition state each model found (D45, D06): N_FOD from
results/fod_model_ts.csv, spin class from our DFT there (d37_forces.csv s2, broken-symmetry if > 0.05).

Version 5: titles in the wording of the paper outline, axis says Transition1x transition state.

Force and barrier errors at the searched saddles on a linear axis (iteration 4 of these figures).

Freezing string + Sella, one panel per model, closed-shell vs broken-symmetry. Every reaction as a dot;
running median (window N_FOD +- 0.1, at least 15 reactions) with the band between the 25th and 75th
percentile. Linear y axis capped at CAP; reactions above it are drawn as triangles at the top edge.

  v4_forces    force error vs DFT (D37, mean absolute component), meV/A, cap 100
  v4_barriers  absolute barrier error vs DFT (D40), meV, cap 100, dashed line at 43 meV

Output figures/paper/v6_*.{png,pdf}

    python pipeline/plot_errors_linear.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
OUT = os.path.join(REPO, 'figures', 'paper')
INK, INK2, GRID = '#1f1f1e', '#5f5e58', '#e4e3df'
M = ('esen', 'uma-s', 'uma-m', 'mace-omol')
LABEL = {'esen': 'eSEN', 'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'mace-omol': 'MACE-OMol'}
SPIN = {'cs': ('Closed-shell', '#2a78d6'), 'bs': ('Broken-symmetry', '#eb6834')}
GRIDX = np.linspace(0, 1.6, 81)
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})

nfod = {(r['model'], r['rxn']): float(r['nfod_model_ts']) for r in csv.DictReader(open(f'{RES}/fod_model_ts.csv'))}
spin = {(r['model'], r['rxn']): ('bs' if abs(float(r['s2'])) > 0.05 else 'cs')
        for r in csv.DictReader(open(f'{RES}/d37_forces.csv'))}


def running(x, y, half=0.1, nmin=15):
    out = []
    for g in GRIDX:
        sel = np.abs(x - g) <= half
        if sel.sum() >= nmin:
            out.append((g, *np.percentile(y[sel], [25, 50, 75])))
    return np.array(out).T if out else np.empty((4, 0))


def figure(rows, value, cap, ylabel, title, name, ref=None):
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey=True)
    fig.suptitle(title, x=0.06, ha='left', fontsize=11)
    for ax, m in zip(axs.flat, M):
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(LABEL[m], fontsize=10, loc='left')
        if ref:
            ax.axhline(ref, color=INK2, lw=1, ls='--')
        for sp, (lab, col) in SPIN.items():
            R = [r for r in rows if r['model'] == m and spin[(m, r['rxn'])] == sp]
            x = np.array([nfod[(m, r['rxn'])] for r in R])
            y = np.array([value(r) for r in R])
            inside = y <= cap
            ax.scatter(x[inside], y[inside], s=7, color=col, alpha=0.3, lw=0)
            ax.scatter(x[~inside], np.full((~inside).sum(), cap * 0.985), s=16, marker='^', color=col, alpha=0.7, lw=0)
            g, q1, med, q3 = running(x, y)
            ax.fill_between(g, q1, np.minimum(q3, cap), color=col, alpha=0.18, lw=0)
            ax.plot(g, med, color=col, lw=2.2, label=lab)
        ax.set_xlim(-0.05, 2.85)
        ax.set_ylim(0, cap)
    if ref:
        axs.flat[1].text(1.62, ref + 1.5, f'{ref} meV', ha='right', va='bottom', fontsize=8, color=INK2)
    axs.flat[0].legend(frameon=False, fontsize=8.5, loc='upper left')
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axs[1]:
        ax.set_xlabel(r"$N_\mathrm{FOD}$ at the model's transition state")
    fig.text(0.99, 0.005, f'Line: running median; band: middle half of reactions; triangles: reactions above {cap}.',
             ha='right', fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


forces = list(csv.DictReader(open(f'{RES}/d37_forces.csv')))
figure(forces, lambda r: 1000 * float(r['f_err_mae']), 100, 'Force error vs DFT\n(mean abs. component, meV/Å)',
       'Force error: model vs DFT at the structure the model identified as the transition state', 'v6_forces')
barriers = list(csv.DictReader(open(f'{RES}/d40_barriers.csv')))
figure(barriers, lambda r: abs(float(r['error_mev'])), 100, '|Barrier error| vs DFT (meV)',
       'Barrier error: model vs DFT at that structure and the model\'s reactant', 'v6_barriers', ref=43)
