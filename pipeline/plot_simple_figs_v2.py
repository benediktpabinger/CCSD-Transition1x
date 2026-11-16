"""Second iteration of the simple figures: every reaction as its own point, one panel per model.

Freezing string + Sella only, against N_FOD at the stored transition state, closed-shell vs
broken-symmetry reactions (spin class as in plot_simple_figs.py). Lines are running statistics in a
sliding window of N_FOD +- 0.1 (at least 15 reactions in the window), no bins.

  v2_success   ticks at 100 (intended saddle) or 0 (not), running success rate
  v2_forces    model force error vs DFT at the searched saddle, one dot per reaction, running median
  v2_barriers  absolute barrier error vs DFT at the same structures, running median, 43 meV line

Output figures/paper/v2_*.{png,pdf}; the first iteration (simple_*) is left as it is.

    python pipeline/plot_simple_figs_v2.py
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

H = os.path.expanduser('~')
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

S = {r['rxn']: r for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{RES}/{n}.csv'))}
spin = {rx: ('cs' if r['group'] in ('baseline', 'closed') else 'bs') for rx, r in S.items()}
nfod = {rx: float(r['nfod']) for rx, r in S.items()}


def running(x, y, stat, half=0.1, nmin=15):
    xs, ys = [], []
    for g in GRIDX:
        sel = np.abs(x - g) <= half
        if sel.sum() >= nmin:
            xs.append(g)
            ys.append(stat(y[sel]))
    return np.array(xs), np.array(ys)


def panels(title):
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey=True)
    fig.suptitle(title, x=0.06, ha='left', fontsize=11)
    for ax, m in zip(axs.flat, M):
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(LABEL[m], fontsize=10, loc='left')
        ax.set_xlim(-0.03, 1.65)
    for ax in axs[1]:
        ax.set_xlabel(r'$N_\mathrm{FOD}$ at the stored transition state')
    return fig, axs


def save(fig, name):
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def per_model(rows_by_model, value, stat, kind):
    for ax, m in zip(AXS.flat, M):
        for sp, (lab, col) in SPIN.items():
            R = [r for r in rows_by_model[m] if spin[r['rxn']] == sp]
            x = np.array([nfod[r['rxn']] for r in R])
            y = np.array([value(r) for r in R])
            if kind == 'ticks':
                rng = np.random.default_rng(0)
                ax.scatter(x, y + rng.uniform(-2.5, 2.5, len(y)) + (1.5 if sp == 'bs' else -1.5), s=6, marker='|',
                           color=col, alpha=0.5, lw=0.8)
                xs, ys = running(x, y, np.mean)
            else:
                ax.scatter(x, y, s=7, color=col, alpha=0.35, lw=0)
                xs, ys = running(x, y, stat)
            ax.plot(xs, ys, color=col, lw=2.2, label=lab)
    AXS.flat[0].legend(frameon=False, fontsize=8.5, loc='best')


# success ------------------------------------------------------------------------------------
rows = {m: [dict(rxn=j['rxn'], ok=j['cls'] == 'route') for j in
            (json.load(open(f)) for f in glob.glob(f'{H}/neb_fsm/check_v2/{m}/*_fsm.json'))] for m in M}
fig, AXS = panels('Search success against multireference character, every search (freezing string + Sella)')
per_model(rows, lambda r: 100.0 * r['ok'], None, 'ticks')
for ax in AXS[:, 0]:
    ax.set_ylabel('Intended saddle found (%)\n(ticks: each search, top found, bottom not)')
AXS.flat[0].set_ylim(-8, 108)
save(fig, 'v2_success')

# forces -------------------------------------------------------------------------------------
rows = {m: [] for m in M}
for r in csv.DictReader(open(f'{RES}/d37_forces.csv')):
    rows[r['model']].append(r)
fig, AXS = panels('Model force error at the searched saddles, every reaction')
per_model(rows, lambda r: 1000 * float(r['f_err_mae']), np.median, 'dots')
AXS.flat[0].set_yscale('log')
for ax in AXS[:, 0]:
    ax.set_ylabel('Force error vs DFT\n(mean abs. component, meV/Å)')
save(fig, 'v2_forces')

# barriers -----------------------------------------------------------------------------------
rows = {m: [] for m in M}
for r in csv.DictReader(open(f'{RES}/d40_barriers.csv')):
    rows[r['model']].append(r)
fig, AXS = panels('Barrier error at the searched saddles, every reaction')
per_model(rows, lambda r: max(abs(float(r['error_mev'])), 0.1), np.median, 'dots')
AXS.flat[0].set_yscale('log')
for ax in AXS.flat:
    ax.axhline(43, color=INK2, lw=1, ls='--')
AXS.flat[1].text(1.62, 47, '43 meV', ha='right', va='bottom', fontsize=8, color=INK2)
for ax in AXS[:, 0]:
    ax.set_ylabel('|Barrier error| vs DFT (meV)')
save(fig, 'v2_barriers')
