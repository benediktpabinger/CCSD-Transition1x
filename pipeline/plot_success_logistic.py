"""Search success against N_FOD as a logistic fit, one panel per model (iteration 3 of the success figure).

Freezing string + Sella, D35 check (pass = intended saddle). For closed-shell and broken-symmetry
reactions separately: P(pass) = 1 / (1 + exp(-(a + b * N_FOD))), maximum likelihood, fitted only over
the N_FOD range the class covers; 95 % band from 1000 bootstrap resamples of reactions. Points are
binned pass rates with 95 % Wilson intervals (bins of 0.2, at least 8 reactions).

Output figures/paper/v3_success.{png,pdf}

    python pipeline/plot_success_logistic.py
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
OUT = os.path.join(REPO, 'figures', 'paper')
INK, INK2, GRID = '#1f1f1e', '#5f5e58', '#e4e3df'
M = ('esen', 'uma-s', 'uma-m', 'mace-omol')
LABEL = {'esen': 'eSEN', 'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'mace-omol': 'MACE-OMol'}
SPIN = {'cs': ('Closed-shell', '#2a78d6', 'o'), 'bs': ('Broken-symmetry', '#eb6834', 's')}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})

S = {r['rxn']: r for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{RES}/{n}.csv'))}
spin = {rx: ('cs' if r['group'] in ('baseline', 'closed') else 'bs') for rx, r in S.items()}
nfod = {rx: float(r['nfod']) for rx, r in S.items()}


def fit(x, y):
    def nll(p):
        z = p[0] + p[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)
    return minimize(nll, [0.0, 0.0], method='BFGS').x


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


rng = np.random.default_rng(0)
fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey=True)
fig.suptitle('Chance that a converged search is the intended saddle (freezing string + Sella)', x=0.06, ha='left', fontsize=11)
for ax, m in zip(axs.flat, M):
    J = [json.load(open(f)) for f in glob.glob(f'{H}/neb_fsm/check_v2/{m}/*_fsm.json')]
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(LABEL[m], fontsize=10, loc='left')
    for sp, (lab, col, mk) in SPIN.items():
        R = [j for j in J if spin[j['rxn']] == sp]
        x = np.array([nfod[j['rxn']] for j in R])
        y = np.array([float(j['cls'] == 'route') for j in R])
        grid = np.linspace(x.min(), min(x.max(), 1.6), 120)
        a, b = fit(x, y)
        boots = []
        for _ in range(1000):
            i = rng.integers(0, len(x), len(x))
            pa, pb = fit(x[i], y[i])
            boots.append(1 / (1 + np.exp(-(pa + pb * grid))))
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        ax.fill_between(grid, 100 * lo, 100 * hi, color=col, alpha=0.18, lw=0)
        ax.plot(grid, 100 / (1 + np.exp(-(a + b * grid))), color=col, lw=2.2, label=lab)
        for b0 in np.arange(0, 1.6, 0.2):
            s = (x >= b0) & (x < b0 + 0.2)
            if s.sum() >= 8:
                k, n = y[s].sum(), s.sum()
                l, h = wilson(k, n)
                xm = x[s].mean()
                ax.errorbar(xm, 100 * k / n, yerr=[[100 * (k / n - l)], [100 * (h - k / n)]], fmt=mk, color=col,
                            ms=5, mec='white', mew=0.8, elinewidth=1, capsize=0)
    ax.set_xlim(-0.03, 1.65)
    ax.set_ylim(0, 100)
axs.flat[0].legend(frameon=False, fontsize=8.5, loc='lower left')
for ax in axs[:, 0]:
    ax.set_ylabel('Intended saddle found (%)')
for ax in axs[1]:
    ax.set_xlabel(r'$N_\mathrm{FOD}$ at the stored transition state')
fig.text(0.99, 0.005, 'Line: logistic fit with 95 % band; points: rates in bins of 0.2 with 95 % intervals.',
         ha='right', fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/v3_success.{ext}', dpi=200, bbox_inches='tight')
print('wrote v3_success')
