"""Simple versions of the main figures: freezing string + Sella only, against N_FOD, closed-shell
against broken-symmetry reactions, no group names.

Spin class per reaction from our own stability analysis via the selection groups (D24, D29):
low-MR baseline and closed-shell twins are closed-shell, the twins' partners and the N_FOD >= 1
reactions are broken-symmetry. N_FOD at the stored transition state. Each line is the mean over
the four OMol25 models, the band their range; bins with fewer than 8 reactions are left out.

  simple_1_success   share of converged searches that are a saddle of the intended reaction (D35 check)
  simple_2_found     what the searches found, four models pooled
  simple_3_forces    model force error against DFT at the searched saddles (D37, mean absolute component)
  simple_4_barriers  share of barriers off by more than 43 meV at the searched saddles (D40)

Output figures/paper/simple_*.{png,pdf}

    python pipeline/plot_simple_figs.py
"""
import csv
import glob
import json
import os
from collections import Counter

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
SPIN = {'cs': ('Closed-shell', '#2a78d6'), 'bs': ('Broken-symmetry', '#eb6834')}
EDGES = [0, .1, .2, .5, .8, 9]
MID = np.arange(len(EDGES) - 1)
BLAB = ['0–0.1', '0.1–0.2', '0.2–0.5', '0.5–0.8', '≥0.8']
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})

S = {r['rxn']: r for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{RES}/{n}.csv'))}
spin = {rx: ('cs' if r['group'] in ('baseline', 'closed') else 'bs') for rx, r in S.items()}
nfod = {rx: float(r['nfod']) for rx, r in S.items()}
binof = lambda rx: int(np.searchsorted(EDGES, nfod[rx], side='right') - 1)   # noqa: E731


def axes(ylabel, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(MID, BLAB)
    ax.set_xlabel(r'$N_\mathrm{FOD}$ (multireference character) at the stored transition state')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5, loc='left')
    return fig, ax


def lines(ax, per_model, fmt=lambda v: v):
    """per_model[model][spin][bin] -> list of values; plots mean of model statistics and range."""
    for sp, (lab, col) in SPIN.items():
        xs, mean, lo, hi, ns = [], [], [], [], []
        for b in MID:
            vals = [per_model[m][sp].get(b) for m in M]
            if any(v is None or len(v) < 8 for v in vals):
                continue
            stats = [fmt(v) for v in vals]
            xs.append(b)
            mean.append(np.mean(stats))
            lo.append(min(stats))
            hi.append(max(stats))
            ns.append(int(np.mean([len(v) for v in vals])))
        ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(xs, mean, color=col, lw=2, marker='o' if sp == 'cs' else 's', ms=6, mec='white', mew=1, label=lab)
    ax.legend(frameon=False, fontsize=9)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


# 1 and 2: searches -----------------------------------------------------------------------
checks = {m: [json.load(open(f)) for f in glob.glob(f'{H}/neb_fsm/check_v2/{m}/*_fsm.json')] for m in M}
per = {m: {sp: {} for sp in SPIN} for m in M}
for m in M:
    for j in checks[m]:
        per[m][spin[j['rxn']]].setdefault(binof(j['rxn']), []).append(j['cls'] == 'route')
fig, ax = axes('Converged searches that are a saddle\nof the intended reaction (%)',
               'Search success falls with multireference character')
lines(ax, per, lambda v: 100 * np.mean(v))
ax.set_ylim(0, 100)
ax.text(1.0, -0.27, 'Line: mean of four OMol25 models; band: their range.',
        transform=ax.transAxes, ha='right', fontsize=7.5, color=INK2)
save(fig, 'simple_1_success')

CLS = (('route', 'Intended reaction', '#2a78d6'), ('one_end', 'Saddle of another step', '#eb6834'),
       ('elsewhere', 'Saddle of an unrelated reaction', '#e87ba4'), ('no_mode', 'No saddle (minimum or flat)', '#8a8981'),
       ('other', 'Other', '#c9c8c1'))
cnt = {b: Counter() for b in MID}
for m in M:
    for j in checks[m]:
        c = j['cls'] if j['cls'] in ('route', 'one_end', 'elsewhere', 'no_mode') else 'other'
        cnt[binof(j['rxn'])][c] += 1
fig, ax = axes('Share of converged searches (%)', 'What the searches found')
bottom = np.zeros(len(MID))
for key, lab, col in CLS:
    v = np.array([100 * cnt[b][key] / max(1, sum(cnt[b].values())) for b in MID])
    ax.bar(MID, v, bottom=bottom, color=col, width=0.72, edgecolor='white', linewidth=1.5, label=lab)
    bottom += v
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
save(fig, 'simple_2_found')

# 3: forces at the searched saddles ---------------------------------------------------------
per = {m: {sp: {} for sp in SPIN} for m in M}
for r in csv.DictReader(open(f'{RES}/d37_forces.csv')):
    per[r['model']][spin[r['rxn']]].setdefault(binof(r['rxn']), []).append(1000 * float(r['f_err_mae']))
fig, ax = axes('Model force error vs DFT\n(median, meV/Å)', 'Force error at the searched saddles rises with multireference character')
lines(ax, per, np.median)
ax.set_ylim(bottom=0)
save(fig, 'simple_3_forces')

# 4: barriers at the searched saddles -------------------------------------------------------
per = {m: {sp: {} for sp in SPIN} for m in M}
for r in csv.DictReader(open(f'{RES}/d40_barriers.csv')):
    per[r['model']][spin[r['rxn']]].setdefault(binof(r['rxn']), []).append(abs(float(r['error_mev'])))
fig, ax = axes('Barriers off by more than\nchemical accuracy, 43 meV (%)', 'Barrier error at the searched saddles grows a tail')
lines(ax, per, lambda v: 100 * np.mean(np.array(v) > 43))
ax.set_ylim(bottom=0)
save(fig, 'simple_4_barriers')
