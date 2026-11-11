"""Draft figures for the RQ2 paper.

Figure 2  search success by group (three methods, four models) and what the failures are
          (freezing string, D35 check, all four models pooled)
Figure 3  force error against DFT: (a) against N_FOD at the stored transition states,
          (b) ratio to the low-MR baseline at the freezing-string structures (D37), 95 % intervals

Output figures/paper/fig2_search.{pdf,png}, figures/paper/fig3_forces.{pdf,png}

    python pipeline/plot_paper_figs.py
"""
import csv
import glob
import json
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
OUT = os.path.join(REPO, 'figures', 'paper')
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID = '#1f1f1e', '#5f5e58', '#e4e3df'
MODELS = ('esen', 'uma-s', 'uma-m', 'mace-omol')
LABEL = {'esen': 'eSEN', 'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'mace-omol': 'MACE-OMol'}
COLOR = dict(zip(MODELS, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')))
MARK = dict(zip(MODELS, ('o', 's', '^', 'D')))
GROUPS = ('baseline', 'closed', 'bs', 'high')
GLABEL = {'baseline': 'Low-MR\nbaseline', 'closed': 'CS\ntwins', 'bs': 'BS', 'high': r'$N_\mathrm{FOD}$' + '\n' + r'$\geq 1$'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})


def style(ax):
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=300, bbox_inches='tight')
    print('wrote', f'{OUT}/{name}.pdf')


# Figure 2 -----------------------------------------------------------------------------
rows = list(csv.DictReader(open(f'{RES}/search_checks.csv')))
fig = plt.figure(figsize=(11.5, 3.8))
gs = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 1.15), wspace=0.35)
methods = (('fsm', 'Freezing string + Sella'), ('neb_idpp', 'NEB, interpolated start'), ('neb_t1x', 'NEB, Transition1x start'))
x = np.arange(len(GROUPS))
for k, (meth, title) in enumerate(methods):
    ax = fig.add_subplot(gs[k])
    style(ax)
    for m in MODELS:
        v = {r['group']: float(r['pct_checked']) for r in rows if r['method'] == meth and r['model'] == m}
        if not v:
            continue
        xs = [i for i, g in enumerate(GROUPS) if g in v]
        ax.plot(xs, [v[GROUPS[i]] for i in xs], color=COLOR[m], lw=1.5, marker=MARK[m], ms=6,
                mec='white', mew=1, label=LABEL[m])
    ax.set_xticks(x, [GLABEL[g] for g in GROUPS], fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=9, loc='left')
    if k == 0:
        ax.set_ylabel('Converged searches that are a\nsaddle of the intended reaction (%)')
    else:
        ax.set_yticklabels([])
fig.axes[0].legend(frameon=False, fontsize=7.5, loc='lower left')
fig.text(0.125, 0.98, 'a', fontweight='bold', fontsize=11)

# failure composition, freezing string, four models pooled
CLS = (('route', 'Intended reaction', '#2a78d6'), ('one_end', 'Saddle of another step', '#eb6834'),
       ('elsewhere', 'Saddle of an unrelated reaction', '#e87ba4'), ('no_mode', 'No saddle (minimum or flat)', '#8a8981'),
       ('other', 'Other (same end, higher order, unconverged)', '#c9c8c1'))
grp = {r['rxn']: r['group'] for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{RES}/{n}.csv'))}
cnt = {g: Counter() for g in GROUPS}
for f in glob.glob(f'{H}/neb_fsm/check_v2/*/*_fsm.json'):
    j = json.load(open(f))
    c = j['cls'] if j['cls'] in ('route', 'one_end', 'elsewhere', 'no_mode') else 'other'
    cnt[grp[j['rxn']]][c] += 1
ax = fig.add_subplot(gs[3])
bottom = np.zeros(len(GROUPS))
for key, lab, col in CLS:
    v = np.array([100 * cnt[g][key] / sum(cnt[g].values()) for g in GROUPS])
    ax.bar(x, v, bottom=bottom, color=col, width=0.7, edgecolor='white', linewidth=1.5, label=lab)
    bottom += v
ax.set_xticks(x, [GLABEL[g] for g in GROUPS], fontsize=7.5)
ax.set_ylim(0, 100)
ax.set_ylabel('Share of converged searches (%)')
ax.set_title('Freezing string: what was found', fontsize=9, loc='left')
ax.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
fig.text(0.70, 0.98, 'b', fontweight='bold', fontsize=11)
save(fig, 'fig2_search')
plt.close(fig)

# Figure 3 -----------------------------------------------------------------------------
sp = list(csv.DictReader(open(f'{RES}/model_sp_ts_all.csv')))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw=dict(width_ratios=(1.1, 1), wspace=0.75))
style(a1)
edges = np.arange(0, 1.61, 0.2)
mid = 0.5 * (edges[1:] + edges[:-1])
for m in MODELS:
    R = [(float(r['nfod']), float(r['f_mae'])) for r in sp if r['model'] == m and r['f_mae']]
    n, e = np.array(R).T
    med = [np.median(e[(n >= lo) & (n < hi)]) if ((n >= lo) & (n < hi)).sum() >= 20 else np.nan
           for lo, hi in zip(edges[:-1], edges[1:])]
    a1.plot(mid, np.array(med) * 1e3, color=COLOR[m], lw=1.5, marker=MARK[m], ms=5, mec='white', mew=1, label=LABEL[m])
a1.set_xlabel(r'$N_\mathrm{FOD}$ at the stored transition state')
a1.set_ylabel('Median force error vs. OMol25 label\n(mean abs. component, meV/Å)')
a1.set_ylim(bottom=0)
a1.set_title('Stored transition states, 10 009 per model', fontsize=9, loc='left')
a1.legend(frameon=False, fontsize=7.5)
a1.text(-0.2, 1.06, 'a', transform=a1.transAxes, fontweight='bold', fontsize=11)

summ = {r['model']: r for r in csv.DictReader(open(f'{RES}/d37_summary.csv'))}
comps = (('cs_mr', r'Closed-shell, $N_\mathrm{FOD}\geq 0.5$' + '\nvs. baseline'), ('high', r'$N_\mathrm{FOD}\geq 1$ vs. baseline'),
         ('bs_twin', 'Broken-symmetry vs.\nclosed-shell twin'))
style(a2)
a2.grid(axis='y', visible=False)
a2.grid(axis='x', color=GRID, lw=0.8)
for i, (k, lab) in enumerate(comps):
    for j, m in enumerate(MODELS):
        s = summ[m]
        y = i + (j - 1.5) * 0.17
        r, lo, hi = float(s[f'{k}_ratio']), float(s[f'{k}_lo']), float(s[f'{k}_hi'])
        a2.plot([lo, hi], [y, y], color=COLOR[m], lw=1.5)
        a2.plot(r, y, marker=MARK[m], color=COLOR[m], ms=6, mec='white', mew=1, label=LABEL[m] if i == 0 else None)
a2.axvline(1, color=INK2, lw=1, ls='--')
a2.set_xscale('log')
a2.set_xticks([0.8, 1, 2, 3, 5], ['0.8', '1', '2', '3', '5'])
a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a2.set_yticks(range(len(comps)), [c[1] for c in comps], fontsize=8)
a2.invert_yaxis()
a2.set_xlabel('Force-error ratio (median, 95 % interval)')
a2.set_title('Freezing-string structures, all four models', fontsize=9, loc='left')
a2.text(-0.75, 1.06, 'b', transform=a2.transAxes, fontweight='bold', fontsize=11)
save(fig, 'fig3_forces')
