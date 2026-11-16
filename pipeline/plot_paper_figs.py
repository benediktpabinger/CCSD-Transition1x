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
# a: success by group and search method; b: what freezing string found; c: success against N_FOD
# (freezing string, four models) with the Transition1x share of each bin; d: our pooled rates next
# to the published ones (Marks et al. 2026: OMol25 models, standard reaction sets, reference-saddle
# criterion; Zhao et al. 2025: best of seven other potentials on Transition1x, connectivity criterion).
rows = list(csv.DictReader(open(f'{RES}/search_checks.csv')))
GL2 = dict(GLABEL, high='BS,\n' + r'$N_\mathrm{FOD}\geq 1$')
fig = plt.figure(figsize=(11.5, 7.6))
gs = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 1.15), height_ratios=(1, 0.95), wspace=0.35, hspace=0.55)
methods = (('fsm', 'Freezing string + Sella'), ('neb_idpp', 'NEB, interpolated start'), ('neb_t1x', 'NEB, Transition1x start'))
x = np.arange(len(GROUPS))
for k, (meth, title) in enumerate(methods):
    ax = fig.add_subplot(gs[0, k])
    style(ax)
    for m in MODELS:
        v = {r['group']: float(r['pct_checked']) for r in rows if r['method'] == meth and r['model'] == m}
        if not v:
            continue
        xs = [i for i, g in enumerate(GROUPS) if g in v]
        ax.plot(xs, [v[GROUPS[i]] for i in xs], color=COLOR[m], lw=1.5, marker=MARK[m], ms=6,
                mec='white', mew=1, label=LABEL[m])
    ax.set_xticks(x, [GL2[g] for g in GROUPS], fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=9, loc='left')
    if k == 0:
        ax.set_ylabel('Converged searches that are a\nsaddle of the intended reaction (%)')
        ax.text(-0.32, 1.12, 'a', transform=ax.transAxes, fontweight='bold', fontsize=11)
    else:
        ax.set_yticklabels([])
fig.axes[0].legend(frameon=False, fontsize=7.5, loc='lower left')

CLS = (('route', 'Intended reaction', '#2a78d6'), ('one_end', 'Saddle of another step', '#eb6834'),
       ('elsewhere', 'Saddle of an unrelated reaction', '#e87ba4'), ('no_mode', 'No saddle (minimum or flat)', '#8a8981'),
       ('other', 'Other (same end, higher order, unconverged)', '#c9c8c1'))
grp = {r['rxn']: r['group'] for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{RES}/{n}.csv'))}
cnt = {g: Counter() for g in GROUPS}
for f in glob.glob(f'{H}/neb_fsm/check_v2/*/*_fsm.json'):
    j = json.load(open(f))
    c = j['cls'] if j['cls'] in ('route', 'one_end', 'elsewhere', 'no_mode') else 'other'
    cnt[grp[j['rxn']]][c] += 1
ax = fig.add_subplot(gs[0, 3])
bottom = np.zeros(len(GROUPS))
for key, lab, col in CLS:
    v = np.array([100 * cnt[g][key] / sum(cnt[g].values()) for g in GROUPS])
    ax.bar(x, v, bottom=bottom, color=col, width=0.7, edgecolor='white', linewidth=1.5, label=lab)
    bottom += v
ax.set_xticks(x, [GL2[g] for g in GROUPS], fontsize=7.5)
ax.set_ylim(0, 100)
ax.set_ylabel('Share of converged searches (%)')
ax.set_title('Freezing string: what was found', fontsize=9, loc='left')
ax.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
ax.text(-0.3, 1.12, 'b', transform=ax.transAxes, fontweight='bold', fontsize=11)

# c: success against N_FOD
sb = list(csv.DictReader(open(f'{RES}/search_by_nfod.csv')))
binr = [r for r in sb if r['model'] == 'esen' and r['nfod_lo'] != 'weighted']
xb = np.arange(len(binr))
blab = [f"{float(r['nfod_lo']):g}–{float(r['nfod_hi']):g}" if r['nfod_hi'] else f"≥{float(r['nfod_lo']):g}" for r in binr]
ax = fig.add_subplot(gs[1, 0:2])
style(ax)
for m in MODELS:
    v = [100 * float(r['success']) for r in sb if r['model'] == m and r['nfod_lo'] != 'weighted']
    ax.plot(xb, v, color=COLOR[m], lw=1.5, marker=MARK[m], ms=6, mec='white', mew=1, label=LABEL[m])
ax.set_xticks(xb, blab, fontsize=8)
ax.set_xlabel(r'$N_\mathrm{FOD}$ at the stored transition state (grey: share of Transition1x)')
ax.set_ylim(0, 100)
ax.set_ylabel('Saddle of the intended reaction (%)')
ax.set_title('Freezing string + Sella, by multireference character', fontsize=9, loc='left')
for xi, r in zip(xb, binr):
    ax.text(xi, 3, f"{100 * float(r['t1x_share']):.0f} %", ha='center', fontsize=7, color=INK2)
ax.text(-0.15, 1.12, 'c', transform=ax.transAxes, fontweight='bold', fontsize=11)

# d: our pooled rates next to the published ones
ax = fig.add_subplot(gs[1, 2:4].subgridspec(1, 2, width_ratios=(0.3, 1))[1])
style(ax)
ax.grid(axis='y', visible=False)
ax.grid(axis='x', color=GRID, lw=0.8)
low = {m: [float(r['success']) for r in sb if r['model'] == m and r['nfod_lo'] != 'weighted' and float(r['nfod_lo']) < 0.2] for m in MODELS}
nlow = {m: [int(r['n_checked']) for r in sb if r['model'] == m and r['nfod_lo'] != 'weighted' and float(r['nfod_lo']) < 0.2] for m in MODELS}
ours_low = [100 * np.average(low[m], weights=nlow[m]) for m in MODELS]
ours_w = [100 * float(r['success']) for m in MODELS for r in sb if r['model'] == m and r['nfod_lo'] == 'weighted']
ax.axvspan(91.8, 96.6, ymin=0.47, ymax=0.78, color='#c9c8c1', alpha=0.6, lw=0)
ax.text(94.2, 3.75, 'Marks et al.\n(OMol25 models,\nstandard sets)', ha='center', va='center', fontsize=7, color=INK)
for j, (m, v) in enumerate(zip(MODELS, ours_low)):
    ax.plot(v, 2.55 + (j - 1.5) * 0.12, ls='none', marker=MARK[m], color=COLOR[m], ms=6, mec='white', mew=1)
ax.plot([69, 69], [0.55, 1.45], color=INK2, lw=2)
ax.text(69.6, 1.25, 'Zhao et al.\n(Transition1x)', ha='left', va='center', fontsize=7, color=INK)
for j, (m, v) in enumerate(zip(MODELS, ours_w)):
    ax.plot(v, 0.75 + (j - 1.5) * 0.12, ls='none', marker=MARK[m], color=COLOR[m], ms=6, mec='white', mew=1)
ax.set_yticks([0.75, 2.55], ['Ours,\nTransition1x mix', 'Ours,\n' + r'$N_\mathrm{FOD}$ < 0.2'], fontsize=8)
ax.set_ylim(0, 4.3)
ax.set_xlim(55, 100)
ax.set_xlabel('Success rate (%)')
ax.set_title('Our rates next to the published benchmarks', fontsize=9, loc='left')
ax.text(-0.42, 1.12, 'd', transform=ax.transAxes, fontweight='bold', fontsize=11)
save(fig, 'fig2_search')
plt.close(fig)

# Figure 3 -----------------------------------------------------------------------------
sp = list(csv.DictReader(open(f'{RES}/model_sp_ts_all.csv')))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw=dict(width_ratios=(1.1, 1), wspace=0.75))
style(a1)
# bins of 0.2 up to 1.2, everything above merged (the bins above 1.2 hold about 30 reactions each);
# shaded band: 95 % bootstrap interval of the median, 2000 resamples per bin
edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 9])
mid = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.35])
rng = np.random.default_rng(0)
for m in MODELS:
    R = [(float(r['nfod']), float(r['f_mae'])) for r in sp if r['model'] == m and r['f_mae']]
    n, e = np.array(R).T
    med, lo95, hi95, cnt = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = e[(n >= lo) & (n < hi)] * 1e3
        b = np.median(rng.choice(v, (2000, len(v))), axis=1)
        med.append(np.median(v))
        lo95.append(np.percentile(b, 2.5))
        hi95.append(np.percentile(b, 97.5))
        cnt.append(len(v))
    a1.fill_between(mid, lo95, hi95, color=COLOR[m], alpha=0.18, lw=0)
    a1.plot(mid, med, color=COLOR[m], lw=1.5, marker=MARK[m], ms=5, mec='white', mew=1, label=LABEL[m])
for x0, c in zip(mid, cnt):
    a1.text(x0, 0.6, f'{c}', ha='center', va='bottom', fontsize=6.5, color=INK2)
a1.text(0.0, 0.6, 'n =', ha='right', va='bottom', fontsize=6.5, color=INK2)
a1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.35], ['0', '0.2', '0.4', '0.6', '0.8', '1.0', '>1.2'])
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
plt.close(fig)

# Figure 4 -----------------------------------------------------------------------------
# DFT residual force at the structures each model's freezing-string search called converged
# (D37): largest per-atom DFT force (D34 measure for convergence), median with 95 % bootstrap
# interval per group and model; the models' own criterion was 0.05 eV/A.
d37 = list(csv.DictReader(open(f'{RES}/d37_forces.csv')))
G4 = {'baseline': 'Low-MR\nbaseline', 'closed': 'CS\ntwins', 'bs': 'BS', 'high': 'BS,\n' + r'$N_\mathrm{FOD}\geq 1$'}
fig, ax = plt.subplots(figsize=(5.2, 3.6))
style(ax)
rng = np.random.default_rng(0)
for j, m in enumerate(MODELS):
    xs, med, lo95, hi95 = [], [], [], []
    for i, g in enumerate(GROUPS):
        v = np.array([float(r['f_dft']) for r in d37 if r['model'] == m and r['group'] == g])
        b = np.median(rng.choice(v, (2000, len(v))), axis=1)
        xs.append(i + (j - 1.5) * 0.14)
        med.append(np.median(v))
        lo95.append(np.percentile(b, 2.5))
        hi95.append(np.percentile(b, 97.5))
    ax.vlines(xs, lo95, hi95, color=COLOR[m], lw=1.5)
    ax.plot(xs, med, ls='none', marker=MARK[m], color=COLOR[m], ms=6, mec='white', mew=1, label=LABEL[m])
ax.axhline(0.05, color=INK2, lw=1, ls='--')
ax.text(3.45, 0.05, 'model\ncriterion', ha='left', va='center', fontsize=7, color=INK2)
ax.set_yscale('log')
ax.set_yticks([0.03, 0.05, 0.1, 0.2, 0.4], ['0.03', '0.05', '0.1', '0.2', '0.4'])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks(range(len(GROUPS)), [G4[g] for g in GROUPS], fontsize=8)
ax.set_xlim(-0.5, 3.9)
ax.set_ylabel('DFT residual force at the model-\nconverged structure (eV/Å, median)')
ax.set_title('Freezing-string saddles that the models call converged', fontsize=9, loc='left')
ax.legend(frameon=False, fontsize=7.5, loc='upper left')
save(fig, 'fig4_residual')
