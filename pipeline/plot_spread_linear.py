"""Same comparison on a linear axis.

The data span five orders of magnitude, so a linear axis cannot show all of it:
either the outliers set the scale and everything else collapses onto the
baseline, or the axis is cut and the outliers sit outside it. The second is
chosen here -- the axis covers the bulk of both groups, and every point beyond
it is drawn at the top edge as a triangle with its value written next to it, so
nothing is silently dropped.
"""
import itertools
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = '/home/energy/s242862'
HA_MEV = 27211.386
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results'}

res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
TOP = [res[i]['rxn'] for i in range(26)]
MID = [res[i - 1]['rxn'] for i in [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
LOW = [res[i]['rxn'] for i in range(n - 10, n)]
grp = {}
for r in TOP: grp[r] = 'high'
for r in MID: grp.setdefault(r, 'mid')
for r in LOW: grp.setdefault(r, 'low')


def read_xyz(p):
    L = open(p).read().split('\n')
    m = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + m]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B):
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def reactive(rx, sym):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1]) for e in rb[:2]]
    from ase.data import atomic_numbers, covalent_radii
    r, p = (f'{H}/orca_neb_results/{rx}/reactant.xyz',
            f'{H}/orca_neb_results/{rx}/product.xyz')
    if not (os.path.exists(r) and os.path.exists(p)):
        return []
    _, xr = read_xyz(r); _, xp = read_xyz(p)
    cand = []
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            dr = float(np.linalg.norm(xr[i] - xr[j]))
            dp = float(np.linalg.norm(xp[i] - xp[j]))
            rc = 1.3 * (covalent_radii[atomic_numbers[sym[i]]]
                        + covalent_radii[atomic_numbers[sym[j]]])
            if min(dr, dp) < rc:
                cand.append((abs(dp - dr), i, j))
    cand.sort(reverse=True)
    return [(i, j) for _, i, j in cand[:2]]


data = {'single-reference': {'e': [], 'g': [], 'rc': [], 'lab': []},
        'multireference': {'e': [], 'g': [], 'rc': [], 'lab': []}}
for rx in grp:
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    geo = {g['source']: g for g in d['geometries']}
    ref = geo.get('RKS-ref')
    if not ref or ref.get('ext_stable') is None:
        continue
    cls = ('multireference' if ref['ext_stable'] is False
           else 'single-reference')
    es, xs = {}, {}
    for m, dn in MODELS.items():
        g = geo.get(m)
        if g and g.get('ext_stable') is not None:
            es[m] = (g.get('e_rks') if g['ext_stable']
                     else (g.get('bs') or {}).get('e_uks'))
        f = f'{H}/{dn}/{rx}/transition_state.xyz'
        if os.path.exists(f):
            xs[m] = read_xyz(f)
    ev = [v for v in es.values() if v is not None]
    if len(ev) < 2 or len(xs) < 2:
        continue
    sym = list(xs.values())[0][0]
    pairs = reactive(rx, sym)
    gg, rr = [], []
    for a, b in itertools.combinations(xs, 2):
        if xs[a][0] != xs[b][0]:
            continue
        gg.append(kabsch(xs[a][1], xs[b][1]))
        if pairs:
            rr.append(max(
                abs(float(np.linalg.norm(xs[a][1][i] - xs[a][1][j]))
                    - float(np.linalg.norm(xs[b][1][i] - xs[b][1][j])))
                for i, j in pairs))
    data[cls]['e'].append((max(ev) - min(ev)) * HA_MEV)
    data[cls]['g'].append(max(gg) if gg else np.nan)
    data[cls]['rc'].append(max(rr) if rr else np.nan)
    data[cls]['lab'].append(rx)

GROUPS = ['single-reference', 'multireference']
COL = {'single-reference': '#4878a8', 'multireference': '#c0553b'}
PANELS = [('e', 'Barrier spread',
           'largest pairwise difference in\nDFT energy at the predicted TS  [meV]',
           '{:.0f}'),
          ('g', 'Transition-state geometry',
           'largest pairwise Kabsch RMSD  [Å]', '{:.2f}'),
          ('rc', 'Reactive coordinate',
           'largest pairwise difference in the\nbreaking / forming bonds  [Å]',
           '{:.2f}')]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.0))
fig.subplots_adjust(top=0.74, bottom=0.12, left=0.07, right=0.98, wspace=0.32)
rng = np.random.default_rng(0)

for ax, (key, title, ylab, fmt) in zip(axes, PANELS):
    vals = [np.array([v for v in data[g][key] if np.isfinite(v)])
            for g in GROUPS]
    labs = [[l for l, v in zip(data[g]['lab'], data[g][key])
             if np.isfinite(v)] for g in GROUPS]
    # axis covers the bulk: 1.35x the largest upper whisker
    tops = []
    for v in vals:
        q1, q3 = np.percentile(v, [25, 75])
        w = v[v <= q3 + 1.5 * (q3 - q1)]
        tops.append(w.max() if len(w) else v.max())
    ylim = max(tops) * 1.6
    bp = ax.boxplot(vals, widths=0.5, patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                    flierprops=dict(marker='', ls='none'),
                    whiskerprops=dict(color='#555'),
                    capprops=dict(color='#555'))
    for patch, g in zip(bp['boxes'], GROUPS):
        patch.set_facecolor(COL[g]); patch.set_alpha(0.28)
        patch.set_edgecolor(COL[g]); patch.set_linewidth(1.4)
    for i, (g, v, lb) in enumerate(zip(GROUPS, vals, labs), start=1):
        inside = v <= ylim
        x = i + rng.uniform(-0.13, 0.13, len(v))
        ax.scatter(x[inside], v[inside], s=26, color=COL[g], alpha=0.75,
                   zorder=3, edgecolor='white', linewidth=0.6)
        out = np.where(~inside)[0]
        for k, idx in enumerate(sorted(out, key=lambda j: v[j])):
            xo = i - 0.33 + 0.20 * (k % 4)
            ax.plot(xo, ylim * 0.975, marker='^', ms=9, color=COL[g],
                    clip_on=False, zorder=6)
            # label inside the axes, reading downwards, so it cannot collide
            # with the header text above the panel
            ax.annotate(f'{lb[idx]}  {fmt.format(v[idx])}',
                        xy=(xo, ylim * 0.94), rotation=-90,
                        ha='center', va='top', fontsize=7.4,
                        color=COL[g], zorder=6)
    ax.set_ylim(0, ylim)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'{g}\nn = {len(v)}' for g, v in zip(GROUPS, vals)])
    ax.set_title(title, fontsize=12, pad=9)
    ax.set_ylabel(ylab, fontsize=9.5)
    ax.grid(axis='y', ls=':', color='#bbb', lw=0.7)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for i, v in enumerate(vals, start=1):
        med = np.median(v)
        ax.annotate(f'median {med:.4g}', xy=(i, med), xytext=(22, 4),
                    textcoords='offset points', ha='left', fontsize=8.5,
                    color='#333')

fig.suptitle('Disagreement among the three OMol25 models  '
             '(UMA-S, UMA-M, eSEN)', fontsize=15, y=0.972, x=0.07, ha='left')
fig.text(0.07, 0.912,
         'One point per reaction, showing how far the three models sit apart '
         'from each other. No DFT reference enters — this is model against '
         'model.', fontsize=9.8, color='#333', ha='left')
fig.text(0.07, 0.876,
         'Linear axis, cut to show the bulk. Points beyond the cut are drawn '
         'as triangles at the top edge and labelled, so none is hidden.',
         fontsize=9.8, color='#333', ha='left')
fig.text(0.07, 0.826,
         'single-reference   the restricted (RKS) solution at the reference '
         'transition state is externally stable: no lower spin-broken solution '
         'exists, the closed-shell picture is valid.',
         fontsize=9.2, color=COL['single-reference'], ha='left')
fig.text(0.07, 0.790,
         'multireference     the restricted solution is externally unstable: a '
         'spin-broken (UKS) solution lies below it, so a single determinant '
         'does not describe the transition state.',
         fontsize=9.2, color=COL['multireference'], ha='left')

fig.savefig('/home/energy/s242862/model_spread_linear.png', dpi=200,
            facecolor='white')
print('written')
