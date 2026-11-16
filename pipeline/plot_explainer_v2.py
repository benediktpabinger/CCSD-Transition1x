"""Explainer figure, version 2: DFT curves only; the model enters as its returned transition state (purple dot).

Left, low multireference (rxn0914, baseline): reactant, the saddle found by MACE-OMol freezing string
+ Sella (class route under the D35 check) and product; model (dashed) lies on DFT (solid).
Right, high multireference (rxn0268, N_FOD >= 1, broken-symmetry): the intended transition state is
the stored Transition1x one (grey path R -> P); the model's search ends on a saddle whose downhill
ends are the reactant and an intermediate I (D35 check, class one_end), drawn as the orange path R -> I.
The model's transition state (purple dot) sits beside (saddle shifted: force error) and above (barrier error)
the DFT saddle; on the left it sits on the DFT saddle.
Energy profiles and offsets are schematic; molecules are the real structures. Table numbers are the
rounded headline results for freezing string + Sella on four models.

Output figures/paper/explainer_v2.{png,pdf}

    python pipeline/plot_explainer_v2.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_check import bonds  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(REPO, 'figures', 'paper')
INK, INK2 = '#1f1f1e', '#6b6a64'
LOW, HIGH, MODEL = '#2a78d6', '#eb6834', '#7a3fb0'
ELEM = {1: ('#ffffff', 0.16), 6: ('#5b5b5b', 0.25), 7: ('#3060d8', 0.24), 8: ('#e0302a', 0.23)}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})


def kabsch(x, ref):
    a, b = x - x.mean(0), ref - ref.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return a @ (u @ np.diag([1, 1, d]) @ vt)


def view(ref):
    c = ref - ref.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    return v[:, ::-1]                       # largest spread along x, then y, smallest along z


def mol(ax, Z, x, ref, rot, cx, cy, size):
    p = kabsch(x, ref) @ rot
    span = np.ptp(kabsch(ref, ref) @ rot, axis=0)[:2].max()
    s = size / span
    xy = p[:, :2] * s + [cx, cy]
    for i, j in bonds(Z, x):
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color='#8a8984', lw=2.0, zorder=6, solid_capstyle='round')
    for k in np.argsort(p[:, 2]):
        col, r = ELEM[int(Z[k])]
        ax.add_patch(Circle(xy[k], r * s * 1.05, facecolor=col, edgecolor='#4a4a46', lw=0.7, zorder=7))


def bump(x, x0, w, h):
    return h * np.exp(-((x - x0) / w) ** 2)


def load(rx):
    d = f'{H}/neb_fsm/fsm/mace-omol/{rx}'
    R, P, T = read(f'{d}/reactant.xyz'), read(f'{d}/product.xyz'), read(f'{d}/transition_state.xyz')
    return R.numbers, R.positions, P.positions, T.positions


fig = plt.figure(figsize=(8.0, 5.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.0)
ax.axis('off')
dash = (0, (3, 2))

# table ------------------------------------------------------------------------------------------
ax.text(5.0, 6.72, 'OMol25 potentials at transition states', fontsize=13, ha='center', fontweight='bold')
ax.text(2.45, 6.28, 'low multireference', fontsize=10, ha='center', color=LOW, fontweight='bold')
ax.text(7.55, 6.28, 'high multireference', fontsize=10, ha='center', color=HIGH, fontweight='bold')
rows = (('search finds the intended TS', '≈ 90 %', '≈ 25 %'),
        ('force error vs DFT', 'low', '≈ 3× higher'),
        ('barrier error vs DFT', 'low', '1 in 5 > 43 meV'))
for k, (lab, lo, hi) in enumerate(rows):
    y = 5.92 - 0.3 * k
    ax.text(5.0, y, lab, fontsize=8.5, ha='center', color=INK2)
    ax.text(2.45, y, lo, fontsize=9.5, ha='center', color=LOW, fontweight='bold' if k == 0 else 'normal')
    ax.text(7.55, y, hi, fontsize=9.5, ha='center', color=HIGH, fontweight='bold' if k == 0 else 'normal')
ax.plot([0.4, 9.6], [5.15, 5.15], color='#dddcd6', lw=0.8)
ax.text(5.0, 4.93, 'curves: DFT energy profile   ·   purple dot: transition state returned by the model',
        fontsize=8, ha='center', color=INK2)

# left: low multireference ------------------------------------------------------------------------
Z, xr, xp, xt = load('rxn0914')
rot = view(xr)
x = np.linspace(0.45, 4.45, 200)
yd = 1.95 + bump(x, 2.45, 0.62, 1.75) - 0.3 * (x - 0.45) / 4.0
ax.plot(x, yd, color=LOW, lw=2.4)
ax.plot(x[np.argmax(yd)], yd.max(), 'o', ms=9, color=MODEL, mec='white', mew=1.2, zorder=8)
mol(ax, Z, xr, xr, rot, 0.8, 1.3, 0.85)
mol(ax, Z, xp, xr, rot, 4.15, 1.0, 0.85)
mol(ax, Z, xt, xr, rot, 2.47, 4.4, 0.85)
ax.text(0.75, 0.72, 'reactant', fontsize=8, ha='center', color=INK2)
ax.text(4.15, 0.42, 'product', fontsize=8, ha='center', color=INK2)
ax.text(3.25, 4.4, 'model TS = DFT TS\n(intended saddle)', fontsize=8, ha='left', va='center', color=LOW)

# right: high multireference ----------------------------------------------------------------------
Z, xr, xp, xt = load('rxn0268')
rot = view(xr)
j = json.load(open(f'{H}/neb_fsm/check_v2/mace-omol/rxn0268_fsm.json'))
xi = np.array(j['pos_a'] if j['end_a'] == 'X' else j['pos_b'])
xs = np.array(json.load(open(f'{H}/t1x_ts_all.json'))['rxn0268']['positions'])
xa = np.linspace(5.3, 9.6, 200)
ya = 2.25 + bump(xa, 8.1, 0.45, 1.85) - 0.45 * (xa - 5.3) / 4.3
ax.plot(xa, ya, color=INK2, lw=2.0)
xb = np.linspace(5.3, 7.2, 160)
yb = 2.25 + bump(xb, 6.15, 0.3, 1.05) - 0.55 / (1 + np.exp(-(xb - 6.55) / 0.12))
ax.plot(xb, yb, color=HIGH, lw=2.4)
kd = np.argmax(yb)
mx, my = xb[kd] + 0.32, yb[kd] + 0.42          # model's transition state, beside and above the DFT saddle
ax.plot(mx, my, 'o', ms=9, color=MODEL, mec='white', mew=1.2, zorder=8)
# barrier error: vertical bracket
xbk = xb[kd] - 0.45
ax.plot([xbk + 0.06, xbk, xbk, xbk + 0.06], [yb[kd], yb[kd], my, my], color=HIGH, lw=1.2)
ax.plot([xbk + 0.06, xb[kd]], [yb[kd], yb[kd]], color=HIGH, lw=0.7, ls=':')
ax.plot([xbk + 0.06, mx], [my, my], color=MODEL, lw=0.7, ls=':')
ax.text(xbk - 0.08, 0.5 * (yb[kd] + my), 'barrier\nerror', fontsize=8, ha='right', va='center', color=HIGH)
# force error: saddle shifted sideways
ys = 2.95
ax.plot([xb[kd], xb[kd]], [ys, yb[kd]], color=HIGH, lw=0.7, ls=':')
ax.plot([mx, mx], [ys, my], color=MODEL, lw=0.7, ls=':')
ax.add_patch(FancyArrowPatch((xb[kd], ys), (mx, ys), arrowstyle='<->', mutation_scale=6, color=HIGH, lw=1.0))
ax.plot([5.75, xb[kd] - 0.02], [ys, ys], color=HIGH, lw=0.6)
ax.text(5.7, ys, 'saddle\nshifted\n(force\nerror)', fontsize=8, ha='right', va='center', color=HIGH)
mol(ax, Z, xr, xr, rot, 5.3, 1.45, 0.72)
mol(ax, Z, xi, xr, rot, 7.25, 1.3, 0.72)
mol(ax, Z, xp, xr, rot, 9.3, 1.3, 0.72)
mol(ax, Z, xt, xr, rot, 6.45, 4.45, 0.72)
mol(ax, Z, xs, xr, rot, 8.3, 4.6, 0.72)
ax.text(5.3, 0.92, 'reactant', fontsize=8, ha='center', color=INK2)
ax.text(7.25, 0.78, 'intermediate', fontsize=8, ha='center', color=INK2)
ax.text(9.3, 0.78, 'product', fontsize=8, ha='center', color=INK2)
ax.text(6.45, 4.02, 'model TS: wrong saddle', fontsize=8, ha='center', va='center', color=HIGH)
ax.text(8.3, 4.18, 'intended TS', fontsize=8, ha='center', va='center', color=INK2)

# multireference arrow -------------------------------------------------------------------------------
grad = np.linspace(0, 1, 256)[None, :]
ax.imshow(grad, extent=(0.4, 9.3, 0.18, 0.32), aspect='auto',
          cmap=matplotlib.colors.LinearSegmentedColormap.from_list('mr', [LOW, HIGH]), zorder=1)
ax.add_patch(FancyArrowPatch((9.15, 0.25), (9.75, 0.25), arrowstyle='-|>', mutation_scale=10, color=HIGH, lw=0))
ax.text(5.0, 0.04, r'multireference character ($N_\mathrm{FOD}$)', fontsize=8.5, ha='center', color=INK2)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/explainer_v2.{ext}', dpi=220)
print('wrote explainer_v2')
