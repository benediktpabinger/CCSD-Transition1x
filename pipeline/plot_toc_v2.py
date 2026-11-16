"""Draft TOC graphic, version 2 (ACS: at most 3.25 x 1.75 in). Schematic, not data.

Left: low multireference, one barrier R -> P, the search finds the intended saddle (about 90 %).
Right top: high multireference, a fork from R: the intended saddle to P (dashed, not found) and a
lower saddle to an intermediate I, where the search ends (intended saddle found about 25 %).
Right bottom: zoom on a barrier top, DFT solid and model dashed with their maxima apart: the
model's saddle is not the DFT saddle (force error about 3x). Rounded headline numbers for
freezing string + Sella on four models.

Output figures/paper/toc_v2.{png,pdf} at 3.25 x 1.75 in, and toc_v2_large.png for viewing.

    python pipeline/plot_toc_v2.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(REPO, 'figures', 'paper')
INK, INK2 = '#1f1f1e', '#6b6a64'
LOW, HIGH, MODEL = '#2a78d6', '#eb6834', '#7a3fb0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})


def bump(x, x0, w, h):
    return h * np.exp(-((x - x0) / w) ** 2)


def draw(scale, name, dpi):
    fig = plt.figure(figsize=(3.25 * scale, 1.75 * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis('off')
    fs = 5.6 * scale
    lw = 1.5 * scale

    # multireference axis
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=(0.4, 9.4, 0.5, 0.72), aspect='auto',
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list('mr', [LOW, HIGH]))
    ax.add_patch(FancyArrowPatch((9.25, 0.61), (9.8, 0.61), arrowstyle='-|>', mutation_scale=7 * scale, color=HIGH, lw=0))
    ax.text(0.4, 0.2, 'low multireference', fontsize=fs, color=LOW, va='center')
    ax.text(9.4, 0.2, r'high multireference ($N_\mathrm{FOD}$)', fontsize=fs, color=HIGH, va='center', ha='right')
    ax.text(5.0, 5.12, 'OMol25 potentials at transition states', fontsize=fs * 1.1, ha='center', fontweight='bold')

    # left: one barrier, intended saddle found
    x = np.linspace(0.5, 4.1, 200)
    y = 1.35 + bump(x, 2.3, 0.6, 2.0) - 0.3 * (x - 0.5) / 3.6
    ax.plot(x, y, color=LOW, lw=lw)
    ax.plot(2.3, y[np.argmin(abs(x - 2.3))], 'o', ms=4 * scale, color=INK, zorder=5)
    ax.text(0.55, 1.12, 'R', fontsize=fs, color=INK2)
    ax.text(4.0, 0.85, 'P', fontsize=fs, color=INK2)
    ax.text(2.3, 4.35, '✓ intended saddle found: ≈ 90 %', fontsize=fs, ha='center', fontweight='bold')

    # right top: fork from R; intended saddle to P (dashed, not found), lower saddle to I (found)
    ax.text(7.5, 4.35, '✗ intended saddle found: ≈ 25 %', fontsize=fs, ha='center', fontweight='bold')
    xa = np.linspace(5.3, 9.2, 200)
    ya = 2.25 + bump(xa, 7.6, 0.45, 1.55) - 0.35 * (xa - 5.3) / 3.9
    ax.plot(xa, ya, color=INK2, lw=lw * 0.8, ls=(0, (3, 2)))
    xb = np.linspace(5.3, 7.0, 120)
    yb = 2.25 + bump(xb, 6.0, 0.28, 0.95) - 0.55 * (1 / (1 + np.exp(-(xb - 6.35) / 0.12)))
    ax.plot(xb, yb, color=HIGH, lw=lw)
    i = np.argmax(yb)
    ax.plot(xb[i], yb[i], 'o', ms=4 * scale, color=INK, zorder=5)
    ax.text(5.15, 2.12, 'R', fontsize=fs, color=INK2, ha='right')
    ax.text(9.28, 1.82, 'P', fontsize=fs, color=INK2)
    ax.text(7.05, 1.58, 'I', fontsize=fs, color=INK2)
    ax.text(7.6, 3.92, 'intended, not found', fontsize=fs * 0.8, ha='center', color=INK2)
    ax.text(5.75, 3.25, 'found: saddle\nto an\nintermediate', fontsize=fs * 0.8, ha='right', color=HIGH, va='bottom')

    # right bottom: zoom on a barrier top, DFT vs model maxima apart
    xz = np.linspace(8.2, 9.5, 120)
    yd = 1.0 + bump(xz, 8.75, 0.26, 0.42)
    ym = 1.0 + bump(xz, 8.95, 0.26, 0.44)
    ax.plot(xz, yd, color=INK, lw=lw * 0.8)
    ax.plot(xz, ym, color=MODEL, lw=lw * 0.8, ls=(0, (3, 2)))
    for x0, col in ((8.75, INK), (8.95, MODEL)):
        ax.plot([x0, x0], [0.95, 1.5], color=col, lw=0.6 * scale, ls=':')
    ax.text(8.7, 1.52, 'DFT', fontsize=fs * 0.75, color=INK, ha='right')
    ax.text(9.0, 1.52, 'model', fontsize=fs * 0.75, color=MODEL, ha='left')
    ax.text(8.1, 1.2, 'model saddle ≠ DFT saddle\nforce error ≈ ×3', fontsize=fs * 0.8, color=HIGH, ha='right', va='center')

    fig.savefig(f'{OUT}/{name}.png', dpi=dpi)
    if scale == 1:
        fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig)
    print('wrote', name)


draw(1, 'toc_v2', 600)
draw(3, 'toc_v2_large', 200)
