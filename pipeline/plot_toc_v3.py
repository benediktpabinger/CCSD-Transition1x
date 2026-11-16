"""Draft TOC graphic, version 3 (ACS: at most 3.25 x 1.75 in). Schematic, not data.

Each side shows both effects, DFT solid and model dashed:
  left, low multireference: one barrier R -> P; the model curve lies on the DFT curve and the
        model's saddle is the intended saddle, where DFT has it (about 90 % found).
  right, high multireference: the intended saddle to P (DFT, grey) and a lower saddle to an
        intermediate I (DFT, orange); the model curve follows the second one, shifted, so the model's
        saddle is on the wrong saddle and beside the DFT position (about 25 % found; force error about 3x).

Output figures/paper/toc_v3.{png,pdf} at 3.25 x 1.75 in, and toc_v3_large.png for viewing.

    python pipeline/plot_toc_v3.py
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
    dash = (0, (2.5, 1.6))

    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=(0.4, 9.4, 0.5, 0.72), aspect='auto',
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list('mr', [LOW, HIGH]))
    ax.add_patch(FancyArrowPatch((9.25, 0.61), (9.8, 0.61), arrowstyle='-|>', mutation_scale=7 * scale, color=HIGH, lw=0))
    ax.text(0.4, 0.2, 'low multireference', fontsize=fs, color=LOW, va='center')
    ax.text(9.4, 0.2, r'high multireference ($N_\mathrm{FOD}$)', fontsize=fs, color=HIGH, va='center', ha='right')
    ax.text(5.0, 5.12, 'OMol25 potentials at transition states', fontsize=fs * 1.1, ha='center', fontweight='bold')

    # left: model on DFT, on the intended saddle
    x = np.linspace(0.5, 4.1, 200)
    yd = 1.35 + bump(x, 2.3, 0.6, 2.0) - 0.3 * (x - 0.5) / 3.6
    ym = 1.35 + bump(x, 2.33, 0.6, 2.03) - 0.3 * (x - 0.5) / 3.6
    ax.plot(x, yd, color=LOW, lw=lw)
    ax.plot(x, ym, color=MODEL, lw=lw * 0.75, ls=dash)
    ax.plot(2.33, ym.max(), 'o', ms=4 * scale, color=INK, zorder=5)
    ax.text(0.55, 1.1, 'R', fontsize=fs, color=INK2)
    ax.text(4.0, 0.85, 'P', fontsize=fs, color=INK2)
    ax.text(2.3, 4.35, '✓ intended saddle found: ≈ 90 %', fontsize=fs, ha='center', fontweight='bold')
    ax.text(2.33, 3.72, 'model saddle = DFT saddle', fontsize=fs * 0.8, ha='center', color=INK2)

    # right: intended saddle to P (grey), saddle to an intermediate (orange) with the model shifted
    ax.text(7.5, 4.35, '✗ intended saddle found: ≈ 25 %', fontsize=fs, ha='center', fontweight='bold')
    xa = np.linspace(5.3, 9.2, 200)
    ya = 2.0 + bump(xa, 7.75, 0.45, 1.6) - 0.35 * (xa - 5.3) / 3.9
    ax.plot(xa, ya, color=INK2, lw=lw * 0.8)
    ax.text(7.75, 3.72, 'intended', fontsize=fs * 0.8, ha='center', color=INK2)
    xb = np.linspace(5.3, 7.1, 150)
    step = 0.6 / (1 + np.exp(-(xb - 6.45) / 0.12))
    yb = 2.0 + bump(xb, 6.05, 0.3, 1.0) - step
    ybm = 2.0 + bump(xb, 6.27, 0.3, 1.04) - 0.6 / (1 + np.exp(-(xb - 6.62) / 0.12))
    ax.plot(xb, yb, color=HIGH, lw=lw)
    ax.plot(xb, ybm, color=MODEL, lw=lw * 0.75, ls=dash)
    j, k = np.argmax(ybm), np.argmax(yb)
    ax.plot(xb[j], ybm[j], 'o', ms=4 * scale, color=INK, zorder=5)
    for xx, yy, col in ((xb[k], yb[k], HIGH), (xb[j], ybm[j], MODEL)):
        ax.plot([xx, xx], [1.45, yy], color=col, lw=0.6 * scale, ls=':')
    ax.text(5.15, 1.9, 'R', fontsize=fs, color=INK2, ha='right')
    ax.text(9.25, 1.55, 'P', fontsize=fs, color=INK2)
    ax.text(7.15, 1.3, 'I', fontsize=fs, color=INK2)
    ax.text(5.5, 3.25, 'wrong saddle,\nbeside DFT', fontsize=fs * 0.8, ha='right', color=HIGH, va='bottom')
    ax.text(6.3, 1.1, 'force error ≈ ×3', fontsize=fs * 0.8, ha='center', color=HIGH)

    ax.text(5.0, 4.78, 'solid: DFT   ·   dashed: model', fontsize=fs * 0.8, ha='center', color=INK2)
    fig.savefig(f'{OUT}/{name}.png', dpi=dpi)
    if scale == 1:
        fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig)
    print('wrote', name)


draw(1, 'toc_v3', 600)
draw(3, 'toc_v3_large', 200)
