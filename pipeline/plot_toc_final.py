"""TOC graphic, stripped down for the journal listing (ACS: at most 3.25 x 1.75 in). Schematic.

Two energy profiles over a low-to-high multireference arrow. Left: one barrier, model (dashed) on
DFT (solid), intended saddle found about 90 %. Right: the intended saddle (grey) and a saddle to an
intermediate where the search ends, model beside and above DFT; intended saddle found about 25 %,
model errors about 3x larger. Minimal text so it reads at thumbnail size; the detailed version is
plot_toc_v6.py (for the paper or a talk).

Output figures/paper/toc_final.{png,pdf} at 3.25 x 1.75 in, toc_final_large.png for viewing,
toc_final_actual.png at screen size (150 dpi) to check legibility.

    python pipeline/plot_toc_final.py
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


def draw(scale, name, dpi, pdf=False):
    fig = plt.figure(figsize=(3.25 * scale, 1.75 * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis('off')
    fs = 7.5 * scale
    lw = 1.8 * scale
    dash = (0, (2.2, 1.4))

    # multireference arrow
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=(0.4, 9.3, 0.62, 0.86), aspect='auto',
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list('mr', [LOW, HIGH]))
    ax.add_patch(FancyArrowPatch((9.1, 0.74), (9.8, 0.74), arrowstyle='-|>', mutation_scale=9 * scale, color=HIGH, lw=0))
    ax.text(0.4, 0.27, 'low multireference', fontsize=fs * 0.85, color=LOW, va='center')
    ax.text(9.3, 0.27, 'high multireference', fontsize=fs * 0.85, color=HIGH, va='center', ha='right')

    # left: model on DFT, intended saddle
    x = np.linspace(0.5, 4.2, 200)
    yd = 1.4 + bump(x, 2.35, 0.62, 2.0) - 0.3 * (x - 0.5) / 3.7
    ax.plot(x, yd, color=LOW, lw=lw)
    ax.plot(x, 1.4 + bump(x, 2.37, 0.62, 2.03) - 0.3 * (x - 0.5) / 3.7, color=MODEL, lw=lw * 0.7, ls=dash)
    ax.plot(2.37, yd.max() + 0.03, 'o', ms=4.5 * scale, color=INK, zorder=5)
    ax.text(2.35, 4.25, '✓ ≈ 90 %', fontsize=fs * 1.5, ha='center', fontweight='bold', color=LOW)

    # right: intended saddle (grey) and a saddle to an intermediate where the search ends
    xa = np.linspace(5.4, 9.3, 200)
    ax.plot(xa, 2.0 + bump(xa, 7.8, 0.45, 1.55) - 0.35 * (xa - 5.4) / 3.9, color=INK2, lw=lw * 0.8)
    xb = np.linspace(5.4, 7.1, 150)
    yb = 2.0 + bump(xb, 6.1, 0.3, 1.0) - 0.6 / (1 + np.exp(-(xb - 6.48) / 0.12))
    ybm = 2.0 + bump(xb, 6.26, 0.3, 1.4) - 0.6 / (1 + np.exp(-(xb - 6.6) / 0.12))
    ax.plot(xb, yb, color=HIGH, lw=lw)
    ax.plot(xb, ybm, color=MODEL, lw=lw * 0.7, ls=dash)
    j = np.argmax(ybm)
    ax.plot(xb[j], ybm[j], 'o', ms=4.5 * scale, color=INK, zorder=5)
    ax.text(7.45, 4.25, '✗ ≈ 25 %', fontsize=fs * 1.5, ha='center', fontweight='bold', color=HIGH)
    ax.text(7.45, 1.15, 'model errors ≈ 3× larger', fontsize=fs * 0.85, ha='center', color=MODEL)

    ax.text(5.0, 5.0, 'intended transition state found', fontsize=fs * 0.95, ha='center', color=INK)
    fig.savefig(f'{OUT}/{name}.png', dpi=dpi)
    if pdf:
        fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig)
    print('wrote', name)


draw(1, 'toc_final', 600, pdf=True)
draw(1, 'toc_final_actual', 150)
draw(3, 'toc_final_large', 200)
