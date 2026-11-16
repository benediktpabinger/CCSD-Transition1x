"""Draft TOC graphic (ACS: at most 3.25 x 1.75 in, actual size). Schematic, not data: a low-MR
reaction with one barrier where the search finds the intended saddle, and a high-MR reaction with an
intermediate where the search stops on the saddle of the other step and model and DFT forces disagree.
The two numbers are the rounded headline results (freezing string, four models): intended saddle
about 90 % vs about 25 %, force error about 3x.

Output figures/paper/toc.{png,pdf} at 3.25 x 1.75 in, and toc_large.png for viewing.

    python pipeline/plot_toc.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(REPO, 'figures', 'paper')
INK, INK2 = '#1f1f1e', '#5f5e58'
LOW, HIGH, MODEL = '#2a78d6', '#eb6834', '#7a3fb0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})


def draw(scale, name, dpi):
    fig = plt.figure(figsize=(3.25 * scale, 1.75 * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis('off')
    fs = 6.0 * scale

    # MR axis
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=(0.6, 9.4, 0.62, 0.86), aspect='auto',
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list('mr', [LOW, HIGH]))
    ax.add_patch(FancyArrowPatch((9.2, 0.74), (9.75, 0.74), arrowstyle='-|>', mutation_scale=8 * scale, color=HIGH, lw=0))
    ax.text(0.6, 0.25, 'low multireference', fontsize=fs, color=LOW, va='center')
    ax.text(9.4, 0.25, r'high multireference ($N_\mathrm{FOD}$)', fontsize=fs, color=HIGH, va='center', ha='right')

    # left: one barrier, search on the intended saddle
    x = np.linspace(0.7, 4.3, 200)
    y = 1.6 + 2.2 * np.exp(-((x - 2.5) / 0.65) ** 2) - 0.25 * (x - 0.7) / 3.6
    ax.plot(x, y, color=LOW, lw=1.6 * scale)
    ax.plot(2.5, y[np.argmin(abs(x - 2.5))], 'o', ms=4.5 * scale, color=INK, zorder=5)
    ax.text(2.5, 4.3, '✓ intended saddle found: ≈ 90 %', fontsize=fs, ha='center', color=INK, fontweight='bold')

    # right: intermediate, search on the other step; model and DFT forces disagree
    x = np.linspace(5.7, 9.3, 200)
    y = (1.6 + 1.7 * np.exp(-((x - 6.6) / 0.38) ** 2) + 0.55 * np.exp(-((x - 7.5) / 0.35) ** 2) * 0
         + 1.9 * np.exp(-((x - 8.35) / 0.38) ** 2) - 0.6 * np.exp(-((x - 7.5) / 0.45) ** 2))
    ax.plot(x, y, color=HIGH, lw=1.6 * scale)
    i = np.argmin(abs(x - 6.6))
    ax.plot(x[i], y[i], 'o', ms=4.5 * scale, color=INK, zorder=5)
    ax.text(7.5, 4.3, '✗ intended saddle found: ≈ 25 %', fontsize=fs, ha='center', color=INK, fontweight='bold')
    ax.text(7.5, 3.95, 'often the saddle of another step', fontsize=fs * 0.85, ha='center', color=INK2)
    for dx, col, lab in ((0.55, MODEL, 'model'), (0.95, INK, 'DFT')):
        ang = 0.0 if lab == 'model' else -0.55
        ax.add_patch(FancyArrowPatch((x[i], y[i]), (x[i] + dx * np.cos(ang), y[i] + dx * np.sin(ang)),
                                     arrowstyle='-|>', mutation_scale=6 * scale, color=col, lw=0.9 * scale))
    ax.text(x[i] + 0.62, y[i] - 0.05, 'model', fontsize=fs * 0.8, color=MODEL, va='center')
    ax.text(x[i] + 0.9, y[i] - 0.62, 'DFT', fontsize=fs * 0.8, color=INK)
    ax.text(x[i] - 0.15, y[i] + 0.38, 'force error ≈ ×3', fontsize=fs * 0.85, ha='left', color=HIGH)

    ax.text(5.0, 5.05, 'OMol25 potentials at transition states', fontsize=fs * 1.05, ha='center', fontweight='bold', color=INK)
    fig.savefig(f'{OUT}/{name}.png', dpi=dpi)
    if scale == 1:
        fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig)
    print('wrote', name)


draw(1, 'toc', 600)
draw(3, 'toc_large', 200)
