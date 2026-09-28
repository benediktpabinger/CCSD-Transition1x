"""Three small drawings for the introduction of the defence talk.

pic_dataset.png : a reaction path, ten geometries, a label (E, F) at each.
pic_rq1.png     : the same, a few labels swapped for better ones.
pic_rq2.png     : the same, all labels from another surface whose transition
                  state sits elsewhere; the geometries are not moved.

Run from the repo root:  python docs/defence/draw_dataset_pics.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "docs/defence/pics/"
INK, GREY, TAG, NEW, NAVY = "#222222", "#9a9a9a", "#d9d9de", "#e0731f", "#030f4f"


def profile(x):
    # reactant at 0, transition state at 0.5, product at 1 (lower)
    return 1.0 * np.exp(-((x - 0.5) ** 2) / 0.045) - 0.35 * x


def unrestricted(x):
    # lies below the restricted profile around the barrier, maximum shifted
    return profile(x) - 0.75 * np.exp(-((x - 0.60) ** 2) / 0.035)


def base(ax, show_curve=True):
    x = np.linspace(0, 1, 400)
    if show_curve:
        ax.plot(x, profile(x), color=INK, lw=2.2, zorder=2)
    xs = np.linspace(0.02, 0.98, 10)
    ax.scatter(xs, profile(xs), s=70, color="white", edgecolor=INK, lw=1.8, zorder=4)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.75, 1.55)
    ax.axis("off")
    return xs


def tag(ax, x, y, color, text="E, F", dy=0.25, fc=None):
    ax.add_patch(FancyBboxPatch((x - 0.045, y + dy - 0.06), 0.09, 0.12,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc or color, ec=color, lw=1.2, zorder=5))
    ax.text(x, y + dy, text, ha="center", va="center", fontsize=8.5,
            color="white" if fc is None and color != TAG else INK, zorder=6)
    ax.plot([x, x], [y + 0.04, y + dy - 0.07], color=color, lw=1, zorder=3)


def finish(fig, name):
    fig.savefig(OUT + name, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("written", OUT + name)


# 1. dataset = geometries + labels
fig, ax = plt.subplots(figsize=(6.4, 3.6))
xs = base(ax)
for x in xs:
    tag(ax, x, profile(x), TAG)
ax.text(0.5, -0.62, "geometries: where the path search put them",
        ha="center", fontsize=11, color=INK)
ax.text(0.5, 1.45, "labels: energy and force at each geometry",
        ha="center", fontsize=11, color=INK)
finish(fig, "pic_dataset.png")

# 2. RQ1: a few labels swapped for better ones
fig, ax = plt.subplots(figsize=(6.4, 3.6))
xs = base(ax)
better = {3, 4, 6}
for i, x in enumerate(xs):
    if i in better:
        tag(ax, x, profile(x), NEW)
    else:
        tag(ax, x, profile(x), TAG)
ax.text(0.5, 1.45, "a few labels recomputed at a better level",
        ha="center", fontsize=11, color=NEW)
ax.text(0.5, -0.62, "geometries unchanged", ha="center", fontsize=11, color=INK)
finish(fig, "pic_rq1.png")

# 3. RQ2: all labels from another surface, geometries not moved
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.linspace(0, 1, 400)
xs = base(ax, show_curve=False)
ax.plot(x, profile(x), color=GREY, lw=1.6, ls="--", zorder=1)
ax.plot(x, unrestricted(x), color=NEW, lw=2.2, zorder=2)
for xi in xs:
    tag(ax, xi, profile(xi), NEW)
xm = x[np.argmax(unrestricted(x))]
ax.scatter([xm], [unrestricted(xm)], marker="*", s=180, color=NEW, zorder=6)
ax.annotate("transition state of\nthe new surface", (xm, unrestricted(xm)),
            xytext=(0.18, 1.1), fontsize=10, color=NEW, ha="center",
            arrowprops=dict(arrowstyle="-", color=NEW, lw=1))
ax.text(0.5, 1.45, "all labels from another surface",
        ha="center", fontsize=11, color=NEW)
ax.text(0.5, -0.62, "geometries unchanged, placed by the old surface",
        ha="center", fontsize=11, color=INK)
finish(fig, "pic_rq2.png")
