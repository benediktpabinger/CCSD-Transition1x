"""Two small icons for the deck: the delta head (RQ1) and OMol25 (RQ2).
Square, transparent, in the deck's colours.

Run from the repo root:  python docs/defence/draw_icons.py
Output: docs/defence/pics/icon_delta.png, icon_omol25.png (+ _navy variants)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, FancyBboxPatch

OUT = "docs/defence/pics/"
NEW, NAVY, GREY, DARK = "#e0731f", "#030f4f", "#8c8c92", "#3a3a40"


def delta_icon(name, main, accent):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
    # two levels, an arrow between them, a delta in the arrow
    ax.plot([2.2, 7.8], [2.4, 2.4], color=GREY, lw=7, solid_capstyle="round")
    ax.plot([2.2, 7.8], [7.6, 7.6], color=accent, lw=7, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((5.0, 3.1), (5.0, 6.9), arrowstyle="-|>", mutation_scale=30, lw=4, color=main))
    ax.text(6.6, 5.0, chr(916), ha="center", va="center", fontsize=30, fontweight="bold", color=main)
    fig.savefig(OUT + name, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)
    print("written", OUT + name)


def omol_icon(name, main, accent):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
    # a cloud of molecules: many small nodes, a few bonds, one highlighted
    # a hexagonal cluster of molecules: one big node, six around it, six more outside
    pts = [(5.0, 5.0)]
    for k in range(6):
        a = np.pi / 6 + k * np.pi / 3
        pts.append((5.0 + 2.3 * np.cos(a), 5.0 + 2.3 * np.sin(a)))
    for k in range(6):
        a = k * np.pi / 3
        pts.append((5.0 + 3.9 * np.cos(a), 5.0 + 3.9 * np.sin(a)))
    pts = np.array(pts)
    for i in range(1, 7):
        ax.plot([pts[0, 0], pts[i, 0]], [pts[0, 1], pts[i, 1]], color=GREY, lw=2.6, zorder=1, solid_capstyle="round")
        j = 1 + i % 6
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=GREY, lw=2.6, zorder=1, solid_capstyle="round")
    for k in range(7, 13):
        i, j = 1 + (k - 7) % 6, 1 + (k - 8) % 6
        ax.plot([pts[k, 0], pts[i, 0]], [pts[k, 1], pts[i, 1]], color=GREY, lw=2.0, zorder=1, alpha=0.8)
        ax.plot([pts[k, 0], pts[j, 0]], [pts[k, 1], pts[j, 1]], color=GREY, lw=2.0, zorder=1, alpha=0.8)
    ax.add_patch(Circle(pts[0], 0.9, fc=accent, ec="none", zorder=2))
    for k in range(1, 7):
        ax.add_patch(Circle(pts[k], 0.62, fc=main, ec="none", zorder=2))
    for k in range(7, 13):
        ax.add_patch(Circle(pts[k], 0.45, fc=main, ec="none", zorder=2, alpha=0.75))
    fig.savefig(OUT + name, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)
    print("written", OUT + name)


def omol_icon_text(name, main, accent):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.8, 0.8), 8.4, 8.4, boxstyle="round,pad=0.02,rounding_size=1.2", fc="none", ec=main, lw=4))
    ax.text(5.0, 6.1, "OMol", ha="center", va="center", fontsize=30, fontweight="bold", color=main)
    ax.text(5.0, 3.4, "25", ha="center", va="center", fontsize=30, fontweight="bold", color=accent)
    fig.savefig(OUT + name, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)
    print("written", OUT + name)


delta_icon("icon_delta.png", NAVY, NEW)
delta_icon("icon_delta_navy.png", NAVY, NAVY)
omol_icon("icon_omol25.png", NAVY, NEW)
omol_icon("icon_omol25_navy.png", NAVY, NAVY)
omol_icon_text("icon_omol25_text.png", NAVY, NEW)
