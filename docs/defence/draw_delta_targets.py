"""Two levels of theory along a reaction path: what MACE is trained on and
what the correction head is trained on.

Run from the repo root:  python docs/defence/draw_delta_targets.py
Output: docs/defence/pics/pic_delta_targets.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = "docs/defence/pics/"
NEW, MID, DARK, NAVY, GREY = "#e0731f", "#7f7f86", "#3a3a40", "#030f4f", "#8c8c92"
x = np.linspace(0, 1, 400)
cheap = 0.8 * np.exp(-((x - 0.5) ** 2) / 0.05) - 0.25 * x + 0.12
gap = 0.42 + 0.08 * np.exp(-((x - 0.5) ** 2) / 0.04)           # nearly constant, a little larger at the barrier
expensive = cheap + gap

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.plot(x, cheap, color=GREY, lw=3, label="cheap level:  " + chr(969) + "B97X/6-31G(d)")
ax.plot(x, expensive, color=NEW, lw=3, label="expensive level:  " + chr(969) + "B97M-V/def2-TZVP")
xs = np.linspace(0.06, 0.94, 10)
ax.scatter(xs, np.interp(xs, x, cheap), s=55, color="white", edgecolor=GREY, lw=1.8, zorder=4)
for k in (2, 4, 7):
    xk = xs[k]; y0 = np.interp(xk, x, cheap); y1 = np.interp(xk, x, expensive)
    ax.add_patch(FancyArrowPatch((xk, y0 + 0.02), (xk, y1 - 0.02), arrowstyle="-|>", mutation_scale=14, lw=2, color=NAVY, zorder=5))
    ax.scatter([xk], [y1], s=55, color=NEW, edgecolor=DARK, lw=1.2, zorder=6)
xm = xs[4]; ym = 0.5 * (np.interp(xm, x, cheap) + np.interp(xm, x, expensive))
ax.annotate(chr(916) + ": training target of" + chr(10) + "the correction head", (xm + 0.012, ym), xytext=(0.72, 1.05), fontsize=10.5, color=NAVY, fontweight="bold",
            ha="left", va="center", linespacing=1.3, arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2, shrinkB=4))
xg = xs[1]; yg = np.interp(xg, x, cheap)
ax.add_patch(FancyArrowPatch((xg, yg - 0.30), (xg, yg - 0.03), arrowstyle="-|>", mutation_scale=14, lw=2, color=GREY, zorder=5))
ax.text(xg, yg - 0.34, "training target" + chr(10) + "of MACE", fontsize=10.5, color=DARK, fontweight="bold", ha="center", va="top", linespacing=1.3)
# labels, below the plot
ax.text(0.0, -0.74, "MACE is trained on the cheap level:", fontsize=11, color=DARK, fontweight="bold", va="center", transform=ax.transData)
ax.text(0.0, -0.87, "every geometry of Transition1x, 9.6 million", fontsize=10.5, color=GREY, va="center")
ax.text(0.0, -1.04, "the correction head is trained on the difference " + chr(916) + ":", fontsize=11, color=NAVY, fontweight="bold", va="center")
ax.text(0.0, -1.17, "a few geometries relabelled at the expensive level, under 1 %", fontsize=10.5, color=NAVY, va="center")
ax.text(0.0, -1.36, chr(916) + " is nearly constant along the path: a small offset, far easier to learn than the surface itself", fontsize=10.5, color=NAVY, va="center", style="italic")
ax.set_xlim(0, 1); ax.set_ylim(-1.48, 1.75)
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.5, -0.56, "reaction coordinate", fontsize=11, color=MID, ha="center", va="center"); ax.set_ylabel("energy", fontsize=11, color=MID, y=0.66)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#bbbbbb")
ax.spines["bottom"].set_position(("data", -0.5)); ax.spines["left"].set_bounds(-0.5, 1.75)
ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=[GREY, NEW])
fig.savefig(OUT + "pic_delta_targets.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_delta_targets.png")
