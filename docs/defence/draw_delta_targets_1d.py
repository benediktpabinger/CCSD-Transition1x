"""One geometry, two energies: what MACE is trained on and what the
correction head is trained on. Compact, large type.

Run from the repo root:  python docs/defence/draw_delta_targets_1d.py
Output: docs/defence/pics/pic_delta_targets_1d.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = "docs/defence/pics/"
NEW, MID, DARK, NAVY, GREY = "#e0731f", "#7f7f86", "#3a3a40", "#030f4f", "#8c8c92"
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.3); ax.axis("off")
xa = 4.2                      # the arrows
x0, x1 = 3.5, 5.0             # the level bars
y_cheap, y_exp = 2.5, 5.2
ax.plot([x0, x1], [y_cheap, y_cheap], color=GREY, lw=6, solid_capstyle="round")
ax.plot([x0, x1], [y_exp, y_exp], color=NEW, lw=6, solid_capstyle="round")
ax.text(x1 + 0.3, y_exp + 0.32, "expensive level", va="center", fontsize=16, color=NEW, fontweight="bold")
ax.text(x1 + 0.3, y_exp - 0.35, chr(969) + "B97M-V/def2-TZVP", va="center", fontsize=14, color=NEW)
ax.text(x1 + 0.3, y_cheap + 0.32, "cheap level", va="center", fontsize=16, color=GREY, fontweight="bold")
ax.text(x1 + 0.3, y_cheap - 0.35, chr(969) + "B97X/6-31G(d)", va="center", fontsize=14, color=GREY)
# MACE: ground to cheap level
ax.add_patch(FancyArrowPatch((xa, 0.55), (xa, y_cheap - 0.15), arrowstyle="-|>", mutation_scale=20, lw=3, color=GREY))
ax.text(xa - 0.4, 0.5 * (0.55 + y_cheap) + 0.25, "MACE", ha="right", va="center", fontsize=17, color=DARK, fontweight="bold")
ax.text(xa - 0.4, 0.5 * (0.55 + y_cheap) - 0.4, "learns the cheap energy", ha="right", va="center", fontsize=13, color=DARK)
# head: the difference
ax.add_patch(FancyArrowPatch((xa, y_cheap + 0.15), (xa, y_exp - 0.15), arrowstyle="-|>", mutation_scale=20, lw=3, color=NAVY))
ax.text(xa - 0.4, 0.5 * (y_cheap + y_exp) + 0.25, "correction head", ha="right", va="center", fontsize=17, color=NAVY, fontweight="bold")
ax.text(xa - 0.4, 0.5 * (y_cheap + y_exp) - 0.4, "learns the difference " + chr(916), ha="right", va="center", fontsize=13, color=NAVY)
ax.plot([x0, x1], [0.4, 0.4], color="#bbbbbb", lw=1.2)
ax.text(0.5 * (x0 + x1), 0.05, "one geometry", ha="center", va="center", fontsize=12, color=MID, style="italic")
ax.text(x1 + 0.3, 1.2, "MACE + " + chr(916) + " = expensive energy", va="center", fontsize=15, color=DARK, fontweight="bold")
fig.savefig(OUT + "pic_delta_targets_1d.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_delta_targets_1d.png")
