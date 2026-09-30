"""One geometry, two energies: what MACE is trained on and what the
correction head is trained on. An energy-level sketch.

Run from the repo root:  python docs/defence/draw_delta_targets_1d.py
Output: docs/defence/pics/pic_delta_targets_1d.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = "docs/defence/pics/"
NEW, MID, DARK, NAVY, GREY = "#e0731f", "#7f7f86", "#3a3a40", "#030f4f", "#8c8c92"
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
x0, x1 = 4.4, 6.8
y_cheap, y_exp = 3.6, 7.2
# energy axis
ax.add_patch(FancyArrowPatch((0.9, 0.8), (0.9, 9.2), arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#bbbbbb"))
ax.text(0.55, 5.0, "energy", rotation=90, ha="center", va="center", fontsize=11, color=MID)
# the two levels at one geometry
ax.plot([x0, x1], [y_cheap, y_cheap], color=GREY, lw=4, solid_capstyle="round")
ax.plot([x0, x1], [y_exp, y_exp], color=NEW, lw=4, solid_capstyle="round")
ax.text(x1 + 0.3, y_cheap, "cheap level  " + chr(969) + "B97X/6-31G(d)", va="center", fontsize=11, color=GREY)
ax.text(x1 + 0.3, y_exp, "expensive level  " + chr(969) + "B97M-V/def2-TZVP", va="center", fontsize=11, color=NEW)
# MACE target: from the bottom up to the cheap level
ax.add_patch(FancyArrowPatch((x0 + 0.5, 0.9), (x0 + 0.5, y_cheap - 0.12), arrowstyle="-|>", mutation_scale=16, lw=2.2, color=GREY))
ax.text(x0 + 0.5 - 0.35, 0.5 * (0.9 + y_cheap) + 0.3, "MACE", ha="right", va="center", fontsize=12, color=DARK, fontweight="bold")
ax.text(x0 + 0.5 - 0.35, 0.5 * (0.9 + y_cheap) - 0.25, "training target:" + chr(10) + "the cheap energy", ha="right", va="top", fontsize=9.5, color=DARK, linespacing=1.3)
# head target: the difference
ax.add_patch(FancyArrowPatch((x0 + 0.5, y_cheap + 0.12), (x0 + 0.5, y_exp - 0.12), arrowstyle="-|>", mutation_scale=16, lw=2.2, color=NAVY))
ax.text(x0 + 0.5 - 0.35, 0.5 * (y_cheap + y_exp) + 0.55, chr(916), ha="right", va="center", fontsize=15, color=NAVY, fontweight="bold")
ax.text(x0 + 0.5 - 0.35, 0.5 * (y_cheap + y_exp) + 0.05, "correction head" + chr(10) + "training target:" + chr(10) + "the difference", ha="right", va="top", fontsize=9.5, color=NAVY, linespacing=1.3)
# geometry marker
ax.text(0.5 * (x0 + x1), 0.35, "one geometry", ha="center", va="center", fontsize=10.5, color=MID, style="italic")
ax.plot([x0, x1], [0.75, 0.75], color="#bbbbbb", lw=1)
# the sum
ax.text(x1 + 0.3, y_exp + 1.4, "MACE +  " + chr(916) + "  =  the expensive energy", va="center", fontsize=11.5, color=DARK, fontweight="bold")
fig.savefig(OUT + "pic_delta_targets_1d.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_delta_targets_1d.png")
