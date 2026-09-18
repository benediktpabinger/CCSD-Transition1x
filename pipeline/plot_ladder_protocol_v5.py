"""Flow diagram of the residual-force protocol (Section 4.1.5), v5: the three
steps at the stored Transition1x transition states only. The re-optimisation
is a control and is described in the text, not drawn.
Output: Pictures/fig_ladder_protocol_v5.png

Run: python pipeline/plot_ladder_protocol_v5.py   (from the repo root)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.5, 2.9))
ax.set_xlim(0, 20.2)
ax.set_ylim(-1.6, 2.6)
ax.axis("off")

GREY, BLUE, RED, INK, DIM = "#e6e6e6", "#dbe8f5", "#f6dada", "#222222", "#555555"
W, H = 4.3, 2.25
XS = [0.3, 5.4, 10.5, 15.6]
Y = 0.0
BIG, SMALL = 9.8, 7.6

def box(x, y, title, big, small_above=None, small_below=None, fc=GREY, w=W, h=H):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec=INK, lw=1.1))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=INK)
    cy = y + 0.95
    if small_above:
        ax.text(x + w / 2, cy + 0.62, small_above, ha="center", va="center",
                fontsize=SMALL, color=DIM)
    ax.text(x + w / 2, cy, "\n".join(big), ha="center", va="center",
            fontsize=BIG, color=INK, linespacing=1.15)
    if small_below:
        ax.text(x + w / 2, cy - 0.62, small_below, ha="center", va="center",
                fontsize=SMALL, color=DIM)

def result(x, y, text):
    ax.text(x + W / 2, y - 0.25, text, ha="center", va="top", fontsize=SMALL,
            style="italic", color=INK, linespacing=1.25)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.4, color=INK,
                                 shrinkA=0, shrinkB=0))

MEAS = "measure residual force at"

box(XS[0], Y, "geometries", ["Transition1x", "transition states"],
    small_above="45 geometries", small_below="as stored in the dataset")
box(XS[1], Y, "step 1", ["ωB97X/6-31G(d)", "restricted"], small_above=MEAS)
box(XS[2], Y, "step 2", ["ωB97M-V/def2-TZVPD", "restricted"], small_above=MEAS, fc=BLUE)
box(XS[3], Y, "step 3", ["ωB97M-V/def2-TZVPD", "unrestricted"], small_above=MEAS,
    small_below="with stability analysis", fc=RED)
result(XS[1], Y, "force ≈ 0\nrelaxed here")
result(XS[2], Y, "force =\nlevel change")
result(XS[3], Y, "force = level change\n+ spin-formalism change")
for a, b in zip(XS[:-1], XS[1:]):
    arrow(a + W, Y + H / 2, b, Y + H / 2)

fig.savefig("Pictures/fig_ladder_protocol_v5.png", dpi=220, bbox_inches="tight")
print("written Pictures/fig_ladder_protocol_v5.png")
