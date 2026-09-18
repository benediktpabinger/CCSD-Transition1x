"""Flow diagram of the residual-force protocol (Section 4.1.5).
Output: Pictures/fig_ladder_protocol.png

Run: python pipeline/plot_ladder_protocol.py   (from the repo root)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.5, 5.0))
ax.set_xlim(0, 20.2)
ax.set_ylim(0.9, 9.3)
ax.axis("off")

GREY, BLUE, RED, INK = "#e6e6e6", "#dbe8f5", "#f6dada", "#222222"
W, H = 4.3, 2.25            # box size
XS = [0.3, 5.4, 10.5, 15.6] # column x positions
Y1, Y2 = 6.9, 2.3           # row baselines

def box(x, y, title, lines, fc, w=W, h=H):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec=INK, lw=1.1))
    ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + 0.85, "\n".join(lines), ha="center", va="center",
            fontsize=8.6, color=INK, linespacing=1.25)

def result(x, y, text, color=INK):
    ax.text(x + W / 2, y - 0.25, text, ha="center", va="top", fontsize=8.8,
            style="italic", color=color, linespacing=1.25)

def arrow(x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.4, color=INK,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2 - 0.15, (y0 + y1) / 2, label, ha="right",
                va="center", fontsize=8.8, color=INK)

# ---- row 1: geometries as stored
box(XS[0], Y1, "geometries", ["45 transition states", "as stored in Transition1x"], GREY)
box(XS[1], Y1, "step 1", ["residual force at", "ωB97X-D3/6-31G(d)", "restricted"], GREY)
box(XS[2], Y1, "step 2", ["residual force at", "ωB97M-V/def2-TZVPD", "restricted"], BLUE)
box(XS[3], Y1, "step 3", ["residual force at", "ωB97M-V/def2-TZVPD", "ground-state surface", "(stability analysis)"], RED)
result(XS[1], Y1, "force ≈ 0\nrelaxed here")
result(XS[2], Y1, "force =\nlevel change")
result(XS[3], Y1, "force = level change\n+ surface change")
for a, b in zip(XS[:-1], XS[1:]):
    arrow(a + W, Y1 + H / 2, b, Y1 + H / 2)

# ---- row 2: re-optimised
box(XS[0], Y2, "re-optimised", ["CI-NEB at", "ωB97M-V/def2-TZVPD", "restricted"], GREY)
box(XS[2], Y2, "step 2", ["residual force at", "ωB97M-V/def2-TZVPD", "restricted"], BLUE)
box(XS[3], Y2, "step 3", ["residual force at", "ωB97M-V/def2-TZVPD", "ground-state surface"], RED)
result(XS[2], Y2, "force ≈ 0\nlevel change removed")
result(XS[3], Y2, "force =\nsurface change only")
arrow(XS[0] + W / 2, Y1 - 0.75, XS[0] + W / 2, Y2 + H, label="re-optimise")
arrow(XS[0] + W, Y2 + H / 2, XS[2], Y2 + H / 2)
arrow(XS[2] + W, Y2 + H / 2, XS[3], Y2 + H / 2)

fig.savefig("Pictures/fig_ladder_protocol.png", dpi=220, bbox_inches="tight")
print("written Pictures/fig_ladder_protocol.png")
