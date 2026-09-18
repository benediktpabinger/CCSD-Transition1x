"""Results version of the residual-force protocol figure, v2: the three steps at
the stored Transition1x transition states, each box with the median residual
force of the two groups and the factor between steps. The re-optimisation
control is in Table tab:ladder and the text only.
Numbers are the medians of Table tab:ladder (results/hinge_t1x.csv, hinge_tables.py).
Output: Pictures/fig_ladder_results_v2.png

Run: python pipeline/plot_ladder_results_v2.py   (from the repo root)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10.5, 3.3))
ax.set_xlim(0, 20.2)
ax.set_ylim(-2.1, 2.6)
ax.axis("off")

GREY, BLUE, RED, INK, DIM = "#e6e6e6", "#dbe8f5", "#f6dada", "#222222", "#555555"
CS, BS = "#1f6f8b", "#b03a2e"   # closed-shell / broken-symmetry colours
W, H = 4.3, 2.25
XS = [0.3, 5.4, 10.5, 15.6]
Y = 0.0
BIG, SMALL = 9.8, 7.6

# medians of the residual force [eV/A], as printed in Table tab:ladder
N = {"s1": ("0.0147", "0.0138"), "s2": ("0.609", "0.589"), "s3": ("0.609", "1.636")}

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

def numbers(x, y, key, note=None):
    cs, bs = N[key]
    ax.text(x + W / 2, y - 0.22, f"closed-shell  {cs}", ha="center", va="top",
            fontsize=8.2, color=CS, fontweight="bold")
    ax.text(x + W / 2, y - 0.62, f"broken-symmetry  {bs}", ha="center", va="top",
            fontsize=8.2, color=BS, fontweight="bold")
    if note:
        ax.text(x + W / 2, y - 1.02, note, ha="center", va="top", fontsize=SMALL,
                style="italic", color=INK)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.4, color=INK,
                                 shrinkA=0, shrinkB=0))

MEAS = "median residual force [eV/Å] at"

box(XS[0], Y, "geometries", ["Transition1x", "transition states"],
    small_above="27 closed-shell, 18 broken-symmetry", small_below="as stored in the dataset")
box(XS[1], Y, "step 1", ["ωB97X/6-31G(d)", "restricted"], small_above=MEAS)
box(XS[2], Y, "step 2", ["ωB97M-V/def2-TZVPD", "restricted"], small_above=MEAS, fc=BLUE)
box(XS[3], Y, "step 3", ["ωB97M-V/def2-TZVPD", "unrestricted"], small_above=MEAS,
    small_below="with stability analysis", fc=RED)
numbers(XS[1], Y, "s1", "relaxed here: 44 of 45 below 0.05")
numbers(XS[2], Y, "s2", "level change: × 40\nboth groups alike")
numbers(XS[3], Y, "s3", "spin formalism: × 2.8\nbroken-symmetry only")
for a, b in zip(XS[:-1], XS[1:]):
    arrow(a + W, Y + H / 2, b, Y + H / 2)

fig.savefig("Pictures/fig_ladder_results_v2.png", dpi=220, bbox_inches="tight")
print("written Pictures/fig_ladder_results_v2.png")
