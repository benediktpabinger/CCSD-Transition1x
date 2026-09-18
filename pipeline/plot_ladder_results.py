"""Flow diagram of the residual-force protocol (Section 4.1.5), results version:
same layout as plot_ladder_protocol_v3.py, but every box carries the measured
median residual force of the two groups, and the arrows carry the factors.
Numbers are the medians of Table tab:ladder (results/hinge_t1x.csv, hinge_tables.py).
Output: Pictures/fig_ladder_results.png

Run: python pipeline/plot_ladder_results.py   (from the repo root)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fig, ax = plt.subplots(figsize=(11.2, 6.4))
ax.set_xlim(-1.4, 20.2)
ax.set_ylim(-1.1, 9.3)
ax.axis("off")

GREY, BLUE, RED, INK, DIM = "#e6e6e6", "#dbe8f5", "#f6dada", "#222222", "#555555"
CS, BS = "#1f6f8b", "#b03a2e"   # closed-shell / broken-symmetry colours
W, H = 4.3, 2.25
XS = [0.3, 5.4, 10.5, 15.6]
Y1, Y2 = 6.9, 1.3
BIG, SMALL = 9.8, 7.6

# medians of the residual force [eV/A], Table tab:ladder
N = {"s1": ("0.0147", "0.0138"), "s2": ("0.609", "0.589"), "s3": ("0.609", "1.636"),
     "r2": ("0.039", "0.042"), "r3": ("0.039", "1.87")}   # as printed in Table tab:ladder

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
    """Two coloured lines under a box: closed-shell and broken-symmetry medians."""
    cs, bs = N[key]
    fmt = str
    ax.text(x + W / 2, y - 0.22, f"closed-shell  {fmt(cs)}", ha="center", va="top",
            fontsize=8.2, color=CS, fontweight="bold")
    ax.text(x + W / 2, y - 0.62, f"broken-symmetry  {fmt(bs)}", ha="center", va="top",
            fontsize=8.2, color=BS, fontweight="bold")
    if note:
        ax.text(x + W / 2, y - 1.02, note, ha="center", va="top", fontsize=SMALL,
                style="italic", color=INK)

def arrow(x0, y0, x1, y1, label=None, above=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.4, color=INK,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2 - 0.15, (y0 + y1) / 2, label, ha="right",
                va="center", fontsize=SMALL, color=INK)
    if above:
        ax.text((x0 + x1) / 2, y0 + 0.18, above, ha="center", va="bottom",
                fontsize=SMALL, color=INK, linespacing=1.15)

MEAS = "median residual force [eV/Å] at"

# ---- swim lanes: one per set of geometries
def lane(y_top, y_bot, label):
    ax.add_patch(FancyBboxPatch((-1.3, y_bot), 21.4, y_top - y_bot,
                                boxstyle="round,pad=0.0,rounding_size=0.12",
                                fc="#f7f7f7", ec="#bbbbbb", lw=0.9, zorder=-5))
    ax.text(-0.95, (y_top + y_bot) / 2, label, rotation=90, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=DIM)

lane(Y1 + H + 0.25, Y1 - 1.75, "Transition1x geometries\nas stored")
# (labels kept short; the boxes carry the level)
lane(Y2 + H + 0.25, Y2 - 1.75, "re-optimised\ngeometries")

# ---- row 1: geometries as stored
box(XS[0], Y1, "geometries", ["Transition1x", "transition states"],
    small_above="27 closed-shell, 18 broken-symmetry", small_below="as stored in the dataset")
box(XS[1], Y1, "step 1", ["ωB97X-D3/6-31G(d)", "restricted"], small_above=MEAS)
box(XS[2], Y1, "step 2", ["ωB97M-V/def2-TZVPD", "restricted"], small_above=MEAS, fc=BLUE)
box(XS[3], Y1, "step 3", ["ωB97M-V/def2-TZVPD", "unrestricted"], small_above=MEAS,
    small_below="with stability analysis", fc=RED)
numbers(XS[1], Y1, "s1", "relaxed here: 44 of 45 below 0.05")
numbers(XS[2], Y1, "s2", "level change: × 40\nboth groups alike")
numbers(XS[3], Y1, "s3", "spin formalism: × 2.8\nbroken-symmetry only")
for a, b in zip(XS[:-1], XS[1:]):
    arrow(a + W, Y1 + H / 2, b, Y1 + H / 2)

# ---- row 2: re-optimised
box(XS[0], Y2, "re-optimised", ["ωB97M-V/def2-TZVPD", "restricted"],
    small_above="CI-NEB at", small_below="33 of 45 converged: 18 + 15")
box(XS[2], Y2, "step 2", ["ωB97M-V/def2-TZVPD", "restricted"], small_above=MEAS, fc=BLUE)
box(XS[3], Y2, "step 3", ["ωB97M-V/def2-TZVPD", "unrestricted"], small_above=MEAS,
    small_below="with stability analysis", fc=RED)
numbers(XS[2], Y2, "r2", "level change removed")
numbers(XS[3], Y2, "r3", "spin formalism: × 32\nbroken-symmetry only")
LANE1_BOT, LANE2_TOP = Y1 - 1.75, Y2 + H + 0.25
arrow(XS[0] + W / 2, LANE1_BOT, XS[0] + W / 2, LANE2_TOP, label="re-optimise")

# ---- icon: atoms shifting (hollow = before, filled = after), right of the arrow
ICON = "#444444"
s = 0.72
ix, iy = XS[0] + W / 2 + 1.6, (LANE1_BOT + LANE2_TOP) / 2
before = [(-0.85 * s, 0.42 * s), (0.00, -0.45 * s), (0.85 * s, 0.38 * s)]
shift = [(-0.20 * s, -0.62 * s), (0.00, 0.66 * s), (0.30 * s, -0.58 * s)]
for (bx, by), (dx, dy) in zip(before, shift):
    ax.add_patch(Circle((ix + bx, iy + by), 0.19 * s, fc="white", ec=ICON, lw=1.1, ls=(0, (2, 2))))
    L = (dx**2 + dy**2) ** 0.5; ux, uy = dx / L, dy / L; r = 0.22 * s
    ax.add_patch(FancyArrowPatch((ix + bx + r * ux, iy + by + r * uy),
                                 (ix + bx + dx - r * ux, iy + by + dy - r * uy),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.2, color=ICON,
                                 shrinkA=0, shrinkB=0, zorder=3))
    ax.add_patch(Circle((ix + bx + dx, iy + by + dy), 0.19 * s, fc=ICON, ec=ICON, lw=1.0))
after = [(ix + bx + dx, iy + by + dy) for (bx, by), (dx, dy) in zip(before, shift)]
for (x0, y0), (x1, y1) in zip(after[:-1], after[1:]):
    ax.plot([x0, x1], [y0, y1], color=ICON, lw=1.3, zorder=0)

arrow(XS[0] + W, Y2 + H / 2, XS[2], Y2 + H / 2)
arrow(XS[2] + W, Y2 + H / 2, XS[3], Y2 + H / 2)

fig.savefig("Pictures/fig_ladder_results.png", dpi=220, bbox_inches="tight")
print("written Pictures/fig_ladder_results.png")
