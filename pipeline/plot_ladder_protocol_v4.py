"""Flow diagram of the residual-force protocol (Section 4.1.5), v4: v3 with two
swim lanes, one for the stored Transition1x geometries and one for the
re-optimised ones. Same layout as plot_ladder_results.py, without the numbers.
Output: Pictures/fig_ladder_protocol_v4.png

Run: python pipeline/plot_ladder_protocol_v4.py   (from the repo root)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fig, ax = plt.subplots(figsize=(11.2, 5.9))
ax.set_xlim(-1.4, 20.2)
ax.set_ylim(-0.4, 9.3)
ax.axis("off")

GREY, BLUE, RED, INK, DIM = "#e6e6e6", "#dbe8f5", "#f6dada", "#222222", "#555555"
W, H = 4.3, 2.25
XS = [0.3, 5.4, 10.5, 15.6]
Y1, Y2 = 6.9, 1.6
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

def arrow(x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.4, color=INK,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2 - 0.15, (y0 + y1) / 2, label, ha="right",
                va="center", fontsize=SMALL, color=INK)

MEAS = "measure residual force at"

# ---- swim lanes: one per set of geometries
def lane(y_top, y_bot, label):
    ax.add_patch(FancyBboxPatch((-1.3, y_bot), 21.4, y_top - y_bot,
                                boxstyle="round,pad=0.0,rounding_size=0.12",
                                fc="#f7f7f7", ec="#bbbbbb", lw=0.9, zorder=-5))
    ax.text(-0.95, (y_top + y_bot) / 2, label, rotation=90, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=DIM)

LANE1_BOT, LANE2_TOP = Y1 - 1.15, Y2 + H + 0.25
lane(Y1 + H + 0.25, LANE1_BOT, "Transition1x geometries\nas stored")
lane(LANE2_TOP, Y2 - 1.15, "re-optimised\ngeometries")

# ---- row 1: geometries as stored
box(XS[0], Y1, "geometries", ["Transition1x", "transition states"],
    small_above="45 geometries", small_below="as stored in the dataset")
box(XS[1], Y1, "step 1", ["ωB97X-D3/6-31G(d)", "restricted"], small_above=MEAS)
box(XS[2], Y1, "step 2", ["ωB97M-V/def2-TZVPD", "restricted"], small_above=MEAS, fc=BLUE)
box(XS[3], Y1, "step 3", ["ωB97M-V/def2-TZVPD", "unrestricted"], small_above=MEAS,
    small_below="with stability analysis", fc=RED)
result(XS[1], Y1, "force ≈ 0\nrelaxed here")
result(XS[2], Y1, "force =\nlevel change")
result(XS[3], Y1, "force = level change\n+ spin-formalism change")
for a, b in zip(XS[:-1], XS[1:]):
    arrow(a + W, Y1 + H / 2, b, Y1 + H / 2)

# ---- row 2: re-optimised
box(XS[0], Y2, "re-optimised", ["ωB97M-V/def2-TZVPD", "restricted"],
    small_above="CI-NEB at", small_below="45 geometries")
box(XS[2], Y2, "step 2", ["ωB97M-V/def2-TZVPD", "restricted"], small_above=MEAS, fc=BLUE)
box(XS[3], Y2, "step 3", ["ωB97M-V/def2-TZVPD", "unrestricted"], small_above=MEAS,
    small_below="with stability analysis", fc=RED)
result(XS[2], Y2, "force ≈ 0\nlevel change removed")
result(XS[3], Y2, "force =\nspin-formalism change only")
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

fig.savefig("Pictures/fig_ladder_protocol_v4.png", dpi=220, bbox_inches="tight")
print("written Pictures/fig_ladder_protocol_v4.png")
