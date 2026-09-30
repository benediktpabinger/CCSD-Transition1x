"""The simplest reaction picture: HCN -> HNC, three atoms. Reactant,
transition state and product as flat ball-and-stick drawings.

Run from the repo root:  python docs/defence/draw_molecules_simple.py
Output: docs/defence/pics/simple_reactant.png, simple_ts.png, simple_product.png, simple_strip.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = "docs/defence/pics/"
DARK, NAVY, MID = "#3a3a40", "#030f4f", "#7f7f86"
RAD = {"H": 0.30, "C": 0.46, "N": 0.46}
FILL = {"H": "#e3e6f3", "C": "#8c8c92", "N": "#b9bfd6"}
C, N = (0.0, 0.0), (1.55, 0.0)
states = {
    "reactant": dict(H=(-1.35, 0.0), solid=[("H", "C")], dashed=[]),
    "transition state": dict(H=(0.775, 1.25), solid=[], dashed=[("H", "C"), ("H", "N")]),
    "product": dict(H=(2.9, 0.0), solid=[("H", "N")], dashed=[]),
}


def draw(ax, st, title):
    pos = {"C": C, "N": N, "H": st["H"]}
    # the C-N bond, triple: three parallel lines
    for dy in (-0.13, 0.0, 0.13):
        ax.plot([C[0], N[0]], [C[1] + dy, N[1] + dy], color="#6f6f75", lw=3.2, solid_capstyle="round", zorder=1)
    for a, b in st["solid"]:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#6f6f75", lw=5, solid_capstyle="round", zorder=1)
    for a, b in st["dashed"]:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=NAVY, lw=3, ls=(0, (2, 1.6)), zorder=1)
    for el in ("C", "N", "H"):
        hot = el == "H"
        ax.add_patch(Circle(pos[el], RAD[el], fc=FILL[el], ec=NAVY if hot else DARK, lw=2.2 if hot else 1.5, zorder=2))
        ax.text(pos[el][0], pos[el][1], el, ha="center", va="center", fontsize=15 if el != "H" else 12,
                color="white" if el == "C" else (NAVY if hot else DARK), fontweight="bold", zorder=3)
    ax.set_xlim(-2.0, 3.55); ax.set_ylim(-1.35, 1.85); ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.775, -1.2, title, ha="center", va="center", fontsize=14, fontweight="bold", color=MID)


for title, st in states.items():
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    draw(ax, st, title)
    name = "simple_" + {"reactant": "reactant", "transition state": "ts", "product": "product"}[title] + ".png"
    fig.savefig(OUT + name, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("written", OUT + name)
fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.2), gridspec_kw={"wspace": 0.02})
for ax, (title, st) in zip(axes, states.items()):
    draw(ax, st, title)
fig.savefig(OUT + "simple_strip.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "simple_strip.png")
