"""A hydrogen transfer on one line: C-H ... O  ->  C ... H ... O  ->  C ... H-O.
Reactant, transition state and product as flat ball-and-stick drawings; short
stubs stand for the rest of the molecule.

Run from the repo root:  python docs/defence/draw_molecules_transfer.py
Output: docs/defence/pics/transfer_reactant.png, transfer_ts.png, transfer_product.png, transfer_strip.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = "docs/defence/pics/"
DARK, NAVY, MID = "#3a3a40", "#030f4f", "#7f7f86"
RAD = {"H": 0.30, "C": 0.46, "O": 0.46}
FILL = {"H": "#e3e6f3", "C": "#8c8c92", "O": "#d9d9de"}
C, O = (0.0, 0.0), (3.1, 0.0)
states = {"reactant": 1.05, "transition state": 1.55, "product": 2.05}


def draw(ax, xh, title):
    # stubs: the rest of the molecule
    for (x0, y0), (dx, dy) in [(C, (-0.85, 0.6)), (C, (-0.85, -0.6)), (O, (0.85, 0.55))]:
        ax.plot([x0, x0 + dx], [y0, y0 + dy], color="#b8b8bd", lw=5, solid_capstyle="round", zorder=0)
    H = (xh, 0.0)
    if title == "reactant":
        ax.plot([C[0], H[0]], [0, 0], color="#6f6f75", lw=5, solid_capstyle="round", zorder=1)
    elif title == "product":
        ax.plot([H[0], O[0]], [0, 0], color="#6f6f75", lw=5, solid_capstyle="round", zorder=1)
    else:
        ax.plot([C[0], H[0]], [0, 0], color=NAVY, lw=3, ls=(0, (2, 1.6)), zorder=1)
        ax.plot([H[0], O[0]], [0, 0], color=NAVY, lw=3, ls=(0, (2, 1.6)), zorder=1)
    for el, p in (("C", C), ("O", O), ("H", H)):
        hot = el == "H"
        ax.add_patch(Circle(p, RAD[el], fc=FILL[el], ec=NAVY if hot else DARK, lw=2.2 if hot else 1.5, zorder=2))
        ax.text(p[0], p[1], el, ha="center", va="center", fontsize=15 if el != "H" else 12,
                color="white" if el == "C" else (NAVY if hot else DARK), fontweight="bold", zorder=3)
    ax.set_xlim(-1.3, 4.4); ax.set_ylim(-1.45, 1.0); ax.set_aspect("equal"); ax.axis("off")
    ax.text(1.55, -1.25, title, ha="center", va="center", fontsize=14, fontweight="bold", color=MID)


short = {"reactant": "reactant", "transition state": "ts", "product": "product"}
for title, xh in states.items():
    fig, ax = plt.subplots(figsize=(3.8, 1.7))
    draw(ax, xh, title)
    fig.savefig(OUT + "transfer_" + short[title] + ".png", dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("written", OUT + "transfer_" + short[title] + ".png")
fig, axes = plt.subplots(1, 3, figsize=(11.4, 1.7), gridspec_kw={"wspace": 0.02})
for ax, (title, xh) in zip(axes, states.items()):
    draw(ax, xh, title)
fig.savefig(OUT + "transfer_strip.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "transfer_strip.png")
