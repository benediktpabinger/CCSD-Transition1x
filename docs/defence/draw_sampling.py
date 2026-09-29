# -*- coding: utf-8 -*-
"""Die geschichtete Auswahl entlang eines Reaktionspfades, fuer den Vortrag.

figures/stratified_sampling.png (die Fassung der Arbeit) sagt dasselbe, aber
in Blau. Im Foliensatz ist Orange durchgehend die Farbe fuer 'das, was wir
getan haben', und die ausgewaehlten Geometrien sind genau das. Ausserdem
sitzen die Beschriftungen hier groesser, weil das Bild auf der Folie nur rund
sechs Zoll breit ist.

Die Aussage: nicht 20 von 950 irgendwo, sondern 20 entlang der Form des
Pfades. Die beiden Abschnitte am Sattel sind kurz, also liegt die Haelfte der
Auswahl dort -- und der Uebergangszustand selbst liegt auf einer
Abschnittsgrenze und ist deshalb immer dabei.

    python docs/defence/draw_sampling.py docs/defence/pics
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#030F4F"
ORANGE = "#E0731F"
INK = "#3A3A40"
DIM = "#7F7F86"
BAND = ("#F5F5F7", "#E9E9EE")


def energy(x):
    return (np.exp(-((x - 0.50) / 0.21) ** 2)
            - 0.55 / (1.0 + np.exp(-(x - 0.58) / 0.07)))


xs = np.linspace(0, 1, 800)
ys = energy(xs)
xts = float(xs[int(np.argmax(ys))])
mid = 0.5 * (xts + 1.0)
edges = [0.0, 0.25, xts, mid, 1.0]
names = ["before the climb", "climb", "descent", "down to product"]

fig, ax = plt.subplots(figsize=(11.4, 3.95))
lo, hi = ys.min(), ys.max()
pad = 0.26 * (hi - lo)
ax.set_xlim(0, 1)
ax.set_ylim(lo - 1.05 * pad, hi + 0.62 * pad)

for i in range(4):
    ax.axvspan(edges[i], edges[i + 1], color=BAND[i % 2], lw=0, zorder=0)

ax.plot(xs, ys, color=INK, lw=2.6, zorder=3, solid_capstyle="round")

# fuenf gleichmaessig verteilte Punkte je Abschnitt
for i in range(4):
    for x in np.linspace(edges[i], edges[i + 1], 5):
        ax.plot([x], [energy(x)], "o", ms=9, mfc=ORANGE, mec="white",
                mew=1.4, zorder=5)

# Der Uebergangszustand liegt auf einer Grenze und ist deshalb immer dabei.
ax.plot([xts], [energy(xts)], "o", ms=16, mfc=ORANGE, mec=NAVY, mew=2.0,
        zorder=6)
ax.axvline(xts, color=DIM, ls="--", lw=1.1, zorder=1)
ax.annotate("the transition state lies on a\nboundary, so it is always in",
            (xts, energy(xts)), xytext=(xts - 0.045, hi + 0.50 * pad),
            textcoords="data", ha="right", va="top", fontsize=13.5,
            color=NAVY, linespacing=1.35,
            arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.1, shrinkB=10))

yb = lo - 0.52 * pad
for i in range(4):
    c = 0.5 * (edges[i] + edges[i + 1])
    ax.text(c, yb, names[i], ha="center", va="center", fontsize=12.5,
            style="italic", color=DIM)
    ax.text(c, yb - 0.30 * pad, "5 points", ha="center", va="center",
            fontsize=15, fontweight="bold", color=ORANGE)

ax.text(0.012, energy(0.0) + 0.10 * pad, "reactant", ha="left", va="bottom",
        fontsize=12.5, color=DIM)
ax.text(0.988, energy(1.0) + 0.10 * pad, "product", ha="right", va="bottom",
        fontsize=12.5, color=DIM)

ax.set_xlabel("reaction path   (frame 0 … ~950)", fontsize=13,
              color=INK, labelpad=9)
ax.set_ylabel("energy", fontsize=13, color=INK)
ax.set_xticks([]); ax.set_yticks([])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#C8C8CE")

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
p = os.path.join(out, "sampling.png")
fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
print("geschrieben:", p, "  Sattel bei Rahmen", round(xts * 950))
