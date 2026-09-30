"""The kink: where the broken-symmetry solution sets in, the lowest surface
switches from the restricted branch to the unrestricted one and its curvature
changes abruptly. A smooth model rounds the corner off.

Run from the repo root:  python docs/defence/draw_kink.py
Output: docs/defence/pics/pic_kink.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/defence/pics/"
NEW, MID, DARK, NAVY = "#e0731f", "#7f7f86", "#3a3a40", "#030f4f"
x = np.linspace(0, 1, 600)
rks = np.exp(-((x - 0.5) ** 2) / 0.09)                     # restricted: a smooth hill
x0 = 0.40                                                   # where the broken-symmetry solution sets in
bs = rks - 2.2 * np.clip(x - x0, 0, None) * np.exp(-((x - 0.62) ** 2) / 0.06)
low = np.minimum(rks, bs)                                   # the unrestricted surface: the lowest solution
# a smooth model fitted to the lowest surface: heavy smoothing
k = np.exp(-0.5 * (np.linspace(-3, 3, 241)) ** 2); k /= k.sum()
pad = np.pad(low, 120, mode="edge"); model = np.convolve(pad, k, mode="same")[120:-120]

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.fill_between(x, -0.3, low, color="#c9c9ce", lw=0, zorder=0)
m = x >= x0
ax.plot(x, rks, color="#8c8c92", lw=2.2, ls="--", label="restricted solution")
ax.plot(x[m], bs[m], color=NEW, lw=3.2, label="broken-symmetry solution, where it exists")
ax.plot(x[~m], low[~m], color="#8c8c92", lw=3.2)
ax.plot([], [], color=DARK, lw=3.2, label="the surface: the lowest of the two")
ax.plot(x, model, color=NAVY, lw=2.2, ls=(0, (4, 2)), label="a smooth model")
i0 = np.argmin(np.abs(x - x0))
ax.scatter([x[i0]], [low[i0]], s=140, facecolor="white", edgecolor=DARK, lw=2, zorder=5)
ax.annotate("kink: the broken-symmetry\nsolution sets in", (x[i0], low[i0]), xytext=(0.03, 1.05), fontsize=10.5, color=DARK,
            ha="left", va="center", arrowprops=dict(arrowstyle="-", color=DARK, lw=1))
ax.set_xlim(0, 1); ax.set_ylim(-0.3, 1.55)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("reaction coordinate", fontsize=11, color=MID); ax.set_ylabel("energy", fontsize=11, color=MID)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#bbbbbb")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=9.5, labelcolor=["#6f6f75", NEW, DARK, NAVY])
ax.set_title("the surface has a corner, the model has none", fontsize=12.5, fontweight="bold", color=MID)
fig.savefig(OUT + "pic_kink.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_kink.png")
