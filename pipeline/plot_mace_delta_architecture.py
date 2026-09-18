"""Architecture figure of Chapter 3 (fig:mace-delta-arch), v4: rebuilt after the
level correction (wB97X/6-31G(d), no D3) and with both readouts shown as
reading the invariant channels, as the Background states.
Output: figures/mace_delta_architecture_v4.png

Run: python pipeline/plot_mace_delta_architecture.py   (from the repo root)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figures/mace_delta_architecture_v4.png"
INK, DIM, GREY, BLUE, BLUE_EDGE = "#222222", "#666666", "#e8e8ec", "#dbe8fb", "#2b5fd9"

fig, ax = plt.subplots(figsize=(11, 10.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 10.2); ax.axis("off")

def box(x, y, w, h, title, sub=None, fc="white", ec=INK, lw=1.4, tsize=15, ssize=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=lw))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=ssize, color=INK)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.6, color=INK, shrinkA=0, shrinkB=0))

def note(x, y, text, ha="left", size=11.5):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=DIM, style="italic", linespacing=1.3)

# legend
ax.add_patch(FancyBboxPatch((7.2, 9.55), 0.45, 0.28, boxstyle="round,pad=0.01,rounding_size=0.06", fc=GREY, ec=INK, lw=1.2))
ax.text(7.8, 9.69, "trained on Transition1x,\nfrozen while the head is trained", ha="left", va="center", fontsize=11, color=INK, linespacing=1.3)
ax.add_patch(FancyBboxPatch((7.2, 9.02), 0.45, 0.28, boxstyle="round,pad=0.01,rounding_size=0.06", fc=BLUE, ec=BLUE_EDGE, lw=1.6))
ax.text(7.8, 9.16, "trained on the energy difference", ha="left", va="center", fontsize=11, color=INK)

# geometry
box(3.9, 9.35, 3.0, 0.55, "geometry  R", tsize=15)
note(5.05, 9.05, "positions and elements\nof N atoms", ha="right", size=11)
arrow(5.4, 9.35, 5.4, 8.75)
# encoder
box(3.4, 8.0, 4.0, 0.75, "MACE encoder", "72.4M parameters", fc=GREY)
arrow(5.4, 8.0, 5.4, 7.45)
# features
ax.add_patch(FancyBboxPatch((1.7, 6.5), 7.4, 1.05, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=INK, lw=1.4))
ax.text(5.4, 7.32, "per-atom features", ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
ax.text(5.4, 6.85, "17,408 numbers per atom describing its chemical environment;\n2,048 of them do not change when the molecule is rotated",
        ha="center", va="center", fontsize=11.5, color=INK, linespacing=1.3)
# branches
arrow(3.3, 6.5, 2.8, 5.85); arrow(7.7, 6.5, 8.2, 5.85)
note(2.6, 6.15, "the 2,048 invariant\nchannels only", ha="right", size=11)
note(8.4, 6.15, "the 2,048 invariant\nchannels only", ha="left", size=11)
box(1.0, 5.05, 3.6, 0.8, "energy readout", "17,442 parameters", fc=GREY)
box(6.4, 5.05, 3.6, 0.8, "correction head", "131,136 parameters", fc=BLUE, ec=BLUE_EDGE, lw=2.0)
arrow(2.8, 5.05, 2.8, 4.25); arrow(8.2, 5.05, 8.2, 4.25)
note(3.0, 4.7, "one number per atom,\nsummed over the N atoms", ha="left", size=11)
note(8.4, 4.7, "one number per atom,\nsummed over the N atoms", ha="left", size=11)
box(1.9, 3.7, 1.8, 0.55, r"$E_\mathrm{MACE}$", tsize=15)
box(7.5, 3.7, 1.4, 0.55, r"$\Delta$", tsize=15)
note(2.8, 3.3, "predicts the energy at\nωB97X/6-31G(d)", ha="center", size=11)
note(8.2, 3.3, "predicts the correction from\nωB97X/6-31G(d) to ωB97M-V/def2-TZVP", ha="center", size=11)
# merge
arrow(2.8, 2.85, 4.4, 2.15); arrow(8.2, 2.85, 6.6, 2.15)
box(2.0, 1.15, 7.0, 1.0, "", fc="white")
ax.text(5.5, 1.85, r"$E^{\,\omega\mathrm{B97M\text{-}V/def2\text{-}TZVP}}_{\mathrm{MACE}+\Delta} \;=\; E^{\,\omega\mathrm{B97X/6\text{-}31G(d)}}_{\mathrm{MACE}} \;+\; \Delta$",
        ha="center", va="center", fontsize=15, color=INK)
ax.text(5.5, 1.40, r"$F_{\mathrm{MACE}+\Delta} \;=\; F_{\mathrm{MACE}} - \nabla_R\,\Delta$", ha="center", va="center", fontsize=14, color=INK)
note(5.5, 0.85, "one forward pass computes both terms", ha="center", size=11.5)

fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("written", OUT)
