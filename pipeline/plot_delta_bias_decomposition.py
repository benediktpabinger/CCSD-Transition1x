"""Bias decomposition figure of Chapter 3 (fig:bias-decomposition), rebuilt
from results/delta_fixed_head/eval_benchmark_sp_fixed_full_nod3.json, i.e. with
the cheap-level single points at wB97X/6-31G(d) without D3.

Per MR tier: barrier bias against the wB97M-V/def2-TZVP reference of the cheap
level (dashed), of MACE (teal arrow = baseline error, black tick) and of
MACE+Delta (teal arrow = baseline error, light orange = the true level gap the
head removes, dark orange = head error, black tick = resulting bias).
Output: figures/bias_decomposition_v5.png

Run: python pipeline/plot_delta_bias_decomposition.py   (from the repo root)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

JSON = "results/delta_fixed_head/eval_benchmark_sp_fixed_full_nod3.json"
OUT = "figures/bias_decomposition_v5.png"
TOP = ['rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885', 'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150']
BOT = ['rxn9246', 'rxn4498', 'rxn1061', 'rxn4003', 'rxn4004', 'rxn4063', 'rxn4114', 'rxn4060', 'rxn1961', 'rxn1962']
MID = ['rxn0896', 'rxn1154', 'rxn5690', 'rxn4513', 'rxn7955', 'rxn4519', 'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155']
TIERS = [("low MR", BOT), ("mid MR", MID), ("high MR", TOP)]
TEAL, ORANGE, ORANGE_LIGHT, INK, GREY = "#2f7f92", "#d95f2b", "#f2b59a", "#111111", "#666666"

rows = {r['rxn']: r for r in json.load(open(JSON))['reactions']}

def rel(e):
    e = np.array(e); return (e - e[0]) * 1000.0

def bias(tier, m):
    return float(np.mean([rel(rows[x][f'e_{m}_eV']).max() - rel(rows[x]['e_wb97m_eV']).max() for x in tier]))

def varrow(ax, x, y0, y1, color, lw=6):
    """Vertical arrow from y0 to y1 drawn as a thick line plus a triangle head."""
    d = y1 - y0
    if abs(d) < 6:
        ax.plot([x - 0.02, x + 0.02], [y1, y1], color=color, lw=3, solid_capstyle="butt")
        return
    s = 1 if d > 0 else -1
    ax.plot([x, x], [y0, y1 - s * 9], color=color, lw=lw, solid_capstyle="butt", zorder=3)
    ax.plot([x], [y1 - s * 4], marker="^" if s > 0 else "v", color=color, markersize=13, lw=0, zorder=4)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
for ax, (name, tier) in zip(axes, TIERS):
    cheap, mace, corr = bias(tier, 'wb97x'), bias(tier, 'mace'), bias(tier, 'delta')
    base = mace - cheap                 # baseline error, same for MACE and MACE+Delta
    head = corr - base                  # head error = total - baseline
    ax.axhline(0, color=INK, ls="--", lw=1.2)
    ax.axhline(cheap, color=TEAL, ls="--", lw=1.2, alpha=0.7)
    ax.text(0.98, cheap + 6, "ωB97X/6-31G(d)", color=TEAL, ha="right", va="bottom", fontsize=10, transform=ax.get_yaxis_transform())
    x1, x2 = 0.30, 0.66
    # MACE
    varrow(ax, x1, cheap, mace, TEAL)
    ax.plot([x1 - 0.08, x1 + 0.08], [mace, mace], color=INK, lw=3, zorder=5)
    ax.text(x1 - 0.10, mace, f"MACE\n{mace:+.0f}", ha="right", va="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(x1 + 0.05, (cheap + mace) / 2 if abs(base) > 30 else mace - 22, f"base {base:+.0f}", ha="left", va="center", fontsize=9, color=TEAL)
    # MACE+Delta: baseline arrow, light bar = gap removal, dark arrow = head error, tick = result
    varrow(ax, x2, cheap, mace, TEAL)
    xb = x2 + 0.08; w = 0.035
    ax.add_patch(Rectangle((xb - w / 2, min(mace, base)), w, abs(mace - base), color=ORANGE_LIGHT, lw=0, zorder=2))
    ax.plot([x2, xb], [mace, mace], color="#bbbbbb", lw=1, zorder=1)
    varrow(ax, xb, base, corr, ORANGE)
    ax.plot([xb - 0.08, xb + 0.08], [corr, corr], color=INK, lw=3, zorder=5)
    ax.text(xb + 0.10, corr + 4, f"MACE+Δ\n{corr:+.0f}", ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
    ax.text(xb + 0.10, corr - 6, f"head {head:+.0f}", ha="left", va="top", fontsize=9, color=ORANGE)
    ax.set_xlim(0, 1.05); ax.set_ylim(-260, 290)
    ax.set_title(name, loc="left", fontsize=15, fontweight="bold")
    ax.set_xticks([]); [ax.spines[s].set_visible(False) for s in ("top", "right", "bottom")]
axes[0].set_ylabel("barrier bias vs.\nωB97M-V reference (meV)", fontsize=12)
axes[0].text(0.02, -14, "reference: ωB97M-V/def2-TZVP (target)", color=GREY, fontsize=10, va="top", transform=axes[0].get_yaxis_transform())
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=TEAL, lw=6, label="baseline error  $E_\\mathrm{MACE}-E^{\\omega\\mathrm{B97X}}$: MACE's error on its own level"),
           Line2D([0], [0], color=ORANGE_LIGHT, lw=8, label="the level gap the head removes  $E^{\\omega\\mathrm{B97M}}-E^{\\omega\\mathrm{B97X}}$"),
           Line2D([0], [0], color=ORANGE, lw=6, label="head error  $\\Delta-(E^{\\omega\\mathrm{B97M}}-E^{\\omega\\mathrm{B97X}})$"),
           Line2D([0], [0], color=INK, lw=3, label="resulting bias")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("written", OUT)
for name, tier in TIERS:
    cheap, mace, corr = bias(tier, 'wb97x'), bias(tier, 'mace'), bias(tier, 'delta')
    print(f"{name}: cheap {cheap:+.0f} MACE {mace:+.0f} MACE+D {corr:+.0f} | baseline {mace-cheap:+.0f} head {corr-(mace-cheap):+.0f}")
