"""Barrier-error dumbbell figure of Chapter 3 (fig:barrier-errors), rebuilt from
results/delta_fixed_head/eval_benchmark_sp_fixed_full_nod3.json (cheap-level
single points at wB97X/6-31G(d) without D3).

Top: per-reaction barrier error of MACE and MACE+Delta against the
wB97M-V/def2-TZVP reference, by MR tier; off-scale values as triangles.
Bottom: absolute barriers as grouped bars (wB97X DFT, MACE, wB97M-V reference,
MACE+Delta).
Output: figures/barrier_errors_dumbbell_v6.png

Run: python pipeline/plot_delta_barrier_dumbbell.py   (from the repo root)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = "results/delta_fixed_head/eval_benchmark_sp_fixed_full_nod3.json"
OUT = "figures/barrier_errors_dumbbell_v6.png"
TOP = ['rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885', 'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150']
BOT = ['rxn9246', 'rxn4498', 'rxn1061', 'rxn4003', 'rxn4004', 'rxn4063', 'rxn4114', 'rxn4060', 'rxn1961', 'rxn1962']
MID = ['rxn0896', 'rxn1154', 'rxn5690', 'rxn4513', 'rxn7955', 'rxn4519', 'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155']
TIERS = [("low MR", BOT), ("mid MR", MID), ("high MR", TOP)]
TEAL, ORANGE, TEAL_L, ORANGE_L, INK, GREY = "#2f7f92", "#d95f2b", "#a9cdd6", "#f2b59a", "#111111", "#666666"
YLIM = 400

rows = {r['rxn']: r for r in json.load(open(JSON))['reactions']}
def rel(e):
    e = np.array(e); return (e - e[0]) * 1000.0
def barrier(r, m): return rel(r[f'e_{m}_eV']).max()
def err(r, m): return barrier(r, m) - barrier(r, 'wb97m')

fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex="col", gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.06, "wspace": 0.05})
for j, (name, tier) in enumerate(TIERS):
    top, bot = axes[0, j], axes[1, j]
    x = np.arange(len(tier))
    em = np.array([err(rows[r], 'mace') for r in tier]); ed = np.array([err(rows[r], 'delta') for r in tier])
    for i in range(len(tier)):
        a, b = np.clip(em[i], -YLIM, YLIM), np.clip(ed[i], -YLIM, YLIM)
        top.plot([x[i], x[i]], [a, b], color="#c8c8c8", lw=2.5, zorder=1)
    for vals, col in ((em, TEAL), (ed, ORANGE)):
        inside = np.abs(vals) <= YLIM
        top.scatter(x[inside], vals[inside], color=col, s=55, zorder=3)
        for i in np.where(~inside)[0]:
            y = YLIM if vals[i] > 0 else -YLIM
            top.scatter([x[i]], [y * 0.97], marker="^" if vals[i] > 0 else "v", color=col, s=90, zorder=4)
            top.text(x[i] + 0.3, y * (0.97 if col == TEAL else 0.84), f"{vals[i]:+.0f}", color=col, fontsize=9, ha="left", va="center")
    top.axhline(0, color=INK, ls="--", lw=1.2)
    top.set_ylim(-YLIM - 60, YLIM + 10)
    top.text(0.03, 0.96, name, transform=top.transAxes, fontsize=16, fontweight="bold", va="top")
    top.text(0.03, 0.88, f"MAE {np.mean(np.abs(em)):.0f} → {np.mean(np.abs(ed)):.0f} meV", transform=top.transAxes, fontsize=12, color=GREY, va="top")
    # bottom: absolute barriers
    bw = 0.18
    series = [('wb97x', TEAL_L, "ωB97X/6-31G(d) (DFT)"), ('mace', TEAL, "MACE (target: ωB97X/6-31G(d))"),
              ('wb97m', ORANGE_L, "ωB97M-V/def2-TZVP (DFT, reference)"), ('delta', ORANGE, "MACE+Δ (target: ωB97M-V/def2-TZVP)")]
    for k, (m, col, lab) in enumerate(series):
        vals = [barrier(rows[r], m) / 1000.0 for r in tier]
        bot.bar(x + (k - 1.5) * bw, vals, width=bw, color=col, label=lab if j == 0 else None, zorder=2)
    bot.set_xticks(x); bot.set_xticklabels(tier, rotation=90, fontsize=9)
    bot.set_ylim(0, 6.8)
    for ax in (top, bot):
        ax.set_facecolor("#f3f3f5" if j != 1 else "#ebebf0")
        for xi in x: ax.axvline(xi, color="#dddddd", lw=0.8, zorder=0)
        [ax.spines[s].set_visible(False) for s in ("top", "right")]
    if j > 0:
        top.tick_params(labelleft=False); bot.tick_params(labelleft=False)
axes[0, 0].set_ylabel("barrier error at fixed geometries\n$E_a^\\mathrm{model}-E_a^\\mathrm{ref}$ (meV)", fontsize=12)
axes[1, 0].set_ylabel("absolute barrier (eV)", fontsize=12)
axes[0, 0].text(0.03, 0.02 + 0.07, "reference", transform=axes[0, 0].transAxes, fontsize=10, color=GREY) if False else None
from matplotlib.lines import Line2D
axes[0, 0].legend(handles=[Line2D([0], [0], marker="o", color=TEAL, lw=0, markersize=8, label="MACE (target: ωB97X/6-31G(d))"),
                            Line2D([0], [0], marker="o", color=ORANGE, lw=0, markersize=8, label="MACE+Δ (target: ωB97M-V/def2-TZVP)")],
                   loc="lower left", fontsize=10, frameon=False)
axes[1, 0].legend(loc="upper left", fontsize=9, frameon=False, ncol=1)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("written", OUT)
