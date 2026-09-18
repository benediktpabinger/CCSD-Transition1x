"""Force-error dumbbell figure of Chapter 3 (fig:force-errors), rebuilt from
results/delta_fixed_head/eval_benchmark_sp_fixed_full_nod3.json (cheap-level
single points at wB97X/6-31G(d) without D3).

Per reaction: force MAE against the wB97M-V/def2-TZVP reference of MACE (teal)
and MACE+Delta (orange), connected; the light tick marks the wB97X/6-31G(d)
single points, i.e. the level gap itself. By MR tier.
Output: figures/force_errors_dumbbell_v2.png

Run: python pipeline/plot_delta_force_dumbbell.py   (from the repo root)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = "results/delta_fixed_head/eval_benchmark_sp_fixed_full_nod3.json"
OUT = "figures/force_errors_dumbbell_v2.png"
TOP = ['rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885', 'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150']
BOT = ['rxn9246', 'rxn4498', 'rxn1061', 'rxn4003', 'rxn4004', 'rxn4063', 'rxn4114', 'rxn4060', 'rxn1961', 'rxn1962']
MID = ['rxn0896', 'rxn1154', 'rxn5690', 'rxn4513', 'rxn7955', 'rxn4519', 'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155']
TIERS = [("low MR", BOT), ("mid MR", MID), ("high MR", TOP)]
TEAL, ORANGE, TEAL_L, INK, GREY = "#2f7f92", "#d95f2b", "#a9cdd6", "#111111", "#666666"

rows = {r['rxn']: r for r in json.load(open(JSON))['reactions']}
def fmae(r, m):
    return np.abs(np.array(r[f'f_{m}_eV_per_ang']) - np.array(r['f_wb97m_eV_per_ang'])).mean() * 1000.0

fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), sharey=True, gridspec_kw={"wspace": 0.05})
for j, (name, tier) in enumerate(TIERS):
    ax = axes[j]; x = np.arange(len(tier))
    fm = np.array([fmae(rows[r], 'mace') for r in tier]); fd = np.array([fmae(rows[r], 'delta') for r in tier]); fx = np.array([fmae(rows[r], 'wb97x') for r in tier])
    for i in range(len(tier)):
        ax.plot([x[i], x[i]], [fd[i], fm[i]], color="#c8c8c8", lw=2.5, zorder=1)
        ax.plot([x[i] - 0.25, x[i] + 0.25], [fx[i], fx[i]], color=TEAL_L, lw=4, zorder=2, solid_capstyle="round")
    ax.scatter(x, fm, color=TEAL, s=55, zorder=3); ax.scatter(x, fd, color=ORANGE, s=55, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(tier, rotation=90, fontsize=9)
    ax.set_ylim(0, 340)
    ax.text(0.03, 0.96, name, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    ax.text(0.03, 0.88, f"MAE {np.mean(fm):.0f} → {np.mean(fd):.0f} meV/Å", transform=ax.transAxes, fontsize=12, color=GREY, va="top")
    ax.set_facecolor("#f3f3f5" if j != 1 else "#ebebf0")
    for xi in x: ax.axvline(xi, color="#dddddd", lw=0.8, zorder=0)
    [ax.spines[s].set_visible(False) for s in ("top", "right")]
axes[0].set_ylabel("force MAE vs. ωB97M-V reference  (meV/Å)", fontsize=12)
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([0], [0], marker="o", color=TEAL, lw=0, markersize=8, label="MACE (target: ωB97X/6-31G(d))"),
                        Line2D([0], [0], marker="o", color=ORANGE, lw=0, markersize=8, label="MACE+Δ (target: ωB97M-V/def2-TZVP)"),
                        Line2D([0], [0], color=TEAL_L, lw=4, label="ωB97X single points (level gap)")],
               loc="upper right", fontsize=10, frameon=False)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("written", OUT)
