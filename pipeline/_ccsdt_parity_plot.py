import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REACTIONS = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150']
ORCA  = {'rxn7949':3209.6,'rxn8832':2621.4,'rxn1320':3051.2,'rxn4113':5345.6,'rxn8885':3563.7,
         'rxn7945':3923.3,'rxn7937':3858.3,'rxn6196':4281.8,'rxn0346':3336.0,'rxn1150':3460.0}
UMA_S = {'rxn7949':2530.7,'rxn8832':2400.4,'rxn1320':2657.5,'rxn4113':5345.7,'rxn8885':1269.8,
         'rxn7945':3931.2,'rxn7937':3856.3,'rxn6196':4326.2,'rxn0346':3288.4,'rxn1150':3443.7}
FW2   = {'rxn7949':874.8 ,'rxn8832':3243.8,'rxn1320':3057.9,'rxn4113':None  ,'rxn8885':3528.7,
         'rxn7945':5099.8,'rxn7937':3929.5,'rxn6196':4390.0,'rxn0346':3375.4,'rxn1150':3513.9}

dev_uma = {r: UMA_S[r]-ORCA[r] for r in REACTIONS}
dev_fw2 = {r: (FW2[r]-ORCA[r]) if FW2[r] is not None else None for r in REACTIONS}

mae_uma = np.mean([abs(v) for v in dev_uma.values()])
vals_fw2 = [abs(v) for v in dev_fw2.values() if v is not None]
mae_fw2 = np.mean(vals_fw2)
n_fw2 = len(vals_fw2)

print(f'MAE UMA-S  (n={len(dev_uma)}): {mae_uma:.1f} meV')
print(f'MAE fw2    (n={n_fw2}, rxn4113 excluded - CCSD did not converge): {mae_fw2:.1f} meV')

fig, ax = plt.subplots(figsize=(7, 7))
all_vals = [ORCA[r] for r in REACTIONS] + [UMA_S[r] for r in REACTIONS] + [v for v in FW2.values() if v is not None]
lo, hi = min(all_vals) - 300, max(all_vals) + 300
ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y = x', zorder=1)

for r in REACTIONS:
    ax.scatter(ORCA[r], UMA_S[r], color='tab:blue', s=70, edgecolor='black', linewidth=0.5, zorder=3)
    ax.annotate(r.replace('rxn',''), (ORCA[r], UMA_S[r]), fontsize=7, xytext=(4,4), textcoords='offset points', color='tab:blue')
    if FW2[r] is not None:
        ax.scatter(ORCA[r], FW2[r], color='tab:red', s=70, marker='^', edgecolor='black', linewidth=0.5, zorder=3)
        ax.annotate(r.replace('rxn',''), (ORCA[r], FW2[r]), fontsize=7, xytext=(4,-10), textcoords='offset points', color='tab:red')

ax.scatter([], [], color='tab:blue', s=70, edgecolor='black', label=f'CCSD(T)@UMA-S geom (MAE={mae_uma:.0f} meV)')
ax.scatter([], [], color='tab:red', marker='^', s=70, edgecolor='black', label=f'CCSD(T)@MACE+δ fw2 geom (MAE={mae_fw2:.0f} meV, n={n_fw2})')

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel('CCSD(T) forward barrier @ ORCA wB97M-V NEB geometry (meV)')
ax.set_ylabel('CCSD(T) forward barrier @ alternate geometry (meV)')
ax.set_title('High-MR reactions: CCSD(T) barrier parity\n(geometry quality test — same level of theory, different TS geometry)')
ax.legend(loc='upper left', fontsize=9)
ax.set_aspect('equal')
fig.tight_layout()
out = r'c:\Transition 1X\Transition 1x\Transition1x\benchmark_plots\ccsdt_parity_high_mr.png'
fig.savefig(out, dpi=150)
print('Saved', out)
