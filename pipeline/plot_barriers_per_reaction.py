import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open('../barrier_comparison_p10.json') as f:
    data = json.load(f)

reactions = data['reactions']
rxn_labels = [r['rxn'] for r in reactions]
t1x  = np.array([r['t1x_meV']  for r in reactions])
orca = np.array([r['orca_meV'] for r in reactions])
mace = np.array([r['mace_meV'] for r in reactions])

# Sort by T1x barrier for readability
order = np.argsort(t1x)
t1x  = t1x[order]
orca = orca[order]
mace = mace[order]
rxn_labels = [rxn_labels[i] for i in order]

x = np.arange(len(reactions))

fig, ax = plt.subplots(figsize=(20, 6))

ax.plot(x, t1x,  color='steelblue',  lw=1.2, label='T1x  wB97X-D3/6-31G(d)',     zorder=3)
ax.plot(x, mace, color='darkorange', lw=1.2, label='MACE p10 (epoch 291)',         zorder=4)
ax.plot(x, orca, color='seagreen',   lw=1.2, label='ORCA NEB  wB97M-V/def2-TZVP', zorder=2)

ax.fill_between(x, t1x, mace, alpha=0.12, color='darkorange')
ax.fill_between(x, t1x, orca, alpha=0.10, color='seagreen')

ax.set_xlabel('Reaction (sorted by T1x barrier height)', fontsize=11)
ax.set_ylabel('Barrier height (meV)', fontsize=11)
ax.set_title('Barrier Heights per Reaction — T1x vs MACE vs ORCA NEB  (n=279)', fontsize=13, fontweight='bold')

ax.set_xlim(0, len(reactions) - 1)
ax.legend(fontsize=10)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(axis='y', lw=0.4, alpha=0.5)

plt.tight_layout()
plt.savefig('../barrier_per_reaction_p10.png', dpi=150, bbox_inches='tight')
print("Saved barrier_per_reaction.png")
