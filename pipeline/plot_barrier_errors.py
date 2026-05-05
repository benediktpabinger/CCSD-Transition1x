import json
import numpy as np
import matplotlib.pyplot as plt

with open('../barrier_comparison_p10.json') as f:
    data = json.load(f)

reactions = data['reactions']
t1x  = np.array([r['t1x_meV']  for r in reactions])
orca = np.array([r['orca_meV'] for r in reactions])
mace = np.array([r['mace_meV'] for r in reactions])

errors = {
    'MACE vs T1x\n(wB97X-D3/6-31G(d))':       mace - t1x,
    'MACE vs ORCA NEB\n(wB97M-V/def2-TZVP)':  mace - orca,
    'T1x vs ORCA NEB\n(wB97M-V/def2-TZVP)':   t1x  - orca,
}
colors = ['darkorange', 'seagreen', 'steelblue']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Barrier Height Errors (meV)  —  n=279 converged NEB reactions', fontsize=13, fontweight='bold')

for ax, (label, err), color in zip(axes, errors.items(), colors):
    rmse = np.sqrt(np.mean(err**2))
    mae  = np.mean(np.abs(err))
    bias = np.mean(err)
    ax.hist(err, bins=40, color=color, alpha=0.8, edgecolor='white', linewidth=0.4)
    ax.axvline(0,    color='black', lw=1.2, ls='--', label='zero')
    ax.axvline(bias, color='red',   lw=1.5, ls='-',  label=f'bias={bias:+.0f} meV')
    ax.set_xlabel('Error (meV)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95, f'RMSE={rmse:.0f} meV\nMAE={mae:.0f} meV',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

max_y = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(0, max_y)

plt.tight_layout()
plt.savefig('../barrier_errors_p10.png', dpi=150, bbox_inches='tight')
print("Saved barrier_errors.png")
