"""Analysis: wB97x vs CCSD/DZ vs CCSD/TZ for rxn0103"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- Load DZ results ---
dz_files = sorted(glob.glob('data/ccsd_rxn0103/results_*.json'))
dz_raw = []
for f in dz_files:
    dz_raw.extend(json.load(open(f)))
dz = {r['config_index']: r for r in dz_raw if r['success']}

# --- Load TZ results ---
tz_raw = json.load(open('data/ccsd_tz_rxn0103.json'))
tz = {r['config_index']: r for r in tz_raw if r['success']}

# --- Common indices ---
common = sorted(set(dz.keys()) & set(tz.keys()))
print(f"DZ configs:     {len(dz)}")
print(f"TZ configs:     {len(tz)}")
print(f"Common:         {len(common)}")

idx    = np.array(common)
ccsd_dz = np.array([dz[i]['ccsd.energy'] for i in common])
ccsd_tz = np.array([tz[i]['ccsd.energy'] for i in common])
wb97    = np.array([dz[i]['wB97x.energy'] for i in common])

# Relative energies
dz_rel  = ccsd_dz - ccsd_dz.min()
tz_rel  = ccsd_tz - ccsd_tz.min()
wb97_rel = wb97 - wb97.min()

dz_tz_diff  = ccsd_tz - ccsd_dz
dz_wb97_diff = ccsd_dz - wb97
tz_wb97_diff = ccsd_tz - wb97

print(f"\nCCSD/TZ - CCSD/DZ:  mean={dz_tz_diff.mean():.4f}  std={dz_tz_diff.std():.4f}  max={np.abs(dz_tz_diff).max():.4f} eV")
print(f"CCSD/DZ - wB97x:    mean={dz_wb97_diff.mean():.4f}  std={dz_wb97_diff.std():.4f}  max={np.abs(dz_wb97_diff).max():.4f} eV")
print(f"CCSD/TZ - wB97x:    mean={tz_wb97_diff.mean():.4f}  std={tz_wb97_diff.std():.4f}  max={np.abs(tz_wb97_diff).max():.4f} eV")

# --- Plots ---
fig, axes = plt.subplots(3, 1, figsize=(13, 11))

# Plot 1: Energy profiles
ax = axes[0]
ax.plot(idx, wb97_rel,  'r-o', markersize=3, linewidth=1, label='wB97x/6-31G(d)')
ax.plot(idx, dz_rel,    'b-o', markersize=3, linewidth=1, label='CCSD/cc-pVDZ')
ax.plot(idx, tz_rel,    'g-o', markersize=3, linewidth=1, label='CCSD/cc-pVTZ')
ax.set_ylabel('Relative Energy (eV)')
ax.set_title('rxn0103 (C3H5NO2) — Energy profile along reaction path')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: TZ vs DZ difference (basis set effect)
ax = axes[1]
ax.plot(idx, dz_tz_diff, 'purple', marker='o', markersize=3, linewidth=1)
ax.axhline(dz_tz_diff.mean(), color='k', linestyle='--',
           label=f'mean={dz_tz_diff.mean():.4f} eV, std={dz_tz_diff.std():.4f} eV')
ax.set_ylabel('CCSD/TZ - CCSD/DZ (eV)')
ax.set_title('Basis set effect (TZ vs DZ)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: CCSD/TZ vs wB97x difference (method effect)
ax = axes[2]
ax.plot(idx, tz_wb97_diff, 'darkorange', marker='o', markersize=3, linewidth=1)
ax.axhline(tz_wb97_diff.mean(), color='k', linestyle='--',
           label=f'mean={tz_wb97_diff.mean():.4f} eV, std={tz_wb97_diff.std():.4f} eV')
ax.set_xlabel('Config Index')
ax.set_ylabel('CCSD/TZ - wB97x (eV)')
ax.set_title('Method effect: CCSD/TZ vs wB97x/6-31G(d)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'data/ccsd_tz_rxn0103_analysis.png'
plt.savefig(out, dpi=150)
print(f"\nPlot saved: {out}")
