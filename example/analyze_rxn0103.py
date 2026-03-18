"""Analysis of CCSD vs wB97x results for rxn0103"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# Load all results
files = sorted(glob.glob('data/ccsd_rxn0103/*.json'))
all_results = []
for f in files:
    all_results.extend(json.load(open(f)))

total = len(all_results)
ok = sum(r['success'] for r in all_results)
failed = total - ok

print(f"Configs gesamt: {total}")
print(f"Erfolgreich:    {ok}")
print(f"Fehlgeschlagen: {failed}")

# Extract energies
configs = [r for r in all_results if r['success']]
configs.sort(key=lambda r: r['config_index'])

idx = np.array([r['config_index'] for r in configs])
ccsd = np.array([r['ccsd.energy'] for r in configs])
wb97 = np.array([r['wB97x.energy'] for r in configs])
diff = ccsd - wb97

print(f"\nCCSD Energie:  min={ccsd.min():.3f}  max={ccsd.max():.3f}  mean={ccsd.mean():.3f} eV")
print(f"wB97x Energie: min={wb97.min():.3f}  max={wb97.max():.3f}  mean={wb97.mean():.3f} eV")
print(f"\nDiff (CCSD-wB97x): min={diff.min():.3f}  max={diff.max():.3f}  mean={diff.mean():.3f}  std={diff.std():.4f} eV")

# Relative energies (subtract minimum)
ccsd_rel = ccsd - ccsd.min()
wb97_rel = wb97 - wb97.min()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Energy profile along reaction path
ax1 = axes[0]
ax1.plot(idx, ccsd_rel, 'b-o', markersize=3, label='CCSD/cc-pVDZ', linewidth=1)
ax1.plot(idx, wb97_rel, 'r-o', markersize=3, label='wB97x/6-31G(d)', linewidth=1)
ax1.set_xlabel('Config Index')
ax1.set_ylabel('Relative Energie (eV)')
ax1.set_title('rxn0103 (C3H5NO2) — Energieprofil entlang Reaktionspfad')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Systematic offset
ax2 = axes[1]
ax2.plot(idx, diff, 'g-o', markersize=3, linewidth=1)
ax2.axhline(diff.mean(), color='k', linestyle='--', label=f'Mittelwert: {diff.mean():.3f} eV')
ax2.set_xlabel('Config Index')
ax2.set_ylabel('CCSD - wB97x (eV)')
ax2.set_title('Systematischer Offset zwischen CCSD und wB97x')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/ccsd_rxn0103/ccsd_vs_wb97x_rxn0103.png', dpi=150)
print("\nPlot gespeichert: data/ccsd_rxn0103/ccsd_vs_wb97x_rxn0103.png")
