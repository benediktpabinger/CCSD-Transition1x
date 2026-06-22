import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read

HOME = '/home/energy/s242862'
OUT_DIR = f'{HOME}/benchmark_plots'
os.makedirs(OUT_DIR, exist_ok=True)

HIGH10 = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150']
METHODS = {
    'ORCA wB97M-V': 'orca_neb_results',
    'UMA-S':        'uma_neb_results',
    'UMA-M':        'uma_m_neb_results',
    'eSEN':         'esen_neb_results',
    'MACE+delta fw2': 'mace_delta_neb_results_fw2',
}
METHOD_COLOR = {'ORCA wB97M-V': 'black', 'UMA-S': 'tab:blue', 'UMA-M': 'tab:purple',
                'eSEN': 'tab:brown', 'MACE+delta fw2': 'tab:red'}

def kabsch_rmsd(P, Q):
    P = P - P.mean(axis=0); Q = Q - Q.mean(axis=0)
    H = P.T @ Q; U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((P @ R.T) - Q)**2, axis=1))))

def load_xyz(path):
    return read(path).get_positions() if os.path.exists(path) else None

print('=== CASSCF+NEVPT2 OptTS barriers (all 10 High-MR reactions) ===')
header = '{:<10} {:>13} {:>10} {:>10} {:>10} {:>10}  notes'.format(
    'rxn', 'active space', 'threshold', 'solver', 'fwd (meV)', 'rev (meV)')
print(header)
print('-' * 100)

mr_opt = {}
mr_ncas = {}
for rxn in HIGH10:
    jpath = f'{HOME}/nevpt2_optts_results/{rxn}_avas/nevpt2_optts_results.json'
    if not os.path.exists(jpath):
        print(f'{rxn:<10}  MISSING'); continue
    with open(jpath) as f:
        d = json.load(f)
    ncas    = d['ncas']; nelecas = d['nelecas']
    fwd     = d['barrier_forward_meV']; rev = d['barrier_reverse_meV']
    thresh  = d.get('avas_threshold', '?')
    solver  = d.get('solver_used_ts', '?')
    note = ''
    if ncas <= 2: note = '*** CAS(2,2) - too small ***'
    elif ncas <= 4: note = '(4e,4o) - small but may be valid'
    row_str = '{:<10} ({:d}e,{:d}o){:<3} thr={!s:<5} {:>10} {:>10.1f} {:>10.1f}  {}'.format(
        rxn, nelecas, ncas, '', thresh, solver, fwd, rev, note)
    print(row_str)
    mr_opt[rxn] = {'fwd': fwd, 'rev': rev, 'xyz': f'{HOME}/nevpt2_optts_results/{rxn}_avas/ts_casscf_opt.xyz',
                   'ncas': ncas}
    mr_ncas[rxn] = ncas

print()
print('=== RMSD vs MR-Optimized TS geometry ===')
col_headers = '{:<10}'.format('rxn') + ''.join('{:>16}'.format(m) for m in METHODS) + '  notes'
print(col_headers)
print('-' * (10 + 16*len(METHODS) + 8))

rmsd_rows = {}
# Valid = large active spaces (exclude CAS(2,2) and CAS(4,4) from old pruned run)
VALID = {'rxn7949','rxn8832','rxn4113','rxn8885','rxn7945','rxn7937','rxn0346'}

for rxn in HIGH10:
    if rxn not in mr_opt:
        print('{:<10}  MISSING'.format(rxn)); continue
    mropt = load_xyz(mr_opt[rxn]['xyz'])
    row = {}
    line = '{:<10}'.format(rxn)
    for method, dirname in METHODS.items():
        geo = load_xyz(f'{HOME}/{dirname}/{rxn}/transition_state.xyz')
        v = kabsch_rmsd(mropt, geo) if (mropt is not None and geo is not None) else None
        row[method] = v
        line += '{:>16.3f}'.format(v) if v is not None else '{:>16}'.format('N/A')
    rmsd_rows[rxn] = row
    ncas = mr_ncas.get(rxn, 0)
    flag = 'valid' if rxn in VALID else ('CAS(2,2)-skip' if ncas <= 2 else 'small(4,4)')
    print(line + '  ' + flag)

# Means over valid set only
valid_rxns = [r for r in HIGH10 if r in rmsd_rows and r in VALID]
n = len(valid_rxns)
means_line = '{:<22}'.format(f'Mean(valid n={n})')
means = {}
for m in METHODS:
    vals = [rmsd_rows[r][m] for r in valid_rxns if rmsd_rows[r].get(m) is not None]
    mu = np.mean(vals) if vals else float('nan')
    means[m] = mu
    means_line += '{:>16.3f}'.format(mu)
print()
print(means_line)

# Bar chart - valid reactions only
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(valid_rxns)); width = 0.8 / len(METHODS)

for i, (m, color) in enumerate(METHOD_COLOR.items()):
    vals = [rmsd_rows[r][m] if (r in rmsd_rows and rmsd_rows[r].get(m) is not None) else 0
            for r in valid_rxns]
    mu = means[m]
    ax.bar(x + i*width - 0.4 + width/2, vals, width,
           label=f'{m} (mean={mu:.3f} Å)', color=color)

ax.set_xticks(x); ax.set_xticklabels(valid_rxns, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('RMSD vs CASSCF+NEVPT2 OptTS (Å)')
ax.set_title(
    f'TS geometry RMSD vs MR-Optimized reference — {n} High-MR reactions\n'
    '(CASSCF(AVAS)/NEVPT2 OptTS gold standard, full active spaces, no active-space pruning)')
ax.legend(fontsize=8.5, loc='upper right')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mr_optimized_rmsd_all10.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved to {OUT_DIR}/mr_optimized_rmsd_all10.png')
