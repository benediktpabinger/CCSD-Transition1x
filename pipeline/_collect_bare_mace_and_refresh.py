import sqlite3, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read

HOME = '/home/energy/s242862'
BM_PATH = f'{HOME}/delta_head/full_benchmark_results.json'
OUT_DIR = f'{HOME}/benchmark_plots'
os.makedirs(OUT_DIR, exist_ok=True)

REACTIONS = [
    'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
    'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150',
    'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
    'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
    'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
    'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
]
TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
BOTTOM10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}
MR = {r: ('High' if r in TOP10 else ('Low' if r in BOTTOM10 else 'Mid')) for r in REACTIONS}

BARE_DIR = f'{HOME}/mace_bare_neb_results'

def get_neb_barriers(rxn, base):
    db_path = f'{base}/{rxn}/neb.db'
    fmaxs_path = f'{base}/{rxn}/fmaxs.json'
    con = sqlite3.connect(db_path)
    all_e = [r[0] for r in con.execute('SELECT energy FROM systems ORDER BY id').fetchall()]
    con.close()
    e = np.array(all_e[-10:])
    rel = (e - e[0]) * 1000
    fwd = float(rel.max())
    rev = float(fwd - rel[-1])
    fmax = float(json.load(open(fmaxs_path))[-1])
    return fwd, rev, fmax

def kabsch_rmsd(P, Q):
    P = P - P.mean(axis=0); Q = Q - Q.mean(axis=0)
    H = P.T @ Q; U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((P @ R.T) - Q)**2, axis=1))))

def load_xyz(path):
    return read(path).get_positions() if os.path.exists(path) else None

# ---------- 1. compute bare-MACE NEB barriers, write into full_benchmark_results.json ----------
with open(BM_PATH) as f:
    bm = json.load(f)
bm_by_rxn = {r['rxn']: r for r in bm['reactions']}

bare_results = {}
errors = []
for rxn in REACTIONS:
    try:
        fwd, rev, fmax = get_neb_barriers(rxn, BARE_DIR)
        bare_results[rxn] = {'fwd': fwd, 'rev': rev, 'fmax': fmax}
    except Exception as e:
        errors.append(f'{rxn}: {e}')

if errors:
    print('ERRORS loading bare MACE barriers:', errors)

for r in bm['reactions']:
    rxn = r['rxn']
    if rxn in bare_results:
        r['mace_bare_neb_fwd_meV'] = round(bare_results[rxn]['fwd'], 1)
        r['mace_bare_neb_rev_meV'] = round(bare_results[rxn]['rev'], 1)
        r['mace_bare_neb_fmax']    = round(bare_results[rxn]['fmax'], 4)

with open(BM_PATH, 'w') as f:
    json.dump(bm, f, indent=2)
print(f'Updated {BM_PATH} with bare MACE NEB results for {len(bare_results)} reactions\n')

# ---------- 2. ablation: bare MACE vs MACE+delta fw2, RMSD + barrier MAE vs ORCA, by MR category ----------
METHODS = {
    'MACE (bare)':     ('mace_bare_neb_results',       'mace_bare_neb_fwd_meV'),
    'MACE+delta fw2':  ('mace_delta_neb_results_fw2',  'delta_fw2_neb_fwd_meV'),
}
METHOD_COLOR = {'MACE (bare)': 'tab:gray', 'MACE+delta fw2': 'tab:red'}

rmsd_rows = []
for rxn in REACTIONS:
    orca = load_xyz(f'{HOME}/orca_neb_results/{rxn}/transition_state.xyz')
    row = {'rxn': rxn, 'mr': MR[rxn]}
    for method, (dirname, _) in METHODS.items():
        geo = load_xyz(f'{HOME}/{dirname}/{rxn}/transition_state.xyz')
        row[method] = kabsch_rmsd(orca, geo) if (orca is not None and geo is not None) else None
    rmsd_rows.append(row)

print('=== Ablation: bare MACE vs MACE+delta fw2 ===\n')
print('-- RMSD vs ORCA wB97M-V NEB (Å) --')
print(f'{"rxn":<10} {"MR":<5}' + ''.join(f'{m:>16}' for m in METHODS))
for row in rmsd_rows:
    line = f'{row["rxn"]:<10} {row["mr"]:<5}'
    for m in METHODS:
        v = row[m]
        line += f'{v:>16.3f}' if v is not None else f'{"N/A":>16}'
    print(line)

print()
print(f'{"MR":<6}' + ''.join(f'{m:>16}' for m in METHODS) + f'{"n":>5}')
for cat in ['High', 'Mid', 'Low']:
    line = f'{cat:<6}'
    n = 0
    for m in METHODS:
        vals = [row[m] for row in rmsd_rows if row['mr'] == cat and row[m] is not None]
        n = len(vals)
        line += f'{np.mean(vals):>16.3f}' if vals else f'{"N/A":>16}'
    line += f'{n:>5}'
    print(line)
line = f'{"All":<6}'
for m in METHODS:
    vals = [row[m] for row in rmsd_rows if row[m] is not None]
    line += f'{np.mean(vals):>16.3f}'
line += f'{len(rmsd_rows):>5}'
print(line)

print()
print('-- Forward barrier MAE vs wB97M-V NEB (meV) --')
print(f'{"MR":<6}' + ''.join(f'{m:>16}' for m in METHODS))
for cat in ['High', 'Mid', 'Low', 'All']:
    line = f'{cat:<6}'
    for method, (_, key) in METHODS.items():
        vals = []
        for rxn in REACTIONS:
            if cat != 'All' and MR[rxn] != cat:
                continue
            r = bm_by_rxn[rxn]
            ref = r.get('neb_wb97m_fwd_meV')
            pred = r.get(key)
            if ref is not None and pred is not None:
                vals.append(abs(pred - ref))
        mae = np.mean(vals) if vals else float('nan')
        line += f'{mae:>16.0f}'
    print(line)

# RMSD bar chart by MR category - ablation
fig, ax = plt.subplots(figsize=(7, 4.5))
cats = ['High', 'Mid', 'Low']
x = np.arange(len(cats))
width = 0.8 / len(METHODS)
for i, m in enumerate(METHODS):
    means = []
    for cat in cats:
        vals = [row[m] for row in rmsd_rows if row['mr'] == cat and row[m] is not None]
        means.append(np.mean(vals) if vals else 0)
    ax.bar(x + i*width - 0.4 + width/2, means, width, label=m, color=METHOD_COLOR[m])
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel('Mean TS RMSD vs ORCA wB97M-V NEB (Å)')
ax.set_title('Delta-correction ablation: bare MACE vs MACE+delta fw2')
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/delta_ablation_rmsd.png', dpi=150)
plt.close(fig)
print(f'\nAblation plot saved: {OUT_DIR}/delta_ablation_rmsd.png')

# ---------- 3. refresh MR-Optimized RMSD comparison: now 9/10 valid (exclude rxn1320) ----------
print('\n\n=== MR-Optimized RMSD comparison refresh (n=9, excluding rxn1320) ===\n')

HIGH10 = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150']
ALL_METHODS = {
    'ORCA wB97M-V': 'orca_neb_results',
    'UMA-S':        'uma_neb_results',
    'UMA-M':        'uma_m_neb_results',
    'eSEN':         'esen_neb_results',
    'MACE+delta fw2': 'mace_delta_neb_results_fw2',
}
ALL_METHOD_COLOR = {'ORCA wB97M-V': 'black', 'UMA-S': 'tab:blue', 'UMA-M': 'tab:purple',
                     'eSEN': 'tab:brown', 'MACE+delta fw2': 'tab:red'}

VALID = {'rxn7949','rxn8832','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}

mr_opt = {}
for rxn in HIGH10:
    jpath = f'{HOME}/nevpt2_optts_results/{rxn}_avas/nevpt2_optts_results.json'
    if not os.path.exists(jpath):
        continue
    with open(jpath) as f:
        d = json.load(f)
    mr_opt[rxn] = {'xyz': f'{HOME}/nevpt2_optts_results/{rxn}_avas/ts_casscf_opt.xyz', 'ncas': d['ncas']}

rmsd_rows9 = {}
for rxn in HIGH10:
    if rxn not in mr_opt:
        continue
    mropt = load_xyz(mr_opt[rxn]['xyz'])
    row = {}
    for method, dirname in ALL_METHODS.items():
        geo = load_xyz(f'{HOME}/{dirname}/{rxn}/transition_state.xyz')
        row[method] = kabsch_rmsd(mropt, geo) if (mropt is not None and geo is not None) else None
    rmsd_rows9[rxn] = row

valid_rxns = [r for r in HIGH10 if r in rmsd_rows9 and r in VALID]
n = len(valid_rxns)
print(f'{"rxn":<10}' + ''.join(f'{m:>16}' for m in ALL_METHODS))
for rxn in valid_rxns:
    line = f'{rxn:<10}'
    for m in ALL_METHODS:
        v = rmsd_rows9[rxn][m]
        line += f'{v:>16.3f}' if v is not None else f'{"N/A":>16}'
    print(line)

means = {}
for m in ALL_METHODS:
    vals = [rmsd_rows9[r][m] for r in valid_rxns if rmsd_rows9[r].get(m) is not None]
    means[m] = np.mean(vals) if vals else float('nan')

print()
means_line = f'{"Mean(n="+str(n)+")":<22}'
for m in ALL_METHODS:
    means_line += f'{means[m]:>16.3f}'
print(means_line)

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(valid_rxns)); width = 0.8 / len(ALL_METHODS)
for i, (m, color) in enumerate(ALL_METHOD_COLOR.items()):
    vals = [rmsd_rows9[r][m] if rmsd_rows9[r].get(m) is not None else 0 for r in valid_rxns]
    ax.bar(x + i*width - 0.4 + width/2, vals, width, label=f'{m} (mean={means[m]:.3f} Å)', color=color)
ax.set_xticks(x); ax.set_xticklabels(valid_rxns, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('RMSD vs CASSCF+NEVPT2 OptTS (Å)')
ax.set_title(
    f'TS geometry RMSD vs MR-Optimized reference — {n} High-MR reactions\n'
    '(CASSCF(AVAS)/NEVPT2 OptTS gold standard, full active spaces, no pruning)')
ax.legend(fontsize=8.5, loc='upper right')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mr_optimized_rmsd_all10.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved: {OUT_DIR}/mr_optimized_rmsd_all10.png')
