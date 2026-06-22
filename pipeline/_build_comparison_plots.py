import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'

remote_script = r"""
import json, os
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
MR_COLOR  = {'High': 'tab:red', 'Mid': 'tab:orange', 'Low': 'tab:green'}
MR_MARKER = {'High': 'o', 'Mid': 's', 'Low': '^'}

METHODS = {
    'UMA-S':       ('uma_neb_results',            'uma_neb_fwd_meV'),
    'UMA-M':       ('uma_m_neb_results',           'uma_m_neb_fwd_meV'),
    'eSEN':        ('esen_neb_results',            'esen_neb_fwd_meV'),
    'MACE+d fw2':  ('mace_delta_neb_results_fw2',  'delta_fw2_neb_fwd_meV'),
}
METHOD_COLOR = {'UMA-S': 'tab:blue', 'UMA-M': 'tab:purple', 'eSEN': 'tab:brown', 'MACE+d fw2': 'tab:red'}

# ---------- load barrier data ----------
with open(BM_PATH) as f:
    bm = json.load(f)
bm_by_rxn = {r['rxn']: r for r in bm['reactions']}

def kabsch_rmsd(P, Q):
    P = P - P.mean(axis=0); Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((P @ R.T) - Q)**2, axis=1))))

def load_xyz(path):
    if not os.path.exists(path):
        return None
    return read(path).get_positions()

# ---------- RMSD table ----------
rmsd_rows = []
for rxn in REACTIONS:
    orca = load_xyz(f'{HOME}/orca_neb_results/{rxn}/transition_state.xyz')
    row = {'rxn': rxn, 'mr': MR[rxn]}
    for method, (dirname, _) in METHODS.items():
        geo = load_xyz(f'{HOME}/{dirname}/{rxn}/transition_state.xyz')
        row[method] = kabsch_rmsd(orca, geo) if (orca is not None and geo is not None) else None
    rmsd_rows.append(row)

print(f'{"rxn":<10} {"MR":<5}' + ''.join(f'{m:>12}' for m in METHODS))
print('-' * (15 + 12*len(METHODS)))
for row in rmsd_rows:
    line = f'{row["rxn"]:<10} {row["mr"]:<5}'
    for m in METHODS:
        v = row[m]
        line += f'{v:>12.3f}' if v is not None else f'{"N/A":>12}'
    print(line)

print()
print(f'{"MR":<6}' + ''.join(f'{m:>12}' for m in METHODS) + f'{"n":>5}')
for cat in ['High', 'Mid', 'Low']:
    line = f'{cat:<6}'
    n = 0
    for m in METHODS:
        vals = [row[m] for row in rmsd_rows if row['mr'] == cat and row[m] is not None]
        n = len(vals)
        line += f'{np.mean(vals):>12.3f}' if vals else f'{"N/A":>12}'
    line += f'{n:>5}'
    print(line)
line = f'{"All":<6}'
for m in METHODS:
    vals = [row[m] for row in rmsd_rows if row[m] is not None]
    line += f'{np.mean(vals):>12.3f}'
line += f'{len(rmsd_rows):>5}'
print(line)

# RMSD bar chart by MR category
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
ax.set_title('TS geometry RMSD by MR category')
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/rmsd_by_mr_category.png', dpi=150)
plt.close(fig)

# ---------- barrier error vs wB97M-V ----------
errors = {m: [] for m in METHODS}
mr_list = []
ref_list = []
for rxn in REACTIONS:
    r = bm_by_rxn[rxn]
    ref = r.get('neb_wb97m_fwd_meV')
    if ref is None:
        continue
    ref_list.append(ref)
    mr_list.append(MR[rxn])
    for method, (_, key) in METHODS.items():
        pred = r.get(key)
        errors[method].append((pred - ref) if pred is not None else np.nan)

# Error histogram: small multiples, one panel per method
fig, axes = plt.subplots(1, len(METHODS), figsize=(4*len(METHODS), 4), sharey=True)
bins = np.linspace(-1000, 1000, 21)
for ax, m in zip(axes, METHODS):
    vals = np.array(errors[m])
    vals = vals[~np.isnan(vals)]
    ax.hist(vals, bins=bins, color=METHOD_COLOR[m], edgecolor='black', alpha=0.8)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_title(f'{m}\nMAE={np.mean(np.abs(vals)):.0f} meV, bias={np.mean(vals):+.0f} meV')
    ax.set_xlabel('Predicted - wB97M-V (meV)')
axes[0].set_ylabel('Count')
fig.suptitle('Forward barrier error vs ORCA wB97M-V NEB reference (all 30 reactions)')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/barrier_error_histogram.png', dpi=150)
plt.close(fig)

# Parity plot
fig, ax = plt.subplots(figsize=(6.5, 6.5))
lims = [min(ref_list)-200, max(ref_list)+200]
ax.plot(lims, lims, 'k--', linewidth=1, label='y = x')
for method, (_, key) in METHODS.items():
    xs, ys, mrs = [], [], []
    for rxn in REACTIONS:
        r = bm_by_rxn[rxn]
        ref = r.get('neb_wb97m_fwd_meV')
        pred = r.get(key)
        if ref is None or pred is None:
            continue
        xs.append(ref); ys.append(pred); mrs.append(MR[rxn])
    for cat in ['High', 'Mid', 'Low']:
        xi = [x for x, c in zip(xs, mrs) if c == cat]
        yi = [y for y, c in zip(ys, mrs) if c == cat]
        ax.scatter(xi, yi, color=METHOD_COLOR[method], marker=MR_MARKER[cat],
                   s=50, alpha=0.85, edgecolor='black', linewidth=0.4,
                   label=f'{method} ({cat})' if False else None)
# Build a clean legend: colors = method, marker shapes = MR category
from matplotlib.lines import Line2D
method_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markeredgecolor='black', markersize=8, label=m)
                  for m, c in METHOD_COLOR.items()]
mr_handles = [Line2D([0],[0], marker=mk, color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=cat)
              for cat, mk in MR_MARKER.items()]
leg1 = ax.legend(handles=method_handles, loc='upper left', title='Method', fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=mr_handles, loc='lower right', title='MR category', fontsize=9)
ax.set_xlabel('ORCA wB97M-V NEB forward barrier (meV)')
ax.set_ylabel('Predicted forward barrier (meV)')
ax.set_title('Parity plot: forward barrier vs wB97M-V reference')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/barrier_parity_plot.png', dpi=150)
plt.close(fig)

print()
print('MAE / bias vs wB97M-V (meV), by MR category:')
print(f'{"MR":<6}' + ''.join(f'{m:>14}' for m in METHODS))
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
                vals.append(pred - ref)
        mae = np.mean(np.abs(vals)) if vals else float('nan')
        line += f'{mae:>14.0f}'
    print(line)

print(f'\nPlots saved to {OUT_DIR}/')
"""

sftp = ssh.open_sftp()
with sftp.file(f'{REMOTE}/_build_comparison_plots_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

_, out, err = ssh.exec_command(
    'module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_build_comparison_plots_remote.py'
)
print(out.read().decode('utf-8', errors='replace'))
print(err.read().decode('utf-8', errors='replace'))

local_out = r'c:\Transition 1X\Transition 1x\Transition1x\benchmark_plots'
import os
os.makedirs(local_out, exist_ok=True)
sftp = ssh.open_sftp()
for fname in ['rmsd_by_mr_category.png', 'barrier_error_histogram.png', 'barrier_parity_plot.png']:
    sftp.get(f'/home/energy/s242862/benchmark_plots/{fname}', f'{local_out}\\{fname}')
sftp.close()
print('Pulled plots to', local_out)

ssh.close()
