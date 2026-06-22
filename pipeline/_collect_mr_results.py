import paramiko, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)

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
OUT_DIR = f'{HOME}/benchmark_plots'
os.makedirs(OUT_DIR, exist_ok=True)

HIGH10 = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150']
METHODS = {
    'ORCA wB97M-V': 'orca_neb_results',
    'UMA-S':        'uma_neb_results',
    'UMA-M':        'uma_m_neb_results',
    'eSEN':         'esen_neb_results',
    'MACE+d fw2':   'mace_delta_neb_results_fw2',
}
METHOD_COLOR = {'ORCA wB97M-V': 'black', 'UMA-S': 'tab:blue', 'UMA-M': 'tab:purple',
                'eSEN': 'tab:brown', 'MACE+d fw2': 'tab:red'}

def kabsch_rmsd(P, Q):
    P = P - P.mean(axis=0); Q = Q - Q.mean(axis=0)
    H = P.T @ Q; U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((P @ R.T) - Q)**2, axis=1))))

def load_xyz(path):
    return read(path).get_positions() if os.path.exists(path) else None

# ── 1. Collect all 10 MR-Optimized results ───────────────────────────────────
print('=== CASSCF+NEVPT2 OptTS barriers (all 10 High-MR reactions) ===\n')
print(f'{"rxn":<10} {"active space":>13} {"threshold":>10} {"fwd (meV)":>10} {"rev (meV)":>10}  provenance')
print('-' * 90)

mr_opt = {}
for rxn in HIGH10:
    jpath = f'{HOME}/nevpt2_optts_results/{rxn}_avas/nevpt2_optts_results.json'
    with open(jpath) as f:
        d = json.load(f)
    ncas    = d['ncas']; nelecas = d['nelecas']
    fwd     = d['barrier_forward_meV']; rev = d['barrier_reverse_meV']
    thresh  = d.get('avas_threshold', '?')
    pruned  = d.get('prune_info', {}).get('pruned', 'N/A')
    solver  = d.get('solver_used_ts', '?')
    note    = 'exception' if rxn == 'rxn4113' else f'pruned={pruned}, solver={solver}'
    print(f'{rxn:<10} ({nelecas}e,{ncas}o){"":<3} thr={thresh}   {fwd:>10.1f} {rev:>10.1f}  {note}')
    mr_opt[rxn] = {'fwd': fwd, 'rev': rev, 'xyz': f'{HOME}/nevpt2_optts_results/{rxn}_avas/ts_casscf_opt.xyz'}

# ── 2. RMSD vs MR-Optimized TS (all 10 reactions) ────────────────────────────
print('\n=== RMSD vs MR-Optimized TS geometry (all 10 High-MR reactions) ===\n')
print(f'{"rxn":<10}' + ''.join(f'{m:>14}' for m in METHODS))
print('-' * (10 + 14*len(METHODS)))

rmsd_rows = {}
for rxn in HIGH10:
    mropt = load_xyz(mr_opt[rxn]['xyz'])
    row = {}
    line = f'{rxn:<10}'
    for method, dirname in METHODS.items():
        geo = load_xyz(f'{HOME}/{dirname}/{rxn}/transition_state.xyz')
        v = kabsch_rmsd(mropt, geo) if (mropt is not None and geo is not None) else None
        row[method] = v
        line += f'{v:>14.3f}' if v is not None else f'{"N/A":>14}'
    rmsd_rows[rxn] = row
    print(line)

print()
print(f'{"":10}' + ''.join(f'{m:>14}' for m in METHODS))
for method in METHODS:
    vals = [rmsd_rows[r][method] for r in HIGH10 if rmsd_rows[r][method] is not None]
    print(f'{"Mean":<10}' + ''.join(
        f'{np.mean([rmsd_rows[r][m] for r in HIGH10 if rmsd_rows[r][m] is not None]):>14.3f}'
        for m in METHODS
    ))
    break

means = {}
for method in METHODS:
    vals = [rmsd_rows[r][method] for r in HIGH10 if rmsd_rows[r][method] is not None]
    means[method] = np.mean(vals) if vals else float('nan')
print(f'{"Mean":<10}' + ''.join(f'{means[m]:>14.3f}' for m in METHODS))

# ── 3. Plot updated MR-Optimized RMSD bar chart ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(HIGH10)); width = 0.8 / len(METHODS)
for i, (m, color) in enumerate(METHOD_COLOR.items()):
    vals = [rmsd_rows[r][m] if rmsd_rows[r][m] is not None else 0 for r in HIGH10]
    ax.bar(x + i*width - 0.4 + width/2, vals, width, label=f'{m} (mean={means[m]:.3f} Å)', color=color)
ax.set_xticks(x); ax.set_xticklabels(HIGH10, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('RMSD vs CASSCF+NEVPT2 OptTS (Å)')
ax.set_title('TS geometry RMSD vs MR-Optimized reference — all 10 High-MR reactions\n(gold-standard CASSCF(AVAS)+NEVPT2 OptTS)')
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mr_optimized_rmsd_all10.png', dpi=150)
plt.close(fig)
print(f'\nPlot saved: {OUT_DIR}/mr_optimized_rmsd_all10.png')
"""

sftp = client.open_sftp()
with sftp.file(f'{REMOTE}/_collect_mr_results_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

_, out, err = client.exec_command(
    'module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_collect_mr_results_remote.py'
)
print(out.read().decode('utf-8', errors='replace'))
print(err.read().decode('utf-8', errors='replace'))

local_out = r'c:\Transition 1X\Transition 1x\Transition1x\benchmark_plots'
os.makedirs(local_out, exist_ok=True)
sftp = client.open_sftp()
sftp.get('/home/energy/s242862/benchmark_plots/mr_optimized_rmsd_all10.png',
         f'{local_out}\\mr_optimized_rmsd_all10.png')
sftp.close()
client.close()
