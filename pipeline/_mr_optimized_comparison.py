import paramiko, sys, os
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
OUT_DIR = f'{HOME}/benchmark_plots'
os.makedirs(OUT_DIR, exist_ok=True)

MR_OPT_REACTIONS = ['rxn7949', 'rxn6196', 'rxn0346', 'rxn4113']

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
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((P @ R.T) - Q)**2, axis=1))))

def load_xyz(path):
    if not os.path.exists(path):
        return None
    return read(path).get_positions()

print('=== RMSD vs MR-Optimized (CASSCF+NEVPT2 OptTS) TS geometry ===\n')
print(f'{"rxn":<10}' + ''.join(f'{m:>16}' for m in METHODS))
rmsd_table = {}
for rxn in MR_OPT_REACTIONS:
    mropt_xyz = f'{HOME}/nevpt2_optts_results/{rxn}_avas/ts_casscf_opt.xyz'
    mropt = load_xyz(mropt_xyz)
    line = f'{rxn:<10}'
    rmsd_table[rxn] = {}
    for method, dirname in METHODS.items():
        geo = load_xyz(f'{HOME}/{dirname}/{rxn}/transition_state.xyz')
        v = kabsch_rmsd(mropt, geo) if (mropt is not None and geo is not None) else None
        rmsd_table[rxn][method] = v
        line += f'{v:>16.3f}' if v is not None else f'{"N/A":>16}'
    print(line)

print()
print('Mean RMSD vs MR-Optimized across these 3 reactions:')
for method in METHODS:
    vals = [rmsd_table[rxn][method] for rxn in MR_OPT_REACTIONS if rmsd_table[rxn][method] is not None]
    if vals:
        print(f'  {method:<14} {np.mean(vals):.3f} A')

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(MR_OPT_REACTIONS))
width = 0.8 / len(METHODS)
for i, m in enumerate(METHODS):
    vals = [rmsd_table[rxn][m] for rxn in MR_OPT_REACTIONS]
    ax.bar(x + i*width - 0.4 + width/2, vals, width, label=m, color=METHOD_COLOR[m])
ax.set_xticks(x)
ax.set_xticklabels(MR_OPT_REACTIONS)
ax.set_ylabel('RMSD vs MR-Optimized TS (Å)')
ax.set_title('TS geometry RMSD vs CASSCF+NEVPT2 OptTS (gold-standard MR reference)')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mr_optimized_rmsd.png', dpi=150)
plt.close(fig)
print(f'\\nPlot saved to {OUT_DIR}/mr_optimized_rmsd.png')
"""

sftp = ssh.open_sftp()
with sftp.file(f'{REMOTE}/_mr_optimized_comparison_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

_, out, err = ssh.exec_command(
    'module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_mr_optimized_comparison_remote.py'
)
print(out.read().decode('utf-8', errors='replace'))
print(err.read().decode('utf-8', errors='replace'))

local_out = r'c:\Transition 1X\Transition 1x\Transition1x\benchmark_plots'
os.makedirs(local_out, exist_ok=True)
sftp = ssh.open_sftp()
sftp.get('/home/energy/s242862/benchmark_plots/mr_optimized_rmsd.png', f'{local_out}\\mr_optimized_rmsd.png')
sftp.close()
print('Pulled plot to', local_out)

ssh.close()
