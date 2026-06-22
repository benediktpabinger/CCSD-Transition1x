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

RESULTS_DIR = '/home/energy/s242862/mr_benchmark/results'

def load_fwd(rxn, tag):
    suffix = f'_{tag}' if tag else ''
    path = f'{RESULTS_DIR}/{rxn}_ccsdt{suffix}.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get('barrier_fwd_meV')

rows = []
for rxn in REACTIONS:
    orca = load_fwd(rxn, '')
    uma_s = load_fwd(rxn, 'uma_s')
    fw2 = load_fwd(rxn, 'delta_fw2')
    rows.append({'rxn': rxn, 'mr': MR[rxn], 'orca': orca, 'uma_s': uma_s, 'fw2': fw2})

print(f'{"rxn":<10} {"MR":<5} {"CCSDT@ORCA":>11} {"CCSDT@UMA-S":>12} {"CCSDT@fw2":>10} {"UMA-S-ORCA":>11} {"fw2-ORCA":>9}')
print('-' * 80)
for r in rows:
    orca, uma_s, fw2 = r['orca'], r['uma_s'], r['fw2']
    d_uma = (uma_s - orca) if (uma_s is not None and orca is not None) else None
    d_fw2 = (fw2 - orca) if (fw2 is not None and orca is not None) else None
    def f(x, w=11):
        return f'{x:>{w}.1f}' if x is not None else f'{"N/A":>{w}}'
    print(f'{r["rxn"]:<10} {r["mr"]:<5} {f(orca)} {f(uma_s,12)} {f(fw2,10)} {f(d_uma,11)} {f(d_fw2,9)}')

print()
print('Mean |CCSD(T)-on-geometry minus CCSD(T)-on-ORCA-geometry| barrier difference (MAE, meV), by MR category:')
print(f'{"MR":<6} {"UMA-S":>10} {"MACE+d fw2":>12} {"n":>4}')
for cat in ['High', 'Mid', 'Low']:
    d_uma_list = [abs(r['uma_s'] - r['orca']) for r in rows if r['mr'] == cat and r['uma_s'] is not None and r['orca'] is not None]
    d_fw2_list = [abs(r['fw2'] - r['orca']) for r in rows if r['mr'] == cat and r['fw2'] is not None and r['orca'] is not None]
    n = len(d_uma_list)
    mae_uma = np.mean(d_uma_list) if d_uma_list else float('nan')
    mae_fw2 = np.mean(d_fw2_list) if d_fw2_list else float('nan')
    print(f'{cat:<6} {mae_uma:>10.1f} {mae_fw2:>12.1f} {n:>4}')

d_uma_all = [abs(r['uma_s'] - r['orca']) for r in rows if r['uma_s'] is not None and r['orca'] is not None]
d_fw2_all = [abs(r['fw2'] - r['orca']) for r in rows if r['fw2'] is not None and r['orca'] is not None]
print(f'{"All":<6} {np.mean(d_uma_all):>10.1f} {np.mean(d_fw2_all):>12.1f} {len(d_uma_all):>4}')

print()
print('Signed mean (bias, meV), by MR category:')
print(f'{"MR":<6} {"UMA-S":>10} {"MACE+d fw2":>12}')
for cat in ['High', 'Mid', 'Low', 'All']:
    if cat == 'All':
        d_uma_list = [r['uma_s'] - r['orca'] for r in rows if r['uma_s'] is not None and r['orca'] is not None]
        d_fw2_list = [r['fw2'] - r['orca'] for r in rows if r['fw2'] is not None and r['orca'] is not None]
    else:
        d_uma_list = [r['uma_s'] - r['orca'] for r in rows if r['mr'] == cat and r['uma_s'] is not None and r['orca'] is not None]
        d_fw2_list = [r['fw2'] - r['orca'] for r in rows if r['mr'] == cat and r['fw2'] is not None and r['orca'] is not None]
    bias_uma = np.mean(d_uma_list) if d_uma_list else float('nan')
    bias_fw2 = np.mean(d_fw2_list) if d_fw2_list else float('nan')
    print(f'{cat:<6} {bias_uma:>+10.1f} {bias_fw2:>+12.1f}')
"""

sftp = ssh.open_sftp()
with sftp.file(f'{REMOTE}/_ccsdt_geometry_comparison_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

_, out, err = ssh.exec_command(
    'module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_ccsdt_geometry_comparison_remote.py'
)
print(out.read().decode('utf-8', errors='replace'))
print(err.read().decode('utf-8', errors='replace'))
ssh.close()
