import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])
def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'

remote_script = r"""
import sqlite3, numpy as np, json, os

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
BASE = '/home/energy/s242862/mace_delta_neb_results'
BM_PATH = '/home/energy/s242862/delta_head/full_benchmark_results.json'

def get_barriers(rxn):
    db_path = f'{BASE}/{rxn}/neb.db'
    fmaxs_path = f'{BASE}/{rxn}/fmaxs.json'
    # Use sqlite3 directly to avoid ASE buffer stride errors
    # n_images is always 10 (1 reactant + 8 intermediates + 1 product); hardcoded to
    # avoid dividing by n_iters which breaks when neb.db accumulated data from prior runs.
    con = sqlite3.connect(db_path)
    all_e = [r[0] for r in con.execute('SELECT energy FROM systems ORDER BY id').fetchall()]
    con.close()
    e = np.array(all_e[-10:])
    rel = (e - e[0]) * 1000
    fwd = float(rel.max())
    rev = float(fwd - rel[-1])
    fmax = float(json.load(open(fmaxs_path))[-1])
    return fwd, rev, fmax

# Load benchmark results
with open(BM_PATH) as f:
    bm = json.load(f)
bm_by_rxn = {r['rxn']: r for r in bm['reactions']}

# Collect all barriers
results = {}
errors = []
for rxn in REACTIONS:
    try:
        fwd, rev, fmax = get_barriers(rxn)
        results[rxn] = {'fwd': fwd, 'rev': rev, 'fmax': fmax}
    except Exception as e:
        errors.append(f'{rxn}: {e}')

if errors:
    print('ERRORS:', errors)

# Update full_benchmark_results.json
for r in bm['reactions']:
    rxn = r['rxn']
    if rxn in results:
        r['mace_delta_neb_fwd_meV'] = round(results[rxn]['fwd'], 1)
        r['mace_delta_neb_rev_meV'] = round(results[rxn]['rev'], 1)
        r['mace_delta_neb_fmax']    = round(results[rxn]['fmax'], 4)

with open(BM_PATH, 'w') as f:
    json.dump(bm, f, indent=2)
print(f'Updated {BM_PATH} with mace_delta_neb results for {len(results)} reactions')

# Print full comparison table
print()
print(f'{"rxn":<10} {"MR":<5} {"CCSD(T)":>8} {"wB97M-V":>8} {"eSEN":>8} {"UMA-s":>8} {"MACEd-SP":>10} {"MACEd-NEB":>11} {"fmax":>7}')
print('-' * 85)

mr_errors = {'High': [], 'Mid': [], 'Low': []}
for rxn in REACTIONS:
    mr = 'High' if rxn in TOP10 else ('Low' if rxn in BOTTOM10 else 'Mid')
    r = bm_by_rxn[rxn]
    ccsdt = r.get('ccsdt_fwd_meV', float('nan'))
    wb97  = r.get('neb_wb97m_fwd_meV', float('nan'))
    esen  = r.get('esen_neb_fwd_meV', float('nan'))
    uma   = r.get('uma_neb_fwd_meV', float('nan'))
    mace_sp = r.get('delta_fwd_meV', float('nan'))
    if rxn in results:
        mneb = results[rxn]['fwd']
        fmax = results[rxn]['fmax']
        fmax_str = f'{fmax:.4f}'
        err = mneb - ccsdt
        mr_errors[mr].append(abs(err))
    else:
        mneb = float('nan')
        fmax_str = 'N/A'
    print(f'{rxn:<10} {mr:<5} {ccsdt:>8.0f} {wb97-ccsdt:>+8.0f} {esen-ccsdt:>+8.0f} {uma-ccsdt:>+8.0f} {mace_sp-ccsdt:>+10.0f} {mneb-ccsdt:>+11.0f} {fmax_str:>7}')

print()
print('Values shown as error vs CCSD(T) in meV (except CCSD(T) column which is absolute)')
print()
print('MAE vs CCSD(T):')
for mr, errs in mr_errors.items():
    if errs:
        print(f'  {mr} MR (n={len(errs)}): MACEd-NEB = {np.mean(errs):.0f} meV')
"""

sftp = ssh.open_sftp()
with sftp.file(f'{REMOTE}/_collect_mace_delta_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

print(run('module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_collect_mace_delta_remote.py 2>&1'))
ssh.close()
