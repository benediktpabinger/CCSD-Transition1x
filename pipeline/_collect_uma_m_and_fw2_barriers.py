import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'

remote_script = r"""
import sqlite3, numpy as np, json

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

BM_PATH = '/home/energy/s242862/delta_head/full_benchmark_results.json'
SOURCES = {
    'uma_m':    ('/home/energy/s242862/uma_m_neb_results', 'uma_m_neb'),
    'delta_fw2': ('/home/energy/s242862/mace_delta_neb_results_fw2', 'delta_fw2_neb'),
}

def get_barriers(base, rxn):
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

with open(BM_PATH) as f:
    bm = json.load(f)
bm_by_rxn = {r['rxn']: r for r in bm['reactions']}

all_results = {}
for tag, (base, prefix) in SOURCES.items():
    results = {}
    errors = []
    for rxn in REACTIONS:
        try:
            fwd, rev, fmax = get_barriers(base, rxn)
            results[rxn] = {'fwd': fwd, 'rev': rev, 'fmax': fmax}
        except Exception as e:
            errors.append(f'{rxn}: {e}')
    if errors:
        print(f'{tag} ERRORS:', errors)
    all_results[tag] = results
    for r in bm['reactions']:
        rxn = r['rxn']
        if rxn in results:
            r[f'{prefix}_fwd_meV'] = round(results[rxn]['fwd'], 1)
            r[f'{prefix}_rev_meV'] = round(results[rxn]['rev'], 1)
            r[f'{prefix}_fmax']    = round(results[rxn]['fmax'], 4)
    print(f'{tag}: collected {len(results)}/{len(REACTIONS)} reactions')

with open(BM_PATH, 'w') as f:
    json.dump(bm, f, indent=2)
print(f'\nPatched {BM_PATH}')

print()
print(f'{"rxn":<10} {"MR":<5} {"wB97M-V":>9} {"eSEN":>9} {"UMA-S":>9} {"UMA-M":>9} {"fw1-old":>9} {"fw2-new":>9}')
print('-' * 75)
for rxn in REACTIONS:
    mr = 'High' if rxn in TOP10 else ('Low' if rxn in BOTTOM10 else 'Mid')
    r = bm_by_rxn[rxn]
    wb97 = r.get('neb_wb97m_fwd_meV')
    esen = r.get('esen_neb_fwd_meV')
    uma_s = r.get('uma_neb_fwd_meV')
    uma_m = r.get('uma_m_neb_fwd_meV')
    fw1 = r.get('mace_delta_neb_fwd_meV')
    fw2 = r.get('delta_fw2_neb_fwd_meV')
    def fmt(x):
        return f'{x:.0f}' if x is not None else 'N/A'
    print(f'{rxn:<10} {mr:<5} {fmt(wb97):>9} {fmt(esen):>9} {fmt(uma_s):>9} {fmt(uma_m):>9} {fmt(fw1):>9} {fmt(fw2):>9}')
"""

sftp = ssh.open_sftp()
with sftp.file(f'{REMOTE}/_collect_uma_m_and_fw2_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

_, out, err = ssh.exec_command(
    'module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_collect_uma_m_and_fw2_remote.py'
)
print(out.read().decode('utf-8', errors='replace'))
print(err.read().decode('utf-8', errors='replace'))
ssh.close()
