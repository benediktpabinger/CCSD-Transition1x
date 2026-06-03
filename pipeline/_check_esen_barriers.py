import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

REACTIONS = [
    'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
    'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150',
    'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
    'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
    'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
    'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
]

# Read NEB reference barriers from existing results for comparison
result = run('''python3 -c "
import ase.db, json, numpy as np

REACTIONS = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
             'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150',
             'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
             'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
             'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
             'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962']

with open('/home/energy/s242862/delta_head/full_benchmark_results.json') as f:
    bm = {r['rxn']: r for r in json.load(f)['reactions']}

TOP10    = set(['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'])
MIDDLE10 = set(['rxn0896','rxn1154','rxn5690','rxn4513','rxn7955','rxn4519','rxn4500','rxn2553','rxn8829','rxn1155'])

print(f'{\\'Rxn\\':<12} {\\'Group\\':<5} {\\'eSEN fwd\\':<10} {\\'wB97M-V fwd\\':<13} {\\'CCSD(T) fwd\\':<13} {\\'eSEN-CCSD(T)\\':<12} {\\'ndb rows\\':<10} {\\'fmax\\':<8}')
print('-' * 85)

for rxn in REACTIONS:
    base = f'/home/energy/s242862/esen_neb_results/{rxn}'
    try:
        with ase.db.connect(f'{base}/neb.db') as db:
            rows = list(db.select())
        n_total = len(rows)
        e_last = np.array([r.toatoms().get_potential_energy() for r in rows[-10:]])
        rel = (e_last - e_last[0]) * 1000
        fwd = rel.max()
        ts_idx = int(rel.argmax())

        # fmax from fmaxs.json
        import json as json2
        with open(f'{base}/fmaxs.json') as f2:
            fmaxs = json2.load(f2)
        final_fmax = fmaxs[-1] if fmaxs else None

        ref = bm.get(rxn, {})
        neb_m = ref.get('neb_wb97m_fwd_meV')
        cct   = ref.get('ccsdt_fwd_meV')
        group = 'High' if rxn in TOP10 else ('Mid' if rxn in MIDDLE10 else 'Low')
        diff  = fwd - cct if cct else None

        neb_s  = f'{neb_m:.0f}' if neb_m else 'N/A'
        cct_s  = f'{cct:.0f}'  if cct   else 'N/A'
        diff_s = f'{diff:+.0f}' if diff is not None else 'N/A'
        fmax_s = f'{final_fmax:.3f}' if final_fmax else '?'
        print(f'{rxn:<12} {group:<5} {fwd:<10.0f} {neb_s:<13} {cct_s:<13} {diff_s:<12} {n_total:<10} {fmax_s:<8}')

    except Exception as e:
        print(f'{rxn:<12} ERROR: {e}')
" 2>&1''')

print(result)
ssh.close()
