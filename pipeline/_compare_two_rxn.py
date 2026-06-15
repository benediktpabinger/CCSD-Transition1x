import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

remote_script = r"""
import sqlite3, numpy as np, json

bm_path = '/home/energy/s242862/delta_head/full_benchmark_results.json'
with open(bm_path) as f:
    bm = {r['rxn']: r for r in json.load(f)['reactions']}

for rxn in ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150','rxn9246','rxn4498','rxn1061','rxn4003','rxn4063']:
    base = f'/home/energy/s242862/mace_delta_neb_results/{rxn}'
    con = sqlite3.connect(base + '/neb.db')
    all_e = [r[0] for r in con.execute('SELECT energy FROM systems ORDER BY id').fetchall()]
    con.close()
    e = np.array(all_e[-10:])
    rel = (e - e[0]) * 1000
    fwd = rel.max()
    rev = fwd - rel[-1]
    with open(base + '/fmaxs.json') as f:
        fmax = json.load(f)[-1]
    r = bm[rxn]
    print(f'--- {rxn} ---')
    print(f'  MACE+delta NEB: fwd={fwd:.1f}  rev={rev:.1f}  fmax={fmax:.4f}')
    print(f'  CCSD(T):        fwd={r["ccsdt_fwd_meV"]:.1f}')
    print(f'  wB97M-V (ORCA): fwd={r["neb_wb97m_fwd_meV"]:.1f}')
    print(f'  eSEN NEB:       fwd={r["esen_neb_fwd_meV"]:.1f}')
    print(f'  UMA-s NEB:      fwd={r["uma_neb_fwd_meV"]:.1f}')
    print(f'  MACE (SP):      fwd={r["mace_fwd_meV"]:.1f}')
    print(f'  MACE+d (SP):    fwd={r["delta_fwd_meV"]:.1f}')
    print()
"""

# write and run remotely
sftp = ssh.open_sftp()
with sftp.file('/home/energy/s242862/pipeline/_compare_two_rxn_remote.py', 'w') as f:
    f.write(remote_script)
sftp.close()

print(run('module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_compare_two_rxn_remote.py 2>&1'))
ssh.close()
