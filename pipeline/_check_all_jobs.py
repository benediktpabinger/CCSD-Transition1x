import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])
def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

# Queue overview
print('=== queue ===')
print(run('squeue -u s242862'))

print()

# MACE delta NEB — count converged
print('=== MACE+delta NEB ===')
print(run('''
python3 -c "
import os
BASE = '/home/energy/s242862/mace_delta_neb_results'
rxns = [
    'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
    'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150',
    'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
    'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
    'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
    'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
]
conv = [r for r in rxns if os.path.exists(f'{BASE}/{r}/converged')]
prog = [r for r in rxns if not os.path.exists(f'{BASE}/{r}/converged') and os.path.exists(f'{BASE}/{r}/neb.log')]
miss = [r for r in rxns if r not in conv and r not in prog]
print(f'Converged: {len(conv)}/30 — {conv}')
print(f'Running:   {len(prog)} — {prog}')
print(f'Missing:   {len(miss)} — {miss}')
"
'''))

print()

# ORCA SP — count done
print('=== ORCA SP v2 (train_delta_sp) ===')
print(run('''
python3 -c "
import os, json
BASE = '/home/energy/s242862/train_delta_sp'
rxns = open('/home/energy/s242862/ccsd_dataset/train_delta_rxns.txt').read().splitlines()
done = []
for r in rxns:
    p = f'{BASE}/{r}/results.json'
    if os.path.exists(p):
        d = json.load(open(p))
        if d.get('geometries') and len(d['geometries']) >= 20 and 'forces_wb97m_eV_per_ang' in d['geometries'][0]:
            done.append(r)
print(f'Done (20 geoms + forces): {len(done)}/5000')
print(f'Remaining: {5000 - len(done)}')
"
'''))

ssh.close()
