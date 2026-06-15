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
    'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
    'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
    'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
    'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
]

print('=== queue ===')
print(run('squeue -u s242862 | head -15'))

print()
converged, running, missing = [], [], []
for rxn in REACTIONS:
    base = f'/home/energy/s242862/mace_delta_neb_results/{rxn}'
    has_conv = run(f'test -f {base}/converged && echo yes || echo no') == 'yes'
    has_log  = run(f'test -f {base}/neb.log && echo yes || echo no') == 'yes'
    if has_conv:  converged.append(rxn)
    elif has_log: running.append(rxn)
    else:         missing.append(rxn)

print(f'Converged:   {len(converged)} — {converged}')
print(f'In progress: {len(running)} — {running}')
print(f'Not started: {len(missing)}')

if converged:
    sample = converged[0]
    print()
    print(f'=== last 5 lines of {sample} neb.log ===')
    print(run(f'tail -5 /home/energy/s242862/mace_delta_neb_results/{sample}/neb.log 2>/dev/null'))

ssh.close()
