import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HOST = 'slid.fysik.dtu.dk'
USER = os.environ.get('SSH_USER', 's242862')
PASS = os.environ.get('SSH_PASS')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    o = out.read().decode('utf-8', errors='replace').strip()
    if o: print(o)
    return o

REACTIONS = [
    'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
    'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150',
    'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
    'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
    'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
    'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
]

print('=== queue status ===')
run('squeue -u s242862')

print()
print('=== results so far ===')
converged, running, failed, missing = [], [], [], []
for rxn in REACTIONS:
    base = f'/home/energy/s242862/esen_neb_results/{rxn}'
    has_converged = run(f'test -f {base}/converged && echo yes || echo no').strip() == 'yes'
    has_log = run(f'test -f {base}/neb.log && echo yes || echo no').strip() == 'yes'
    has_db  = run(f'test -f {base}/neb.db && echo yes || echo no').strip() == 'yes'
    if has_converged:
        converged.append(rxn)
    elif has_log or has_db:
        running.append(rxn)
    else:
        missing.append(rxn)

print(f'Converged : {len(converged)} — {converged}')
print(f'In progress: {len(running)} — {running}')
print(f'Not started: {len(missing)}')

print()
print('=== last 20 lines of first running log ===')
if converged or running:
    sample = (converged + running)[0]
    run(f'tail -20 /home/energy/s242862/esen_neb_results/{sample}/neb.log 2>/dev/null || echo no log yet')
    run(f'tail -5 /home/energy/s242862/logs/esen_neb_10438440_0.log 2>/dev/null || echo no slurm log yet')

ssh.close()
