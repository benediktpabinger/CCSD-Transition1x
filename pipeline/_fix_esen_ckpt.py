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

print('=== cancelling jobs ===')
run('scancel 10438410')

print()
print('=== moving checkpoint to correct location ===')
run('mv ~/checkpoints/checkpoints/esen_sm_conserving_all.pt ~/checkpoints/esen_sm_conserving_all.pt')
run('ls -lh ~/checkpoints/esen_sm_conserving_all.pt')

print()
print('=== resubmitting ===')
run('mkdir -p /home/energy/s242862/logs')
run('sbatch /home/energy/s242862/pipeline/job_esen_neb.sh')

print()
print('=== queue ===')
run('squeue -u s242862')

ssh.close()
