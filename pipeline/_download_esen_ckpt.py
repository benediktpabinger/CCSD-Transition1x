import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HOST = 'slid.fysik.dtu.dk'
USER = os.environ.get('SSH_USER', 's242862')
PASS = os.environ.get('SSH_PASS')
HF_TOKEN = os.environ.get('HF_TOKEN')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)

def run(cmd):
    _, out, err = ssh.exec_command(cmd, get_pty=True)
    o = out.read().decode('utf-8', errors='replace').strip()
    if o: print(o)
    return o

print('=== downloading eSEN checkpoint ===')
run('mkdir -p ~/checkpoints')
run(f'''module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='facebook/OMol25',
    filename='checkpoints/esen_sm_conserving_all.pt',
    local_dir='/home/energy/s242862/checkpoints',
    token='{HF_TOKEN}',
)
print('Downloaded to:', p)
"''')

print()
print('=== verifying checkpoint ===')
run('ls -lh ~/checkpoints/esen_sm_conserving_all.pt')

print()
print('=== submitting jobs ===')
run('mkdir -p /home/energy/s242862/logs')
run(f'sbatch /home/energy/s242862/pipeline/job_esen_neb.sh')

print()
print('=== queue status ===')
run('squeue -u s242862')

ssh.close()
