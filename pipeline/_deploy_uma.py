import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'
HOST   = 'slid.fysik.dtu.dk'
USER   = os.environ.get('SSH_USER', 's242862')
PASS   = os.environ.get('SSH_PASS')
HF_TOKEN = os.environ.get('HF_TOKEN')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)
sftp = ssh.open_sftp()

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    o = out.read().decode('utf-8', errors='replace').strip()
    if o: print(o)
    return o

# Upload scripts
for fname in ['uma_neb.py', 'job_uma_neb.sh']:
    sftp.put(f'{LOCAL}\\{fname}', f'{REMOTE}/{fname}')
    print(f'uploaded {fname}')
run(f'chmod +x {REMOTE}/job_uma_neb.sh')
sftp.close()

# Download checkpoint if needed
print()
print('=== checking UMA checkpoint ===')
ckpt = run('ls ~/checkpoints/uma-s-1p2.pt 2>&1')
if 'No such file' in ckpt or 'cannot access' in ckpt:
    print('Downloading uma-s-1p2.pt ...')
    run(f'''module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='facebook/UMA',
    filename='checkpoints/uma-s-1p2.pt',
    local_dir='/home/energy/s242862/checkpoints',
    token='{HF_TOKEN}',
)
print('Downloaded to:', p)
"''')
    run('mv ~/checkpoints/checkpoints/uma-s-1p2.pt ~/checkpoints/uma-s-1p2.pt 2>/dev/null; ls -lh ~/checkpoints/uma-s-1p2.pt')
else:
    print('Checkpoint already exists.')

# Submit
print()
print('=== submitting ===')
run('mkdir -p /home/energy/s242862/logs')
run(f'sbatch {REMOTE}/job_uma_neb.sh')

print()
print('=== queue ===')
run('squeue -u s242862')

ssh.close()
