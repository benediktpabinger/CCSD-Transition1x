import paramiko, os

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'
HOST   = 'slid.fysik.dtu.dk'
USER   = os.environ.get('SSH_USER', 's242862')
PASS   = os.environ.get('SSH_PASS')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)
sftp = ssh.open_sftp()

def run(cmd):
    _, out, err = ssh.exec_command(cmd, get_pty=True)
    o = out.read().decode('utf-8', errors='replace').strip()
    if o:
        print(o)
    return o

# 1. Upload scripts
print('=== uploading scripts ===')
for fname in ['esen_neb.py', 'job_esen_neb.sh', 'setup_esen.sh']:
    sftp.put(f'{LOCAL}\\{fname}', f'{REMOTE}/{fname}')
    print(f'  uploaded {fname}')
run(f'chmod +x {REMOTE}/job_esen_neb.sh {REMOTE}/setup_esen.sh')

sftp.close()

# 2. Install fairchem if needed
print()
print('=== checking fairchem ===')
result = run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "import fairchem.core; print(fairchem.core.__version__)" 2>&1')
if 'ModuleNotFoundError' in result or result == '':
    print('Installing fairchem-core ...')
    run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -m pip install fairchem-core huggingface_hub -q 2>&1 | tail -3')
    print('Done.')
else:
    print(f'Already installed: {result}')

# 3. Download checkpoint if needed
print()
print('=== checking checkpoint ===')
ckpt = run('ls ~/checkpoints/esen_sm_conserving_all.pt 2>&1')
if 'No such file' in ckpt or 'cannot access' in ckpt:
    print('Downloading eSEN checkpoint (this may take a few minutes) ...')
    run('mkdir -p ~/checkpoints')
    run('''module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='facebook/OMol25',
    filename='esen_sm_conserving_all.pt',
    local_dir='/home/energy/s242862/checkpoints'
)
print('Downloaded to:', p)
"''')
else:
    print('Checkpoint already exists.')

# 4. Submit job
print()
print('=== submitting SLURM array job ===')
run('mkdir -p /home/energy/s242862/logs')
job_out = run(f'sbatch {REMOTE}/job_esen_neb.sh')
print(job_out)

print()
print('=== queue status ===')
run('squeue -u s242862')

ssh.close()
