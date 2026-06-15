import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOCAL    = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE   = '/home/energy/s242862/pipeline'
PIPELINE = '/home/energy/s242862/pipeline/delta'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])
sftp = ssh.open_sftp()

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

# Upload updated scripts
print('Uploading scripts...')
sftp.put(f'{LOCAL}\\delta\\train_delta_sp.py',      f'{PIPELINE}/train_delta_sp.py')
sftp.put(f'{LOCAL}\\delta\\job_train_delta_sp.sh',  f'{PIPELINE}/job_train_delta_sp.sh')
sftp.put(f'{LOCAL}\\delta\\sample_train_reactions.py', f'{PIPELINE}/sample_train_reactions.py')
sftp.close()
print('Uploaded.')

# Generate 5000-reaction list
print('\n=== Generating reaction list (5000 reactions) ===')
print(run(f'module load Python/3.13.5-GCCcore-14.3.0 && python3 {PIPELINE}/sample_train_reactions.py --n-reactions 5000 --seed 42 2>&1'))

# Check list
print('\n=== Reaction list ===')
print(run(f'wc -l /home/energy/s242862/ccsd_dataset/train_delta_rxns.txt'))
print(run(f'head -5 /home/energy/s242862/ccsd_dataset/train_delta_rxns.txt'))

# Submit 3 arrays (max 2000 per array)
print('\n=== Submitting SLURM arrays ===')
print(run(f'chmod +x {PIPELINE}/job_train_delta_sp.sh'))
print('Array 1 (reactions 1-2000):')
print(run(f'sbatch --export=OFFSET=0 {PIPELINE}/job_train_delta_sp.sh'))
print('Array 2 (reactions 2001-4000):')
print(run(f'sbatch --export=OFFSET=2000 {PIPELINE}/job_train_delta_sp.sh'))
print('Array 3 (reactions 4001-5000):')
print(run(f'sbatch --array=0-999 --export=OFFSET=4000 {PIPELINE}/job_train_delta_sp.sh'))

# Queue status
print('\n=== Queue ===')
print(run('squeue -u s242862'))

ssh.close()
