"""
Test MACEDeltaCalculator on a single reaction before full NEB submission.
Checks: model loads, z_table/r_max accessible, single energy+forces call works.
"""
import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'
HOST   = 'slid.fysik.dtu.dk'
USER   = os.environ.get('SSH_USER', 's242862')
PASS   = os.environ.get('SSH_PASS')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)
sftp = ssh.open_sftp()

for fname in ['mace_delta_neb.py', 'job_mace_delta_neb.sh']:
    sftp.put(f'{LOCAL}\\{fname}', f'{REMOTE}/{fname}')
    print(f'uploaded {fname}')
run_cmd = f'chmod +x {REMOTE}/job_mace_delta_neb.sh'
ssh.exec_command(run_cmd)
sftp.close()

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

# Submit single test job (rxn4063 — small, low MR, converges fast)
print()
print('=== submitting test job (rxn4063, h200 partition) ===')
test_script = '''#!/bin/bash
#SBATCH --job-name=mace_delta_test
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/home/energy/s242862/logs/mace_delta_test.log

module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.6.0

python3 /home/energy/s242862/pipeline/mace_delta_neb.py \\
    --reaction rxn4063 \\
    --output   /home/energy/s242862/mace_delta_neb_results/rxn4063
'''

# Write test script to cluster
sftp2 = ssh.open_sftp()
with sftp2.open(f'{REMOTE}/job_mace_delta_test.sh', 'w') as f:
    f.write(test_script)
sftp2.close()
print(run(f'chmod +x {REMOTE}/job_mace_delta_test.sh'))
print(run(f'sbatch {REMOTE}/job_mace_delta_test.sh'))
print()
print(run('squeue -u s242862'))

ssh.close()
