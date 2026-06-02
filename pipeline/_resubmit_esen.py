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

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    o = out.read().decode('utf-8', errors='replace').strip()
    if o: print(o)
    return o

# verify fix works before submitting
print('=== verifying fix on cluster ===')
sftp.put(f'{LOCAL}\\esen_neb.py', f'{REMOTE}/esen_neb.py')
print('uploaded esen_neb.py')
result = run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "from fairchem.core import FAIRChemCalculator; calc = FAIRChemCalculator(checkpoint_path=\'/home/energy/s242862/checkpoints/esen_sm_conserving_all.pt\', device=\'cpu\'); print(\'calculator loaded ok\')" 2>&1')

sftp.close()

if 'calculator loaded ok' not in result:
    print('ERROR: calculator still broken, not submitting')
    ssh.close()
    exit(1)

print()
print('=== submitting ===')
run('sbatch /home/energy/s242862/pipeline/job_esen_neb.sh')

print()
print('=== queue ===')
run('squeue -u s242862')

ssh.close()
