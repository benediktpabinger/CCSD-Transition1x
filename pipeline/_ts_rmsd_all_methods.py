import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'

sftp = ssh.open_sftp()
sftp.put(f'{LOCAL}\\_ts_rmsd_remote.py', f'{REMOTE}/_ts_rmsd_remote.py')
sftp.close()

print(run('module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_ts_rmsd_remote.py 2>&1'))
ssh.close()
