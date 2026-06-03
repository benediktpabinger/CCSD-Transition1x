import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])
sftp = ssh.open_sftp()
sftp.put(f'{LOCAL}\\_esen_ts_rmsd.py', f'{REMOTE}/_esen_ts_rmsd.py')
sftp.close()

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    print(out.read().decode('utf-8', errors='replace').strip())

run('module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_esen_ts_rmsd.py 2>&1')
ssh.close()
