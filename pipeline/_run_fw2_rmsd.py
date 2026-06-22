import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'

sftp = ssh.open_sftp()
sftp.put(f'{LOCAL}\\_ts_rmsd_fw2_remote.py', f'{REMOTE}/_ts_rmsd_fw2_remote.py')
sftp.close()

_, out, err = ssh.exec_command(
    'module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_ts_rmsd_fw2_remote.py'
)
print(out.read().decode('utf-8', errors='replace'))
print(err.read().decode('utf-8', errors='replace'))
ssh.close()
