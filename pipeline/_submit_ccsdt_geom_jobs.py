import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'

sftp = ssh.open_sftp()
sftp.put(f'{LOCAL}\\mr_benchmark_ccsdt.py', f'{REMOTE}/mr_benchmark_ccsdt.py')
sftp.put(f'{LOCAL}\\job_mr_ccsdt_uma_s.sh', f'{REMOTE}/job_mr_ccsdt_uma_s.sh')
sftp.put(f'{LOCAL}\\job_mr_ccsdt_delta_fw2.sh', f'{REMOTE}/job_mr_ccsdt_delta_fw2.sh')
sftp.close()

for job in ['job_mr_ccsdt_uma_s.sh', 'job_mr_ccsdt_delta_fw2.sh']:
    _, out, err = ssh.exec_command(f'sbatch {REMOTE}/{job}')
    print(job, '->', out.read().decode())
    print(err.read().decode())

ssh.close()
