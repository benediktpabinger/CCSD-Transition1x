import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])
sftp = ssh.open_sftp()
sftp.put(f'{LOCAL}\\_collect_uma_barriers.py', f'{REMOTE}/_collect_uma_barriers.py')
sftp.close()

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    print(out.read().decode('utf-8', errors='replace').strip())

run('module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_collect_uma_barriers.py 2>&1')

print('\n=== downloading updated JSON ===')
sftp2 = ssh.open_sftp()
sftp2.get('/home/energy/s242862/delta_head/full_benchmark_results.json',
          r'c:\Transition 1X\Transition 1x\Transition1x\full_benchmark_results.json')
sftp2.close()
print('Done.')
ssh.close()
