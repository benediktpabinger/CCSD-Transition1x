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

sftp.put(f'{LOCAL}\\_collect_esen_barriers.py', f'{REMOTE}/_collect_esen_barriers.py')
sftp.close()

run('module load Python/3.13.5-GCCcore-14.3.0 && python3 /home/energy/s242862/pipeline/_collect_esen_barriers.py 2>&1')

# Download updated JSON
print()
print('=== downloading updated full_benchmark_results.json ===')
sftp2 = ssh.open_sftp()
sftp2.get('/home/energy/s242862/delta_head/full_benchmark_results.json',
          r'c:\Transition 1X\Transition 1x\Transition1x\full_benchmark_results.json')
sftp2.close()
print('Downloaded.')

ssh.close()
