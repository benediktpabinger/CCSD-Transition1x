import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HOST = 'slid.fysik.dtu.dk'
USER = os.environ.get('SSH_USER', 's242862')
PASS = os.environ.get('SSH_PASS')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()

# Check first few SLURM logs
for i in range(3):
    log = f'/home/energy/s242862/logs/esen_neb_10438440_{i}.log'
    content = run(f'cat {log} 2>/dev/null || echo NO LOG')
    print(f'=== task {i} ===')
    print(content[-3000:] if len(content) > 3000 else content)
    print()

ssh.close()
