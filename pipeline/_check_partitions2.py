import paramiko, os

HOST = 'slid.fysik.dtu.dk'
USER = os.environ.get('SSH_USER', 's242862')
PASS = os.environ.get('SSH_PASS')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)

def run(cmd):
    _, out, err = ssh.exec_command(cmd, get_pty=True)
    o = out.read().decode('utf-8', errors='replace').strip()
    if o: print(o)
    return o

print('=== available partitions ===')
run('sinfo -o "%P %G %l %D" | sort')

print()
print('=== python/pip ===')
run('which python3 && python3 --version')
run('python3 -m pip --version')

ssh.close()
