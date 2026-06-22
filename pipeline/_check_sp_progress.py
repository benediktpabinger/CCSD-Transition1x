import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])
def run(cmd):
    _, out, err = ssh.exec_command(cmd, get_pty=False, timeout=300)
    out.channel.settimeout(300)
    return (out.read() + err.read()).decode('utf-8', errors='replace').strip()

print(run('squeue -u s242862 | tail -5'))
print()
print(run('ls /home/energy/s242862/train_delta_sp | wc -l'))
print()
print('--- task 0 log tail ---')
print(run('tail -5 /home/energy/s242862/logs/train_delta_sp_10457465_0.log 2>/dev/null'))
ssh.close()
