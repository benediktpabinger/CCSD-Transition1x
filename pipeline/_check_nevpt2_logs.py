import paramiko, sys

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd)
    return out.read().decode('utf-8', errors='replace').strip()

# Get logs for the ext NEVPT2 job array (10389079)
logs = run('ls /home/energy/s242862/logs/ | grep mr_nevpt2_ext').splitlines()
logs = sorted(logs)

for log in logs:
    tail = run(f'tail -4 /home/energy/s242862/logs/{log}')
    # get task id from filename
    print(f'--- {log} ---')
    print(tail)
    print()

ssh.close()
