import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd)
    return out.read().decode('utf-8', errors='replace').strip()
print(run('sinfo --summarize 2>/dev/null'))
print()
print(run('sinfo -o "%P %G %D %t" 2>/dev/null | grep -i gpu | head -20'))
ssh.close()
