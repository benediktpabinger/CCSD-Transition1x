import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    return out.read().decode('utf-8', errors='replace').strip()
print(run('squeue -u s242862 -h'))
print()
success = run('grep -rl "FINAL SINGLE POINT ENERGY" /home/energy/s242862/mr_benchmark/orca_engrad | wc -l')
total = run('find /home/energy/s242862/mr_benchmark/orca_engrad -name sp.out | wc -l')
print(f'EnGrad: {success}/{total} successful')
ssh.close()
