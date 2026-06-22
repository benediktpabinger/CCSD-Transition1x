import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])
def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    print(out.read().decode('utf-8', errors='replace').strip())
run('squeue -u s242862')
print()
run('tail -5 /home/energy/s242862/mace_delta_neb_results/rxn4063_cpu/neb.log 2>/dev/null || echo no neb log yet')
print()
run('test -f /home/energy/s242862/mace_delta_neb_results/rxn4063_cpu/converged && echo CONVERGED || echo not yet')
ssh.close()
