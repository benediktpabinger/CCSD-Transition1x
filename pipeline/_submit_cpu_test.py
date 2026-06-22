import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])
sftp = ssh.open_sftp()
sftp.put(f'{LOCAL}\\mace_delta_neb.py', f'{REMOTE}/mace_delta_neb.py')
sftp.put(f'{LOCAL}\\job_mace_delta_cpu_test.sh', f'{REMOTE}/job_mace_delta_cpu_test.sh')
sftp.close()
print('uploaded')

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    print(out.read().decode('utf-8', errors='replace').strip())

run('chmod +x /home/energy/s242862/pipeline/job_mace_delta_cpu_test.sh')
run('rm -rf /home/energy/s242862/mace_delta_neb_results/rxn4063_cpu')
run('sbatch /home/energy/s242862/pipeline/job_mace_delta_cpu_test.sh')
print()
run('squeue -u s242862')
ssh.close()
