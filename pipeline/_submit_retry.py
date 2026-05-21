import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd)
    return out.read().decode('utf-8', errors='replace').strip()

sftp = ssh.open_sftp()
sftp.put(r'c:/Transition 1X/Transition 1x/Transition1x/pipeline/mr_benchmark_nevpt2.py',
         '/home/energy/s242862/pipeline/mr_benchmark_nevpt2.py')
sftp.put(r'c:/Transition 1X/Transition 1x/Transition1x/pipeline/job_mr_nevpt2_retry.sh',
         '/home/energy/s242862/pipeline/job_mr_nevpt2_retry.sh')
sftp.close()
print('Uploaded')

print(run('sbatch /home/energy/s242862/pipeline/job_mr_nevpt2_retry.sh'))
print()
print(run('squeue -u s242862 -h'))
ssh.close()
