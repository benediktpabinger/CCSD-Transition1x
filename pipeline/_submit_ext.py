import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd)
    return out.read().decode('utf-8', errors='replace').strip()

sftp = ssh.open_sftp()
sftp.put(r'c:/Transition 1X/Transition 1x/Transition1x/pipeline/job_mr_ccsdt_ext.sh',
         '/home/energy/s242862/pipeline/job_mr_ccsdt_ext.sh')
sftp.put(r'c:/Transition 1X/Transition 1x/Transition1x/pipeline/job_mr_nevpt2_ext.sh',
         '/home/energy/s242862/pipeline/job_mr_nevpt2_ext.sh')
sftp.close()
print('Uploaded job scripts')

print(run('sbatch /home/energy/s242862/pipeline/job_mr_ccsdt_ext.sh'))
print(run('sbatch /home/energy/s242862/pipeline/job_mr_nevpt2_ext.sh'))
print()
print(run('squeue -u s242862 -h'))
ssh.close()
