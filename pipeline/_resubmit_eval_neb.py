import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', password='Butterbrot9797')
def run(cmd):
    _, out, _ = ssh.exec_command(cmd)
    return out.read().decode('utf-8', errors='replace').strip()

# Cancel pending job
print(run('scancel 10391794'))
print('Cancelled 10391794')

# Upload updated script
sftp = ssh.open_sftp()
sftp.put(r'c:/Transition 1X/Transition 1x/Transition1x/pipeline/delta/job_eval_delta_neb.sh',
         '/home/energy/s242862/pipeline/delta/job_eval_delta_neb.sh')
sftp.close()

# Resubmit
print(run('sbatch /home/energy/s242862/pipeline/delta/job_eval_delta_neb.sh'))
print(run('squeue -u s242862 -h'))
ssh.close()
