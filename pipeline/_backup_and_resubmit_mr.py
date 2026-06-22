import paramiko, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username='s242862', key_filename=r'C:\Users\PabingerBenedikt\.ssh\dtu_key', timeout=30)

LOCAL  = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
REMOTE = '/home/energy/s242862/pipeline'

# Back up existing successful results before rerunning with the new pipeline
backup_cmd = (
    'cd /home/energy/s242862/nevpt2_optts_results && '
    'for rxn in rxn7949 rxn6196 rxn0346 rxn4113; do '
    '  if [ -d ${rxn}_avas ]; then cp -r ${rxn}_avas ${rxn}_avas_backup_preprune; '
    '  echo "backed up ${rxn}_avas"; fi; '
    'done'
)
_, out, err = ssh.exec_command(backup_cmd)
print(out.read().decode())
print(err.read().decode())

sftp = ssh.open_sftp()
sftp.put(f'{LOCAL}\\mr_casscf_optts.py', f'{REMOTE}/mr_casscf_optts.py')
sftp.close()

_, out, err = ssh.exec_command('sbatch /home/energy/s242862/pipeline/job_casscf_optts_mr_retry.sh')
print(out.read().decode())
print(err.read().decode())

ssh.close()
