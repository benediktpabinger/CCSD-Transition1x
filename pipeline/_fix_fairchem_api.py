import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HOST = 'slid.fysik.dtu.dk'
USER = os.environ.get('SSH_USER', 's242862')
PASS = os.environ.get('SSH_PASS')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    o = out.read().decode('utf-8', errors='replace').strip()
    if o: print(o)
    return o

print('=== cancelling ===')
run('scancel 10438564')

print()
print('=== FAIRChemCalculator signature ===')
run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "from fairchem.core import FAIRChemCalculator; help(FAIRChemCalculator.__init__)" 2>&1')

print()
print('=== pretrained_mlip list ===')
run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "from fairchem.core import pretrained_mlip; print(dir(pretrained_mlip))" 2>&1')

print()
print('=== available pretrained models ===')
run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "from fairchem.core.pretrained_mlip import get_available_pretrained_models; print(get_available_pretrained_models())" 2>&1')

ssh.close()
