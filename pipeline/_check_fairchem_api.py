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
    return out.read().decode('utf-8', errors='replace').strip()

print(run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "import fairchem.core; print(dir(fairchem.core))"'))
print()
print(run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "import fairchem.core; print(fairchem.core.__version__)"'))
print()
print(run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "from fairchem.core.calculate import FAIRChemCalculator; print(\'FAIRChemCalculator ok\')" 2>&1'))
print(run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "from fairchem.core import FAIRChemCalculator; print(\'FAIRChemCalculator ok\')" 2>&1'))

ssh.close()
