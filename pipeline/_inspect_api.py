import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    print(out.read().decode('utf-8', errors='replace').strip())

run('scancel 10438564 2>/dev/null; echo cancelled')
run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "import inspect; from fairchem.core import FAIRChemCalculator; print(inspect.signature(FAIRChemCalculator.__init__))" 2>&1')
run('module load Python/3.13.5-GCCcore-14.3.0 && python3 -c "from fairchem.core.pretrained_mlip import get_available_pretrained_models; print(get_available_pretrained_models())" 2>&1')

ssh.close()
