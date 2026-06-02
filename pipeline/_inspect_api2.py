import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    print(out.read().decode('utf-8', errors='replace').strip())

prefix = 'module load Python/3.13.5-GCCcore-14.3.0 && python3'

run(prefix + ' -c "from fairchem.core import pretrained_mlip; print(dir(pretrained_mlip))" 2>&1')
run(prefix + ' -c "import inspect; from fairchem.core.calculate.mlip_calculators import MLIPPredictUnit; print(inspect.signature(MLIPPredictUnit.__init__))" 2>&1')
run(prefix + ' -c "from fairchem.core.calculate.mlip_calculators import MLIPPredictUnit; help(MLIPPredictUnit.__init__)" 2>&1')

ssh.close()
