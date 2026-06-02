import paramiko, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'], password=os.environ['SSH_PASS'])

def run(cmd):
    _, out, _ = ssh.exec_command(cmd, get_pty=True)
    print(out.read().decode('utf-8', errors='replace').strip())

prefix = 'module load Python/3.13.5-GCCcore-14.3.0 && python3'

# Check signatures of load_predict_unit and get_predict_unit
run(prefix + ' -c "import inspect; from fairchem.core import pretrained_mlip; print(inspect.signature(pretrained_mlip.load_predict_unit))" 2>&1')
run(prefix + ' -c "import inspect; from fairchem.core import pretrained_mlip; print(inspect.signature(pretrained_mlip.get_predict_unit))" 2>&1')

# Try loading from checkpoint
run(prefix + ' -c "from fairchem.core import pretrained_mlip, FAIRChemCalculator; pu = pretrained_mlip.load_predict_unit(\'/home/energy/s242862/checkpoints/esen_sm_conserving_all.pt\'); calc = FAIRChemCalculator(pu); print(\'SUCCESS\')" 2>&1')

ssh.close()
