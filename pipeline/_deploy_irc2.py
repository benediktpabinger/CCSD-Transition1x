"""Upload the IRC script, check its inputs exist, and run a two-step smoke test.

The real runs are four 24-hour jobs. A bug in the projector or the density
chaining would only show up hours in, so the same code path runs first with
IRC_MAX=2 -- enough to exercise the kick, one gradient, one steepest-descent
step and the JSON writing.
"""
import os
import sys

import paramiko

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOCAL = r'c:\Transition 1X\Transition 1x\Transition1x\pipeline'
H = '/home/energy/s242862'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('slid.fysik.dtu.dk', username=os.environ['SSH_USER'],
            password=os.environ['SSH_PASS'])


def run(cmd):
    _, out, err = ssh.exec_command(cmd, get_pty=True)
    t = out.read().decode('utf-8', errors='replace').strip()
    e = err.read().decode('utf-8', errors='replace').strip()
    return t + (('\n' + e) if e else '')


sftp = ssh.open_sftp()
sftp.put(f'{LOCAL}\\bs_irc2.py', f'{H}/bs_irc2.py')
sftp.put(f'{LOCAL}\\job_bs_irc2.sh', f'{H}/job_bs_irc2.sh')
sftp.close()
run(f'sed -i "s/\\r$//" {H}/job_bs_irc2.sh; chmod +x {H}/job_bs_irc2.sh')
print('uploaded')

print('\n=== inputs ===')
check = f'''
for p in \\
  {H}/freq_at_model/rxn1147_UMA-S/hessian.npy \\
  {H}/freq_at_model/rxn7957_UMA-M/hessian.npy \\
  {H}/uma_neb_results/rxn1147/transition_state.xyz \\
  {H}/uma_m_neb_results/rxn7957/transition_state.xyz \\
  {H}/orca_neb_results/rxn1147/reactant.xyz \\
  {H}/orca_neb_results/rxn7957/reactant.xyz ; do
  if [ -f "$p" ]; then echo "ok   $p"; else echo "MISS $p"; fi
done
for r in rxn1147 rxn7957; do
  echo "-- $r ours"
  ls {H}/bs_tsopt_*/$r/*.xyz 2>/dev/null | head -4
  ls {H}/bs_freq*/$r/hessian.npy 2>/dev/null
done
'''
print(run(check))

print('\n=== smoke test (2 steps, rxn1147 ours) ===')
smoke = f'''cat > {H}/job_irc_smoke.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=ircsmoke
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --output={H}/bs_irc2/smoke.out
#SBATCH --error={H}/bs_irc2/smoke.err
source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE
export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000
export IRC_RXN=rxn1147
export IRC_SRC=ours
export IRC_MAX=2
export IRC_OUT={H}/bs_irc2_smoke
mkdir -p $IRC_OUT
python3 {H}/bs_irc2.py
echo "rc=$?"
EOF
mkdir -p {H}/bs_irc2
sbatch {H}/job_irc_smoke.sh'''
print(run(smoke))
ssh.close()
