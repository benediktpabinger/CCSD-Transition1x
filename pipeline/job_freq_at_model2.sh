#!/bin/bash
#SBATCH --job-name=freqmod2
#SBATCH --partition=xeon24el8
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/freq_at_model/slurm2_%A_%a.out
#SBATCH --error=/home/energy/s242862/freq_at_model/slurm2_%A_%a.err

# The two reactions where the models sit furthest below our saddle and nothing
# is decided:
#
#   rxn4522  all three models 1.84 eV below, gradients around 0.08, and our own
#            optimisation never converged -- there is no confirmed saddle of
#            ours to compare against
#   rxn5691  eSEN 163 meV below at gradient 0.068, while our saddle failed the
#            mode test
#
# A near-stationary point below a saddle is either a lower saddle, in which case
# it is the relevant one, or a minimum downhill of the transition state, in
# which case it is no candidate. Only the frequency separates them.
#
# rxn1147 showed why the frequency alone is not enough: all three models there
# are genuine saddles 231 meV lower, but their imaginary mode barely touches the
# reactive coordinate (-0.06 against -0.94) and the forming C-O bond is already
# at 1.50 A, a normal single bond. They sit past the transition state, and their
# saddle belongs to some other motion. The mode comparison has to follow.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn4522 rxn4522 rxn4522 rxn5691 rxn5691 rxn5691)
MODS=(UMA-S   UMA-M   eSEN    UMA-S   UMA-M   eSEN)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
MOD=${MODS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/freq_at_model
echo "Task $SLURM_ARRAY_TASK_ID: $RXN / $MOD  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/freq_at_model.py $RXN $MOD
RC=$?
echo "rc=$RC $(date)"
exit $RC
