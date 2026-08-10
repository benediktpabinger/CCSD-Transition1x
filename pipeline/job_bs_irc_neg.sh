#!/bin/bash
#SBATCH --job-name=bsircneg
#SBATCH --partition=xeon24el8
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_irc/slurm_neg_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_irc/slurm_neg_%A_%a.err

# Negative control for the endpoint test.
#
# The test has returned "connects reactant and product" four times out of four
# so far. A check that never fails is not a check. These three saddles are known
# to be wrong -- their imaginary mode does not move the reactive bonds -- so if
# the test discriminates at all, it has to fail here.
#
#   rxn1320  mode fraction 0.00, bond rates +0.001 and -0.000
#   rxn4518  mode fraction 0.03 at 89 cm-1
#   rxn5691  bond rates -0.014 and -0.009 at 102 cm-1
#
# Expected: both displacement directions relax into the same minimum, or into
# something that is neither the reactant nor the product. Should they instead
# come out connecting the two correctly, the test carries no information and
# the four positive results have to be discarded with it.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn1320 rxn4518 rxn5691)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_irc
echo "Task $SLURM_ARRAY_TASK_ID: $RXN (Negativkontrolle)  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_irc.py $RXN
RC=$?
echo "rc=$RC $(date)"
exit $RC
