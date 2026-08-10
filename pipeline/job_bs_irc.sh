#!/bin/bash
#SBATCH --job-name=bsirc
#SBATCH --partition=xeon24el8
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_irc/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_irc/slurm_%A_%a.err

# Does each saddle point connect the right reactant and product?
#
# Displace along the imaginary mode in both directions, relax, compare the two
# endpoints against the known reactant and product. Not a strict IRC -- PySCF
# has none -- but it answers the question the frequency calculation cannot,
# which is whether the saddle sits on this reaction's path at all.
#
# Six reactions currently rated "wahrscheinlich": frequency and imaginary mode
# are sound but no independent check exists. These are what the run is for.
#   rxn7949 rxn0894 rxn0346 rxn3107 rxn8827 rxn7060
#
# Four rated "bestaetigt" are included as a control on the procedure itself:
# an independent NEB already found the same structures, so these should come
# out clean. If they do not, the method is at fault rather than the structures.
#   rxn8837 rxn8832 rxn1147 rxn7957
#
# The three failing the mode test (rxn1320, rxn4518, rxn5691) are deliberately
# left out -- their saddles belong to a different motion, so tracing where they
# lead would answer nothing.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn7949 rxn0894 rxn0346 rxn3107 rxn8827 rxn7060 \
      rxn8837 rxn8832 rxn1147 rxn7957)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_irc
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_irc.py $RXN
RC=$?
echo "rc=$RC $(date)"
exit $RC
