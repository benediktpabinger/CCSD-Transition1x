#!/bin/bash
#SBATCH --job-name=freqmod
#SBATCH --partition=xeon24el8
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/freq_at_model/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/freq_at_model/slurm_%A_%a.err

# The three reactions where it is still undecided whether the models are wrong.
# Their geometries are nearly stationary -- gradients comparable to the
# reference -- but sit 0.37 to 0.60 A from the saddle we found. If a second
# stationary point is there, this says whether it is a transition state, and
# the energy comparison says which of the two the reaction actually uses.
#
#   rxn1147   all three model geometries are externally STABLE and stationary
#             to 0.050-0.077 eV/A -- the strongest case for the models being
#             right rather than us. Runs as RKS, since no broken solution exists
#             there.
#   rxn7957   externally unstable, BS gradients 0.109-0.137 eV/A
#   rxn7949   externally unstable, BS gradients 0.051-0.248 eV/A
#
# Which surface is used is decided per structure by the stability analysis, not
# assumed.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn1147 rxn1147 rxn1147 rxn7957 rxn7957 rxn7957 rxn7949 rxn7949 rxn7949)
MODS=(UMA-S   UMA-M   eSEN    UMA-S   UMA-M   eSEN    UMA-S   UMA-M   eSEN)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
MOD=${MODS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/freq_at_model
echo "Task $SLURM_ARRAY_TASK_ID: $RXN / $MOD  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/freq_at_model.py $RXN $MOD
RC=$?
echo "rc=$RC $(date)"
exit $RC
