#!/bin/bash
#SBATCH --job-name=tsfrommod
#SBATCH --partition=xeon24el8
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/tsopt_from_model/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/tsopt_from_model/slurm_%A_%a.err

# Can the model geometry replace the path search?
#
# One transition-state optimisation per reaction, started at the model's own
# predicted geometry instead of at the reference. If it reaches the correct
# saddle, the expensive NEB can be skipped and the model is useful even where
# its geometry is not exact. If it goes somewhere else, the prediction is
# misleading rather than merely imprecise -- worse for this purpose than a large
# but harmless error.
#
# The ten reactions whose broken-symmetry saddle survived both the frequency and
# the imaginary-mode test. UMA-M is the representative model; rxn0894 uses UMA-S
# because RKS does not converge at the UMA-M geometry there.
#
# rxn1147 and rxn7060 have externally stable model geometries, so those run as
# plain RKS searches -- decided by the stability analysis, not assumed.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn8837 rxn8832 rxn1147 rxn7957 rxn7949 rxn0894 rxn0346 rxn3107 rxn8827 rxn7060)
MODS=(UMA-M   UMA-M   UMA-M   UMA-M   UMA-M   UMA-S   UMA-M   UMA-M   UMA-M   UMA-M)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
MOD=${MODS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/tsopt_from_model
echo "Task $SLURM_ARRAY_TASK_ID: $RXN / $MOD  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/tsopt_from_model.py $RXN $MOD
RC=$?
echo "rc=$RC $(date)"
exit $RC
