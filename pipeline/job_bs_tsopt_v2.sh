#!/bin/bash
#SBATCH --job-name=bstsv2
#SBATCH --partition=xeon24el8
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/home/energy/s242862/bs_tsopt_v2/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_tsopt_v2/slurm_%A_%a.err

# BS-UKS TS optimisation for the 10 reactions still missing a confirmed
# broken-symmetry transition state, of the 19 whose RKS reference TS is
# externally unstable.
#
#   never attempted (below the old 0.3 A model-error criterion):
#     rxn8832 rxn6196 rxn7060 rxn7957 rxn5690
#   attempted and failed:
#     rxn4113  COLLAPSED before the first step (BS was weak: dE -8.4 meV, S2 0.14)
#     rxn4522  ended at S2 = 0.000
#     rxn8885  ended at S2 = 0.153
#     rxn3107  ended at S2 = 0.179
#     rxn1283  S2 = 0.977 but geometry not converged in 150 steps
#
# v2 differs from the first batch in two ways:
#   - branch-jump recovery: when <S^2> drops below 0.3 the step is redone, first
#     by reseeding from the last good density, then by going back to RKS and
#     following the external instability again
#   - maxsteps 150 -> 300 (rxn1283 needed more than 150)
#
# Writes to bs_tsopt_v2/, so the first batch's results stay intact.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn8832 rxn6196 rxn7060 rxn7957 rxn5690 rxn4113 rxn4522 rxn8885 rxn3107 rxn1283)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_tsopt_v2
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_tsopt_v2.py $RXN
RC=$?
echo "rc=$RC $(date)"
exit $RC
