#!/bin/bash
#SBATCH --job-name=casscf_optts_next
#SBATCH --partition=xeon24el8_week
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120GB
#SBATCH --array=0-15
#SBATCH --output=/home/energy/s242862/logs/casscf_optts_next_%A_%a.log

# CASSCF(AVAS) OptTS + NEVPT2/def2-TZVP for the next-HIGH MR benchmark
# extension (FOD ranks 11-26, 16 reactions). rxn0896 (rank 11) already has
# CCSD(T)/NEVPT2-SP/all 5 NEB methods from the Mid-MR group — only OptTS is
# new for it. Ranks 12-26 are entirely new.
#
# Settings (--avas-threshold 0.4 --no-prune) chosen directly from what
# converged for the original top-10 after iteration (default threshold
# failed on 6/10; threshold=0.4 with pruning collapsed 8/10 to degenerate
# CAS(2,2) spaces; --no-prune fixed those). Starting here to skip repeating
# the same two failed rounds.

set -e

REACTIONS=(
    rxn0896 rxn4518 rxn3107 rxn8837 rxn7060
    rxn5691 rxn1283 rxn8827 rxn4522 rxn7936
    rxn1147 rxn0894 rxn0101 rxn10005 rxn10054
    rxn7957
)
RXN=${REACTIONS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/logs

module load Python/3.13.5-GCCcore-14.3.0

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export BLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export PYSCF_MAX_MEMORY=100000

python3 /home/energy/s242862/pipeline/mr_casscf_optts.py ${RXN} \
    --n-threads 12 \
    --avas-threshold 0.4 \
    --no-prune
