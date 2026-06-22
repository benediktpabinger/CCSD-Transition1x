#!/bin/bash
#SBATCH --job-name=casscf_noprune
#SBATCH --partition=xeon24el8_week
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120GB
#SBATCH --output=/home/energy/s242862/logs/casscf_noprune_%A_%a.log

# Rerun 6 High-MR reactions without active-space pruning.
# These converged in job 10515621 but to CAS(2,2) spaces (too small).
# --no-prune keeps the full AVAS-selected active space.
# mc1step->mc2step fallback still active for convergence robustness.
# Array indices 0-5 map to the 6 reactions below (not the full 10).

set -e

REACTIONS=(rxn7949 rxn8832 rxn1320 rxn8885 rxn7945 rxn7937)
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
