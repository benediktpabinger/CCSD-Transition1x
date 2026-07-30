#!/bin/bash
#SBATCH --job-name=casscf_optts_next_retry2
#SBATCH --partition=xeon24el8_week
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120GB
#SBATCH --array=0-1
#SBATCH --output=/home/energy/s242862/logs/casscf_optts_next_retry2_%A_%a.log

# Second retry round for next-HIGH reactions that failed/stalled in job 10562969:
#   rxn4522  — failed with the same "Nuclear gradients not converged" Scanner
#              error as the first retry batch (norm(grad)=0.0011, very close).
#   rxn10054 — not failed, but stuck oscillating: 23,113 macro-iterations
#              logged (vs 60-300 typical) with dE flipping sign between steps —
#              the default conv_tol=1e-7 is too tight for this active space.
#              Killed manually after 3d10h with no sign of converging.
# Same fix as retry round 1 (job 10564837, which fixed rxn0896/rxn1283/rxn10005
# cleanly): loosen conv_tol 1e-7 -> 1e-6.

set -e

REACTIONS=(rxn4522 rxn10054)
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
    --no-prune \
    --conv-tol 1e-6
