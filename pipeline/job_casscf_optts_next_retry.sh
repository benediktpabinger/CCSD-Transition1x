#!/bin/bash
#SBATCH --job-name=casscf_optts_next_retry
#SBATCH --partition=xeon24el8_week
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120GB
#SBATCH --array=0-2
#SBATCH --output=/home/energy/s242862/logs/casscf_optts_next_retry_%A_%a.log

# Retry for the 3 next-HIGH reactions that failed in job 10562969:
# rxn0896, rxn1283, rxn10005 — all failed with the same pyscf error:
#   "Nuclear gradients of <CASSCF_Scanner> not converged"
# raised inside geometric_solver during the OptTS eigenvector-following walk
# (rxn0896 failed at OptTS cycle 3, rxn1283 at cycle 68, rxn10005 at cycle 63 —
# not an active-space problem, just the Scanner's conv_tol_grad being too tight
# at one particular geometry step). Loosened conv_tol 1e-7 -> 1e-6 here
# (conv_tol_grad sqrt(1e-7)=3.2e-4 -> sqrt(1e-6)=1.0e-3).

set -e

REACTIONS=(rxn0896 rxn1283 rxn10005)
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
