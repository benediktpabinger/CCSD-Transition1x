#!/bin/bash
#SBATCH --job-name=mr_ccsdt
#SBATCH --partition=xeon24el8
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120GB
#SBATCH --array=0-9
#SBATCH --output=/home/energy/s242862/logs/mr_ccsdt_%A_%a.log

# CCSD(T)/def2-TZVP on R, TS, P for the 10 MR benchmark reactions.
# One node per reaction (3 serial calculations: R, TS, P).

set -e

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs

REACTIONS=(rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn7945 rxn7937 rxn6196 rxn0346 rxn1150)
RXN=${REACTIONS[${SLURM_ARRAY_TASK_ID}]}

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export BLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export PYSCF_MAX_MEMORY=100000

python3 /home/energy/s242862/pipeline/mr_benchmark_ccsdt.py ${RXN} --n-threads 12
