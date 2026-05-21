#!/bin/bash
#SBATCH --job-name=mr_nevpt2_retry
#SBATCH --partition=xeon24el8
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120GB
#SBATCH --array=0-8
#SBATCH --output=/home/energy/s242862/logs/mr_nevpt2_retry_%A_%a.log

# Retry NEVPT2/AVAS for reactions that failed CASSCF convergence.
# Uses max_cycle=2000 instead of 1000.

set -e

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs

REACTIONS=(rxn4498 rxn1061 rxn4003 rxn4060 rxn1962 rxn1154 rxn5690 rxn4519 rxn4500)
RXN=${REACTIONS[${SLURM_ARRAY_TASK_ID}]}

OUT=/home/energy/s242862/nevpt2_results/${RXN}_pyscf_avas/nevpt2_results.json
if [ -f "$OUT" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN} already done, skipping."
    exit 0
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export BLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export PYSCF_MAX_MEMORY=100000

python3 /home/energy/s242862/pipeline/mr_benchmark_nevpt2.py ${RXN} \
    --n-threads 12 --max-cycle 2000
