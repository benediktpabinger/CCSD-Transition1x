#!/bin/bash
#SBATCH --job-name=ccsdt_sp_tz
#SBATCH --partition=xeon24el8
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/home/energy/s242862/logs/ccsdt_sp_tz_%A_%a.log
#SBATCH --array=0-9

# CCSD(T)/cc-pVTZ single-points on converged rxn0103 TZ NEB images.
# 10 parallel tasks (one per image), gather step runs after all finish.
# Usage: sbatch ~/pipeline/job_ccsdt_sp_tz.sh

module load Python/3.13.5-GCCcore-14.3.0

export OMP_NUM_THREADS=8

NEB_DIR=~/ccsd_tz_results/rxn0103

python ~/pipeline/ccsd_t_singlepoints.py \
    --neb-dir    ${NEB_DIR} \
    --basis      cc-pVTZ \
    --n-images   10 \
    --n-threads  8 \
    --image-idx  ${SLURM_ARRAY_TASK_ID}
