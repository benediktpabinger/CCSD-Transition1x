#!/bin/bash
#SBATCH --job-name=ccsdt_sel
#SBATCH --partition=xeon24el8
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=/home/energy/s242862/~/logs/ccsdt_sel_%A_%a.log
#SBATCH --array=0-9

# CCSD(T)/cc-pVDZ single-points on 10 CURATOR-selected configs.
# 10 parallel tasks (one per config).
# After all finish, run gather step:
#   python ~/pipeline/ccsd_sp_selected.py --round-dir ~/curator_results/round3 --gather
# Usage: sbatch ~/pipeline/job_ccsd_sp_selected.sh

module load Python/3.13.5-GCCcore-14.3.0

export OMP_NUM_THREADS=8

python -u ~/pipeline/ccsd_sp_selected.py \
    --round-dir ~/curator_results/round3 \
    --basis     cc-pVDZ \
    --n-threads 8 \
    --image-idx ${SLURM_ARRAY_TASK_ID}
