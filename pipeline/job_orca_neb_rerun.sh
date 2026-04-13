#!/bin/bash
#SBATCH --job-name=orca_neb_rerun
#SBATCH --partition=xeon24el8
#SBATCH --time=2-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/energy/s242862/logs/orca_neb_rerun_%A_%a.log
#SBATCH --array=0-286

# Rerun unconverged test reactions:
#   < 70 steps: resume from last neb.db images
#   >= 70 steps: restart with linear interpolation from relaxed endpoints
# Usage: sbatch /home/energy/s242862/pipeline/job_orca_neb_rerun.sh

module load Python/3.13.5-GCCcore-14.3.0
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0

export OMP_NUM_THREADS=1

REACTION=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /home/energy/s242862/ccsd_dataset/test_reactions.txt)
OUTPUT=/home/energy/s242862/orca_neb_results/${REACTION}

echo "Task ${SLURM_ARRAY_TASK_ID}: ${REACTION}"

# Skip if already converged
if [ -f "${OUTPUT}/converged" ]; then
    echo "Already converged, skipping."
    exit 0
fi

python -u /home/energy/s242862/pipeline/orca_neb_rerun.py \
    --reaction   ${REACTION} \
    --output     ${OUTPUT} \
    --orca-cmd   /home/modules/software/ORCA/5.0.4-gompi-2023a/bin/orca \
    --n-threads  8 \
    --neb-fmax   0.15 \
    --cineb-fmax 0.05 \
    --steps      500 \
    --step-threshold 70
