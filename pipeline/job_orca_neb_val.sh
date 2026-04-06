#!/bin/bash
#SBATCH --job-name=orca_neb_val
#SBATCH --partition=xeon24el8
#SBATCH --time=2-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/energy/s242862/~/logs/orca_neb_val_%A_%a.log
#SBATCH --array=0-224

# ωB97M-V/def2-TZVP NEB for all 225 val reactions
# Usage: sbatch /home/energy/s242862/pipeline/job_orca_neb_val.sh

module load Python/3.13.5-GCCcore-14.3.0
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0

export OMP_NUM_THREADS=1

REACTION=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /home/energy/s242862/ccsd_dataset/val_reactions.txt)
OUTPUT=/home/energy/s242862/orca_neb_val_results/${REACTION}

echo "Task ${SLURM_ARRAY_TASK_ID}: ${REACTION}"
echo "Output: ${OUTPUT}"

# Skip if already converged
if [ -f "${OUTPUT}/converged" ]; then
    echo "Already converged, skipping."
    exit 0
fi

python -u /home/energy/s242862/pipeline/orca_neb.py \
    --h5file     /home/energy/s242862/data/Transition1x.h5 \
    --reaction   ${REACTION} \
    --split      val \
    --output     ${OUTPUT} \
    --orca-cmd   /home/modules/software/ORCA/5.0.4-gompi-2023a/bin/orca \
    --n-threads  8 \
    --neb-fmax   0.15 \
    --cineb-fmax 0.05 \
    --steps      500
