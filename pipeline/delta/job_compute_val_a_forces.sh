#!/bin/bash
#SBATCH --job-name=val_a_forces
#SBATCH --partition=xeon24el8
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --array=0-173
#SBATCH --output=/home/energy/s242862/logs/val_a_forces_%A_%a.log

set -e

module load Python/3.13.5-GCCcore-14.3.0
module load ORCA/5.0.4-gompi-2023a
mkdir -p /home/energy/s242862/logs

export TMPDIR=/tmp

RXNLIST=/home/energy/s242862/ccsd_dataset/val_converged.txt
RXN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${RXNLIST})

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN}"

python3 /home/energy/s242862/pipeline/delta/compute_val_a_forces.py ${RXN} --nprocs 8
