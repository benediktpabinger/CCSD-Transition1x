#!/bin/bash
#SBATCH --job-name=val_delta_flip
#SBATCH --partition=xeon24el8
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --array=0-50
#SBATCH --output=/home/energy/s242862/logs/val_delta_flip_%A_%a.log

# wB97M-V/def2-TZVP SPs on 50 uniformly sampled T1x geometries per reaction.
# 51 failed-NEB val reactions, one node per reaction, 50 SPs run serially.

set -e

module load Python/3.13.5-GCCcore-14.3.0
module load ORCA/5.0.4-gompi-2023a
mkdir -p /home/energy/s242862/logs

RXNLIST=/home/energy/s242862/ccsd_dataset/val_failed.txt
RXN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${RXNLIST})

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python3 /home/energy/s242862/pipeline/val_delta_sp_flip.py ${RXN} --n-threads 8
