#!/bin/bash
#SBATCH --job-name=fod_all
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=120G
#SBATCH --output=/home/energy/bepab/logs/fod_all_%A_%a.out
#SBATCH --error=/home/energy/bepab/logs/fod_all_%A_%a.err

# N_FOD an allen 10073 gespeicherten Transition1x-Uebergangszustaenden
# (pipeline/fod_all_ts.py). Ein ganzer xeon24el9-Knoten je Array-Task, darauf
# 6 PySCF-Prozesse mit je 4 Threads, wie in job_fod_screen.sh (Thesis).
# Task i rechnet die Reaktionen i*BLOCK .. (i+1)*BLOCK-1 der sortierten Liste.
#
#   sbatch --array=0-40%18 pipeline/job_fod_all.sh      (41 x 250 >= 10073)

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0

BLOCK=${BLOCK:-250}
export OMP_NUM_THREADS=4
export PYSCF_MAX_MEMORY=15000      # MB je Prozess, 6 x 15 GB < 120 GB
mkdir -p $HOME/logs

cd $HOME/Transition1x
python3 pipeline/fod_all_ts.py --start $(( SLURM_ARRAY_TASK_ID * BLOCK )) --count $BLOCK --workers 6
