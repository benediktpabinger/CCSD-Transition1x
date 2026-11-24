#!/bin/bash
#SBATCH --job-name=fod_model_ts
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=120G
#SBATCH --output=/home/energy/bepab/logs/fod_model_ts_%A_%a.out
#SBATCH --error=/home/energy/bepab/logs/fod_model_ts_%A_%a.err

# N_FOD at the freezing-string transition state of every model (pipeline/fod_model_ts.py, D45).
# One whole xeon24el9 node per array task, 6 PySCF processes with 4 threads each, as in
# job_fod_all.sh. Task i works on structures i*BLOCK .. (i+1)*BLOCK-1 of the sorted list.
#
#   sbatch --array=0-6 pipeline/job_fod_model_ts.sh      (7 x 250 >= 1662)

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0

BLOCK=${BLOCK:-250}
export OMP_NUM_THREADS=4
export PYSCF_MAX_MEMORY=15000      # MB per process, 6 x 15 GB < 120 GB
mkdir -p $HOME/logs

cd $HOME/Transition1x
python3 pipeline/fod_model_ts.py --start $(( SLURM_ARRAY_TASK_ID * BLOCK )) --count $BLOCK --workers 6
