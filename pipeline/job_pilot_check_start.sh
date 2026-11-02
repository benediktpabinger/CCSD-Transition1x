#!/bin/bash
#SBATCH --job-name=pilotstart
#SBATCH --partition=xeon24el9_test
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/energy/bepab/logs/pilotstart_%j.out
#SBATCH --error=/home/energy/bepab/logs/pilotstart_%j.err

# Submit with --dependency on the pilot arrays: builds the final case list, then one
# node per model for pipeline/pilot_check.py.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
cd $HOME/Transition1x
python3 pipeline/pilot_table.py
python3 pipeline/pilot_check.py --list
for M in uma-s uma-m esen mace-omol; do
  sbatch --export=ALL,MODEL=$M pipeline/job_pilot_check.sh
done
