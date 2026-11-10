#!/bin/bash
#SBATCH --job-name=d37forces
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/home/energy/bepab/logs/d37forces_%j.out
#SBATCH --error=/home/energy/bepab/logs/d37forces_%j.err

# Model forces at the freezing-string structures of all four models (D37), one process per
# model with 6 threads each; files that exist are skipped. Then the table.
#
#   sbatch pipeline/job_d37_forces.sh

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
cd $HOME/Transition1x
for M in esen uma-s uma-m mace-omol; do
  if [ "$M" = mace-omol ]; then
    PYTHONPATH=$HOME/venvs/e3nn044${PYTHONPATH:+:$PYTHONPATH} python3 pipeline/d37_table.py --model $M &
  else
    python3 pipeline/d37_table.py --model $M &
  fi
done
wait
python3 pipeline/d37_table.py --table
echo "ENDE $(date)"
