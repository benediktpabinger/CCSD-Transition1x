#!/bin/bash
#SBATCH --job-name=idppstart
#SBATCH --partition=xeon24el9_test
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/energy/bepab/logs/idppstart_%j.out
#SBATCH --error=/home/energy/bepab/logs/idppstart_%j.err

# Submit with --dependency on the interpolated-start NEB arrays (D33): lists every
# converged run of the 510 reactions (pipeline/check_v2_cases.py), then one check
# job per model with the D35 check. Structures already checked are skipped.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
cd $HOME/Transition1x
python3 pipeline/check_v2_cases.py
for M in uma-s uma-m esen mace-omol; do
  sbatch --job-name=chk-idpp-$M --export=ALL,MODEL=$M,OUTDIR=$HOME/neb_pilot/check_idpp_v2,ROOT=$HOME/neb_pilot pipeline/job_pilot_check.sh
done
