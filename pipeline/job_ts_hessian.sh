#!/bin/bash
#SBATCH --job-name=tshess
#SBATCH --partition=xeon24el9_test
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=00:10:00
#SBATCH --output=/home/energy/bepab/logs/tshess_%j.out
#SBATCH --error=/home/energy/bepab/logs/tshess_%j.err
source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
[ "$MODEL" = "mace-omol" ] && export PYTHONPATH=$HOME/venvs/e3nn044
cd $HOME/Transition1x
python3 pipeline/ts_hessian_scan.py --model $MODEL --device cpu
