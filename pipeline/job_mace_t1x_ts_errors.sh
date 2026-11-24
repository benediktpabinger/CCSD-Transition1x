#!/bin/bash
#SBATCH --job-name=mace_t1x_ts
#SBATCH --partition=h200el9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/energy/bepab/logs/mace_t1x_ts_%j.out
#SBATCH --error=/home/energy/bepab/logs/mace_t1x_ts_%j.err

# Thesis MACE model against its own Transition1x labels at every Transition1x transition state
# (pipeline/mace_t1x_ts_errors.py), on one H200 GPU as the thesis evaluation.
#
#   sbatch pipeline/job_mace_t1x_ts_errors.sh

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
export DEVICE=cuda
mkdir -p $HOME/logs

cd $HOME/Transition1x
python3 pipeline/mace_t1x_ts_errors.py
