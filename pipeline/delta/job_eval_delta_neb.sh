#!/bin/bash
#SBATCH --job-name=eval_delta_neb
#SBATCH --partition=h200
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/eval_delta_neb_%j.log

module load Python/3.13.5-GCCcore-14.3.0

python3 -u /home/energy/s242862/pipeline/delta/eval_delta_neb.py --n-images 10
