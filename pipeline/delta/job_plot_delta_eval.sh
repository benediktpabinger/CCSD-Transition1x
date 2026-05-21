#!/bin/bash
#SBATCH --job-name=plot_delta
#SBATCH --partition=h200
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/plot_delta_%j.log

module load Python/3.13.5-GCCcore-14.3.0

python3 -u /home/energy/s242862/pipeline/delta/plot_delta_eval.py --n-reactions 20
