#!/bin/bash
#SBATCH --job-name=force_conserv
#SBATCH --partition=h200
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/force_conserv_%j.log

module load Python/3.13.5-GCCcore-14.3.0

python3 -u /home/energy/s242862/pipeline/_check_force_conservativeness.py
