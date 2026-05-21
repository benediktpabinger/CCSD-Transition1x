#!/bin/bash
#SBATCH --job-name=inspect_mace
#SBATCH --partition=xeon24el8
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=/home/energy/s242862/logs/inspect_mace_%j.log

module load Python/3.13.5-GCCcore-14.3.0
python3 /home/energy/s242862/pipeline/inspect_mace_model.py
