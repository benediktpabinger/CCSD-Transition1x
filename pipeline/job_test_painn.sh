#!/bin/bash
#SBATCH --job-name=test_painn
#SBATCH --partition=a100
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/~/logs/test_painn_%j.log

module load Python/3.13.5-GCCcore-14.3.0
python3 ~/pipeline/test_painn.py
