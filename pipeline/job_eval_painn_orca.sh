#!/bin/bash
#SBATCH --job-name=eval_painn_orca
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/~/logs/eval_painn_orca_%j.log

module load Python/3.13.5-GCCcore-14.3.0

python -u ~/pipeline/eval_painn_orca.py \
    --orca-dir   ~/orca_neb_results \
    --checkpoint ~/painn_results_v2/checkpoints/best.ckpt \
    --n-images   10 \
    --output     ~/eval_painn_orca.json
