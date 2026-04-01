#!/bin/bash
#SBATCH --job-name=eval_delta
#SBATCH --partition=a100
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/~/logs/eval_delta_%j.log

module load Python/3.13.5-GCCcore-14.3.0

python -u ~/pipeline/evaluate_delta_model.py \
    --val-list    ~/ccsd_dataset/val_reactions.txt \
    --ccsd-dir    ~/ccsd_pyscf_results \
    --checkpoint  ~/painn_results/checkpoints/best.ckpt \
    --delta-model ~/curator_results/round1/delta_model.json \
    --n-images    10
