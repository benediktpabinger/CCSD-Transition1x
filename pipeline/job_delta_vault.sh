#!/bin/bash
#SBATCH --job-name=delta_vault
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/~/logs/delta_vault_%j.log

module load Python/3.13.5-GCCcore-14.3.0

python -u ~/pipeline/train_eval_delta_vault.py \
    --ccsd-dir   ~/ccsd_pyscf_results \
    --checkpoint ~/painn_results_v2/checkpoints/best.ckpt \
    --n-images   10 \
    --n-train    200 \
    --output     ~/delta_vault_results
