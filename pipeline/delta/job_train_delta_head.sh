#!/bin/bash
#SBATCH --job-name=train_delta_head
#SBATCH --partition=h200
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/train_delta_head_%j.log

# Prepare cached train/val data, then train the delta correction head.
# Requires all SP results to be in ~/train_delta_sp/, ~/val_delta_sp/,
# and ~/val_delta_sp_flip/ before submission.

set -e

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs

HOME=/home/energy/s242862
PIPELINE=${HOME}/pipeline/delta

echo "=== Step 1: Prepare delta cache ==="
python3 ${PIPELINE}/prepare_delta_data.py

echo "=== Step 2: Train delta head ==="
python3 ${PIPELINE}/train_delta_head.py \
    --epochs 200 \
    --batch-size 32 \
    --lr 1e-3 \
    --val-samples 1024 \
    --huber-delta 0.1 \
    --force-weight 1.0 \
    --patience 10

echo "Done. Head saved to ${HOME}/delta_head/delta_head.pt"
