#!/bin/bash
#SBATCH --job-name=delta_head_v2
#SBATCH --partition=h200
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=/home/energy/s242862/logs/delta_head_v2_%A_%a.log

# Train delta head v2 — 3 parallel runs sweeping force_weight (0.5, 1.0, 2.0).
# MLP_IRREPS=64x0e, batch_size=64, epochs=200 with early stopping.
# Run prepare_delta_data.py first to generate train.pt / val.pt.

set -e

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs

FORCE_WEIGHTS=(0.5 1.0 2.0)
FW=${FORCE_WEIGHTS[$SLURM_ARRAY_TASK_ID]}
FW_FMT=$(printf "%.2f" ${FW})

echo "Task ${SLURM_ARRAY_TASK_ID}: force_weight=${FW}"

python3 /home/energy/s242862/pipeline/delta/train_delta_head.py \
    --force-weight ${FW} \
    --epochs 200 \
    --batch-size 64 \
    --train-samples 10000 \
    --resume /home/energy/s242862/delta_head/delta_head_fw${FW_FMT}.pt

echo "Done. Head saved to /home/energy/s242862/delta_head/delta_head_fw${FW}.pt"
