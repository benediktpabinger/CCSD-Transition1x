#!/bin/bash
#SBATCH --job-name=delta_head_fixed
#SBATCH --partition=h200
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/delta_head_fixed_%j.log

# Retrain the delta head with the corrected irreps declaration
# (see train_delta_head_fixed.py docstring). Fresh training, fw=2.0,
# otherwise identical settings to the v2 production run.

set -e

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs

python3 /home/energy/s242862/pipeline/delta/train_delta_head_fixed.py \
    --force-weight 2.0 \
    --epochs 200 \
    --batch-size 64 \
    --train-samples 10000

echo "Done. Head saved to /home/energy/s242862/delta_head/delta_head_fixed_fw2.00.pt"
