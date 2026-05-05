#!/bin/bash
#SBATCH --job-name=eval_mace
#SBATCH --partition=sm3090_devel
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/eval_mace_%j.log

set -e

module load Python/3.13.5-GCCcore-14.3.0

mkdir -p /home/energy/s242862/logs

echo "=== Evaluating mace_t1x_p5 (best checkpoint) ==="
python3 /home/energy/s242862/pipeline/eval_mace.py \
    --model   /home/energy/s242862/checkpoints/mace_t1x_p5_run-123.model \
    --test    /home/energy/s242862/data/transition1x_test.xyz \
    --output  /home/energy/s242862/eval_mace_p5_best.json \
    --n-test  5000

echo "Done."
