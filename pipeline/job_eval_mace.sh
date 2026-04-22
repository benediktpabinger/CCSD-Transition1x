#!/bin/bash
#SBATCH --job-name=eval_mace
#SBATCH --partition=h200
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/eval_mace_%j.log

set -e

module load Python/3.13.5-GCCcore-14.3.0

mkdir -p /home/energy/s242862/logs

for MODEL in p5 p10; do
    echo "=== Evaluating mace_t1x_${MODEL} ==="
    python3 /home/energy/s242862/pipeline/eval_mace.py \
        --model   /home/energy/s242862/mace_t1x_${MODEL}_compiled.model \
        --test    /home/energy/s242862/data/transition1x_test.xyz \
        --neb-dir /home/energy/s242862/orca_neb_results \
        --output  /home/energy/s242862/eval_mace_${MODEL}.json \
        --n-test  10
done

echo "Done."
