#!/bin/bash
#SBATCH --job-name=compare_barriers
#SBATCH --partition=sm3090_devel
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/compare_barriers_%j.log

set -e

module load Python/3.13.5-GCCcore-14.3.0

mkdir -p /home/energy/s242862/logs

python3 /home/energy/s242862/pipeline/compare_barriers.py \
    --neb-dir /home/energy/s242862/orca_neb_results \
    --t1x-h5  /home/energy/s242862/data/Transition1x.h5 \
    --model   /home/energy/s242862/mace_t1x_p10_epoch291.model \
    --output  /home/energy/s242862/barrier_comparison_p10.json \
    --plot    /home/energy/s242862/barrier_comparison_p10.png

echo "Done."
