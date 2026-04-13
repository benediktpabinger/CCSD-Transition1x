#!/bin/bash
#SBATCH --job-name=painn_smoke
#SBATCH --partition=xeon24el8
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=~/logs/painn_smoke_%j.log

mkdir -p ~/logs
module load Python/3.13.5-GCCcore-14.3.0

python3 ~/pipeline/train_painn.py     --db     ~/data/transition1x_smoke.db     --output ~/painn_smoke     --epochs 5     --batch-size 8     --num-workers 2
