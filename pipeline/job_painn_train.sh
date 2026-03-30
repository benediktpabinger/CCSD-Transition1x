#!/bin/bash
#SBATCH --job-name=painn_train
#SBATCH --partition=a100
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=~/logs/painn_train_%j.log

# Usage: sbatch ~/pipeline/job_painn_train.sh
# Trains PaiNN on full Transition1x DFT training set.
# Step 1: convert H5 → SchNetPack .db (if not already done)
# Step 2: train PaiNN with energy + forces

mkdir -p ~/logs
mkdir -p ~/painn_results
mkdir -p ~/data

module load Python/3.13.5-GCCcore-14.3.0

export OMP_NUM_THREADS=8

DB=~/data/transition1x_train.db
OUTPUT=~/painn_results

# Step 1: Convert H5 to SchNetPack db (skip if already done)
if [ ! -f "${DB}" ]; then
    echo "Converting Transition1x H5 to SchNetPack db..."
    python ~/pipeline/convert_transition1x.py \
        --h5file ~/data/Transition1x.h5 \
        --output ${DB} \
        --split  train
else
    echo "DB already exists, skipping conversion."
fi

# Step 2: Train PaiNN
echo "Training PaiNN..."
python ~/pipeline/train_painn.py \
    --db            ${DB} \
    --output        ${OUTPUT} \
    --epochs        500 \
    --batch-size    32 \
    --lr            1e-4 \
    --cutoff        5.0 \
    --n-atom-basis  128 \
    --n-interactions 3 \
    --energy-weight 0.01 \
    --forces-weight 0.99 \
    --num-workers   8 \
    --gpu
