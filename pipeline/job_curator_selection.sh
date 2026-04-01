#!/bin/bash
#SBATCH --job-name=curator_select
#SBATCH --partition=a100
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/~/logs/curator_select_%j.log

# LCMD+GRAD selection on full DFT training set.
# Extracts 64-dim last-layer features from PaiNN for all 7M configs,
# runs LCMD greedy to select n_select configs, saves geometries for CCSD SP.
# Usage: sbatch ~/pipeline/job_curator_selection.sh

module load Python/3.13.5-GCCcore-14.3.0

export OMP_NUM_THREADS=8

ROUND=2

python -u ~/pipeline/run_curator_selection.py \
    --checkpoint ~/painn_results/checkpoints/best.ckpt \
    --db         ~/data/transition1x_train.db \
    --n-select   10 \
    --output     ~/curator_results/round${ROUND} \
    --batch-size 512 \
    --num-workers 4 \
    --max-configs 100000 \
    --gpu
