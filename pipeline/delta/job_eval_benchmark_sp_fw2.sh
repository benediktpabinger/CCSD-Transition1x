#!/bin/bash
#SBATCH --job-name=eval_bm_sp_fw2
#SBATCH --partition=h200
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/eval_bm_sp_fw2_%j.log

module load Python/3.13.5-GCCcore-14.3.0

python3 -u /home/energy/s242862/pipeline/delta/eval_benchmark_sp_fw2.py
