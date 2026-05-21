#!/bin/bash
#SBATCH --job-name=analyze_bm
#SBATCH --partition=xeon24el8
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=/home/energy/s242862/logs/analyze_bm_%j.log

module load Python/3.13.5-GCCcore-14.3.0

python3 -u /home/energy/s242862/pipeline/delta/analyze_full_benchmark.py
