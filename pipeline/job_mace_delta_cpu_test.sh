#!/bin/bash
#SBATCH --job-name=mace_delta_cpu
#SBATCH --partition=xeon24el8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/home/energy/s242862/logs/mace_delta_cpu_test.log

module load Python/3.13.5-GCCcore-14.3.0

python3 /home/energy/s242862/pipeline/mace_delta_neb.py \
    --reaction rxn4063 \
    --output   /home/energy/s242862/mace_delta_neb_results/rxn4063_cpu \
    --device   cpu
