#!/bin/bash
#SBATCH --job-name=mace_delta_neb_next
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-14
#SBATCH --output=/home/energy/s242862/logs/mace_delta_neb_next_%A_%a.log

# MACE+delta NEB for the next-HIGH MR benchmark extension (FOD ranks 12-26,
# 15 new reactions). rxn0896 (rank 11) already has this from the Mid-MR group.

module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.6.0

REACTIONS=(
    rxn4518 rxn3107 rxn8837 rxn7060 rxn5691
    rxn1283 rxn8827 rxn4522 rxn7936 rxn1147
    rxn0894 rxn0101 rxn10005 rxn10054 rxn7957
)

RXN=${REACTIONS[$SLURM_ARRAY_TASK_ID]}
HOME=/home/energy/s242862
OUTPUT=$HOME/mace_delta_neb_results/$RXN

echo "Reaction: $RXN"
echo "Output:   $OUTPUT"

python3 $HOME/pipeline/mace_delta_neb.py \
    --reaction $RXN \
    --output   $OUTPUT
