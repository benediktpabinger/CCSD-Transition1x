#!/bin/bash
#SBATCH --job-name=maced_neb_fw2
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-29
#SBATCH --output=/home/energy/s242862/logs/mace_delta_neb_fw2_%A_%a.log

# MACE+delta NEB with fw=2.0 head — results go to mace_delta_neb_results_fw2/
# to keep fw=1.0 results intact for comparison.

module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.6.0

REACTIONS=(
    rxn7949 rxn8832 rxn1320 rxn4113 rxn8885
    rxn7945 rxn7937 rxn6196 rxn0346 rxn1150
    rxn9246 rxn4498 rxn1061 rxn4003 rxn4004
    rxn4063 rxn4114 rxn4060 rxn1961 rxn1962
    rxn0896 rxn1154 rxn5690 rxn4513 rxn7955
    rxn4519 rxn4500 rxn2553 rxn8829 rxn1155
)

RXN=${REACTIONS[$SLURM_ARRAY_TASK_ID]}
HOME=/home/energy/s242862
OUTPUT=$HOME/mace_delta_neb_results_fw2/$RXN

echo "Reaction: $RXN"
echo "Output:   $OUTPUT"

python3 $HOME/pipeline/mace_delta_neb.py \
    --reaction $RXN \
    --output   $OUTPUT \
    --head     $HOME/delta_head/delta_head_fw2.00.pt
