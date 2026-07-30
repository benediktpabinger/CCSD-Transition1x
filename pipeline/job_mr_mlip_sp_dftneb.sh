#!/bin/bash
#SBATCH --job-name=mr_mlip_sp_dftneb
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/energy/s242862/logs/mr_mlip_sp_dftneb_%A_%a.log

# UMA-S/UMA-M/eSEN/MACE-bare/MACE+delta single points on the ORCA (DFT) NEB
# reactant/TS/product geometries -- one task per method, looping over all 23
# MR benchmark reactions (10 High(orig) + 13 next-HIGH). Complements
# full_benchmark_results.json / barrier_comparison_optts.json, whose existing
# MLIP barrier columns come from each MLIP's own independently-run NEB, not a
# single point on the fixed DFT geometry.

set -e

module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.6.0
mkdir -p /home/energy/s242862/logs

METHODS=(uma_s uma_m esen mace_bare mace_delta)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

RXNS="rxn7949,rxn8832,rxn1320,rxn4113,rxn8885,rxn7945,rxn7937,rxn6196,rxn0346,rxn1150,rxn0896,rxn7060,rxn8827,rxn1147,rxn10005,rxn4518,rxn3107,rxn4522,rxn7936,rxn0101,rxn10054,rxn7957,rxn8837"

echo "Task ${SLURM_ARRAY_TASK_ID}: method=${METHOD}"

python3 /home/energy/s242862/pipeline/mr_benchmark_mlip_sp_batch.py \
    --method ${METHOD} \
    --rxns ${RXNS}
