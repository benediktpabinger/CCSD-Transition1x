#!/bin/bash
#SBATCH --job-name=mr_mlip_sp_extra3
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/energy/s242862/logs/mr_mlip_sp_extra3_%A_%a.log

# UMA-S/UMA-M/MACE-bare/MACE+delta single points on the ORCA (DFT) NEB
# reactant/TS/product geometries, for rxn5691/rxn1283/rxn0894 -- extends
# the 23-reaction DFT-NEB MLIP comparison to these 3 (see
# job_mr_ccsdt_extra3.sh for why they're being added back in).

set -e

module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.6.0
mkdir -p /home/energy/s242862/logs

METHODS=(uma_s uma_m mace_bare mace_delta)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

RXNS="rxn5691,rxn1283,rxn0894"

echo "Task ${SLURM_ARRAY_TASK_ID}: method=${METHOD}"

python3 /home/energy/s242862/pipeline/mr_benchmark_mlip_sp_batch.py \
    --method ${METHOD} \
    --rxns ${RXNS}
