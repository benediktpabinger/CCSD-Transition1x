#!/bin/bash
#SBATCH --job-name=bs_uks_neb
#SBATCH --array=0-22
#SBATCH --output=logs/bs_uks_neb_%A_%a.out
#SBATCH --error=logs/bs_uks_neb_%A_%a.err
#SBATCH --partition=xeon24el8_week
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00

REACTIONS=(
    rxn7949 rxn8832 rxn1320 rxn4113 rxn8885
    rxn7945 rxn7937 rxn6196 rxn0346 rxn1150
    rxn0896 rxn4518 rxn3107 rxn8837 rxn7060
    rxn8827 rxn4522 rxn7936 rxn1147 rxn0101
    rxn10005 rxn10054 rxn7957
)

RXN=${REACTIONS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID: $RXN"
mkdir -p logs

module load ORCA/5.0.4-gompi-2023a
ORCA_EXE=/home/modules/software/ORCA/5.0.4-gompi-2023a/bin/orca

python3 /home/energy/s242862/pipeline/bs_uks_neb.py \
    "$RXN" \
    --nprocs 8 \
    --maxcore 8000 \
    --orca-path "$ORCA_EXE"
