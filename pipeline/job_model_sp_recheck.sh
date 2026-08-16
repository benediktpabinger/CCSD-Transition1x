#!/bin/bash
#SBATCH --job-name=spcheck
#SBATCH --partition=sm3090el8
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/energy/s242862/model_sp_recheck/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/model_sp_recheck/slurm_%A_%a.err

# Re-evaluate each model at the geometry it itself predicted, and compare the
# fresh forces with the ones stored in transition_state.xyz.
#
# The force-error result says the models are off by 0.031 eV/A at their own
# transition states.  That number comes from stored forces on one side and a
# fresh ORCA gradient on the other.  If the stored forces did not belong to the
# stored coordinates, part of the 0.031 would be an artefact.  One single point
# per structure settles it.
#
# The models are deterministic, so agreement should be at the 1e-6 level.
# Anything approaching 0.03 would mean the force analysis has to be rebuilt.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

H=/home/energy/s242862
M=(UMA-S UMA-M eSEN)
MODEL=${M[$SLURM_ARRAY_TASK_ID]}

mkdir -p $H/model_sp_recheck
cd $H/model_sp_recheck

echo "Task $SLURM_ARRAY_TASK_ID: $MODEL  node $SLURM_NODELIST  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 $H/model_sp_recheck.py --model $MODEL \
        --out $H/model_sp_recheck/${MODEL}.json

echo "Finished $(date)"
