#!/bin/bash
#SBATCH --job-name=esen_neb_next
#SBATCH --partition=sm3090el8
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-14
#SBATCH --output=/home/energy/s242862/logs/esen_neb_next_%A_%a.log

# eSEN NEB for the next-HIGH MR benchmark extension (FOD ranks 12-26,
# 15 new reactions). rxn0896 (rank 11) already has this from the Mid-MR group.

set -e

module load Python/3.13.5-GCCcore-14.3.0

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
LIBCUDA=$(find /usr -name "libcuda.so.1" 2>/dev/null | head -1)
if [ -n "$LIBCUDA" ]; then
    export LD_PRELOAD=$LIBCUDA
fi
export CUDA_VISIBLE_DEVICES=0

mkdir -p /home/energy/s242862/logs

REACTIONS=(
    rxn4518 rxn3107 rxn8837 rxn7060 rxn5691
    rxn1283 rxn8827 rxn4522 rxn7936 rxn1147
    rxn0894 rxn0101 rxn10005 rxn10054 rxn7957
)

RXN=${REACTIONS[$SLURM_ARRAY_TASK_ID]}
OUTPUT=/home/energy/s242862/esen_neb_results/${RXN}
CHECKPOINT=/home/energy/s242862/checkpoints/esen_sm_conserving_all.pt

echo "Reaction: ${RXN}"
echo "Output:   ${OUTPUT}"

if [ -f "${OUTPUT}/converged" ]; then
    echo "Already converged — skipping."
    exit 0
fi

mkdir -p ${OUTPUT}

python3 /home/energy/s242862/pipeline/esen_neb.py \
    --h5file     /home/energy/s242862/data/Transition1x.h5 \
    --reaction   ${RXN} \
    --split      test \
    --output     ${OUTPUT} \
    --checkpoint ${CHECKPOINT} \
    --neb-fmax   0.15 \
    --cineb-fmax 0.05 \
    --steps      500

echo "Done: ${RXN}"
