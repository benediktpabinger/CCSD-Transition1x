#!/bin/bash
#SBATCH --job-name=esen_neb
#SBATCH --partition=sm3090el8
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --output=/home/energy/s242862/logs/esen_neb_%A_%a.log

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
    rxn7949 rxn8832 rxn1320 rxn4113 rxn8885
    rxn7945 rxn7937 rxn6196 rxn0346 rxn1150
    rxn0896 rxn1154 rxn5690 rxn4513 rxn7955
    rxn4519 rxn4500 rxn2553 rxn8829 rxn1155
    rxn9246 rxn4498 rxn1061 rxn4003 rxn4004
    rxn4063 rxn4114 rxn4060 rxn1961 rxn1962
)

RXN=${REACTIONS[$SLURM_ARRAY_TASK_ID]}
OUTPUT=/home/energy/s242862/esen_neb_results/${RXN}
CHECKPOINT=/home/energy/s242862/checkpoints/esen_sm_conserving_all.pt

echo "Reaction: ${RXN}"
echo "Output:   ${OUTPUT}"

# Skip if already converged
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
