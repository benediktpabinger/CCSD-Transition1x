#!/bin/bash
#SBATCH --job-name=convert_extxyz
#SBATCH --partition=xeon24el8
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/energy/s242862/logs/convert_extxyz_%j.log

# Converts all three Transition1x splits from H5 to extxyz for MACE training.
# Run this before job_mace_train.sh.

set -e

H5=/home/energy/s242862/data/Transition1x.h5
DATA=/home/energy/s242862/data

module load Python/3.13.5-GCCcore-14.3.0

for SPLIT in train val test; do
    OUT=${DATA}/transition1x_${SPLIT}.xyz
    if [ -f "${OUT}" ]; then
        echo "${OUT} already exists, skipping."
    else
        echo "Converting ${SPLIT} split..."
        python /home/energy/s242862/pipeline/convert_h5_to_extxyz.py \
            --h5file ${H5} \
            --output ${OUT} \
            --split ${SPLIT}
        echo "Done: ${OUT}"
    fi
done

echo "All splits converted."
