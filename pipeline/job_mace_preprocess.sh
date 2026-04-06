#!/bin/bash
#SBATCH --job-name=mace_preprocess
#SBATCH --partition=xeon24el8
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --output=/home/energy/s242862/logs/mace_preprocess_%j.log

# Convert Transition1x extxyz to MACE sharded HDF5 format.
# Run this once before job_mace_train_scratch.sh.
# Uses 16 processes to create 16 shards → 10x faster training I/O.

set -e

DATA=/home/energy/s242862/data
H5DIR=/home/energy/s242862/data/mace_h5

mkdir -p ${H5DIR}

module load Python/3.13.5-GCCcore-14.3.0

echo "Preprocessing Transition1x data to sharded HDF5..."

mace_prepare_data \
    --train_file       ${DATA}/transition1x_train.xyz \
    --valid_file       ${DATA}/transition1x_val.xyz \
    --test_file        ${DATA}/transition1x_test.xyz \
    --work_dir         ${H5DIR} \
    --h5_prefix        t1x \
    --r_max            6.0 \
    --num_process      16 \
    --energy_key       energy \
    --forces_key       forces \
    --compute_statistics \
    --E0s              "average" \
    --shuffle          True

echo "Done. HDF5 files written to ${H5DIR}"
ls -lh ${H5DIR}
