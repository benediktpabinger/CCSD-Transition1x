#!/bin/bash
#SBATCH --job-name=mace_scratch
#SBATCH --partition=a100
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/mace_scratch_%j.log

# Train MACE from scratch on Transition1x wB97x/6-31G(d).
# Architecture matches MACE-OMol25 (1024 channels, L=3, r_max=6.0).
# Run job_mace_preprocess.sh first to generate sharded HDF5 files.
#
# Usage: sbatch /home/energy/s242862/pipeline/job_mace_train_scratch.sh

set -e

H5DIR=/home/energy/s242862/data/mace_h5
RESULTS=/home/energy/s242862/mace_scratch_results
NAME=mace_t1x_scratch

mkdir -p /home/energy/s242862/logs
mkdir -p ${RESULTS}

module load Python/3.13.5-GCCcore-14.3.0

# Fix CUDA driver visibility
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
LIBCUDA=$(find /usr -name "libcuda.so.1" 2>/dev/null | head -1)
if [ -n "$LIBCUDA" ]; then
    echo "Found libcuda at: $LIBCUDA"
    export LD_PRELOAD=$LIBCUDA
fi
export CUDA_VISIBLE_DEVICES=0

echo "GPU check:"
nvidia-smi | head -14
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check H5 files exist
if [ ! -d "${H5DIR}" ]; then
    echo "ERROR: HDF5 data not found. Run job_mace_preprocess.sh first."
    exit 1
fi

echo "Starting MACE training from scratch..."

mace_run_train \
    --name="${NAME}" \
    --train_file="${H5DIR}/t1x_train.h5" \
    --valid_file="${H5DIR}/t1x_valid.h5" \
    --energy_key="energy" \
    --forces_key="forces" \
    --E0s="average" \
    --loss="mae" \
    --default_dtype="float32" \
    --num_channels=1024 \
    --max_L=3 \
    --r_max=6.0 \
    --num_interactions=2 \
    --num_radial_basis=8 \
    --batch_size=16 \
    --max_num_epochs=100 \
    --optimizer="adamw" \
    --weight_decay=0.0 \
    --clip_grad=10.0 \
    --lr=1e-3 \
    --scheduler="ReduceLROnPlateau" \
    --scheduler_patience=5 \
    --swa \
    --start_swa=80 \
    --ema \
    --ema_decay=0.99 \
    --results_dir="${RESULTS}" \
    --device=cuda \
    --num_workers=4 \
    --wandb \
    --wandb_project="transition1x-mace" \
    --wandb_entity="s242862-danmarks-tekniske-universitet-dtu" \
    --wandb_name="${NAME}_${SLURM_JOB_ID}" \
    --log_level="INFO"
