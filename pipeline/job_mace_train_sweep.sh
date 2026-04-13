#!/bin/bash
#SBATCH --job-name=mace_sweep
#SBATCH --partition=h200
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/mace_sweep_%j.log

# MACE from scratch — LR schedule patience sweep.
# Submit with: sbatch --export=PATIENCE=5 job_mace_train_sweep.sh
#
# Sweep values: 3, 5, 10, 20, 50

set -e

PATIENCE=${PATIENCE:-5}
TRAIN_H5=/home/energy/s242862/t1xtrain
VAL_H5=/home/energy/s242862/t1xval
RESULTS=/home/energy/s242862/mace_sweep_results
NAME=mace_t1x_p${PATIENCE}

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

echo "Starting MACE sweep: patience=${PATIENCE}"

mace_run_train \
    --name="${NAME}" \
    --train_file="${TRAIN_H5}" \
    --valid_file="${VAL_H5}" \
    --energy_key="energy" \
    --forces_key="forces" \
    --E0s="{1: -13.62222753701504, 6: -1029.4130839658328, 7: -1484.8710358098756, 8: -2041.8396277138045, 9: -2712.8213146878606}" \
    --atomic_numbers="[1, 6, 7, 8, 9]" \
    --loss="huber" \
    --default_dtype="float32" \
    --num_channels=1024 \
    --max_L=3 \
    --r_max=6.0 \
    --num_interactions=2 \
    --num_radial_basis=16 \
    --batch_size=64 \
    --max_num_epochs=100 \
    --max_samples_per_epoch=100000 \
    --max_valid_samples=10000 \
    --optimizer="adamw" \
    --weight_decay=0.0 \
    --clip_grad=10.0 \
    --lr=1e-3 \
    --scheduler="ReduceLROnPlateau" \
    --scheduler_patience=${PATIENCE} \
    --results_dir="${RESULTS}" \
    --device=cuda \
    --num_workers=4 \
    --wandb \
    --wandb_project="transition1x-mace" \
    --wandb_entity="s242862-danmarks-tekniske-universitet-dtu" \
    --wandb_name="${NAME}_${SLURM_JOB_ID}" \
    --log_level="INFO"
