#!/bin/bash
#SBATCH --job-name=mace_b256
#SBATCH --partition=h200
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/mace_b256_%j.log

# MACE batch size 256 speed test — compare epoch time vs batch 32

set -e

TRAIN_H5=/home/energy/s242862/t1xtrain
VAL_H5=/home/energy/s242862/t1xval
RESULTS=/home/energy/s242862/mace_sweep_results
NAME=mace_t1x_b256

mkdir -p /home/energy/s242862/logs
mkdir -p ${RESULTS}

module load Python/3.13.5-GCCcore-14.3.0

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
LIBCUDA=$(find /usr -name "libcuda.so.1" 2>/dev/null | head -1)
if [ -n "$LIBCUDA" ]; then
    export LD_PRELOAD=$LIBCUDA
fi
export CUDA_VISIBLE_DEVICES=0

echo "GPU check:"
nvidia-smi | head -14
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "Starting MACE batch=256 test"

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
    --batch_size=256 \
    --max_num_epochs=100 \
    --optimizer="adamw" \
    --weight_decay=0.0 \
    --clip_grad=10.0 \
    --lr=1e-3 \
    --scheduler="ReduceLROnPlateau" \
    --scheduler_patience=5 \
    --results_dir="${RESULTS}" \
    --device=cuda \
    --num_workers=8 \
    --wandb \
    --wandb_project="transition1x-mace" \
    --wandb_entity="s242862-danmarks-tekniske-universitet-dtu" \
    --wandb_name="${NAME}_${SLURM_JOB_ID}" \
    --log_level="INFO"
