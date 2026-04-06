#!/bin/bash
#SBATCH --job-name=mace_train
#SBATCH --partition=a100
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/mace_train_%j.log

# Usage: sbatch /home/energy/s242862/pipeline/job_mace_train.sh
# Fine-tunes MACE-OFF23-medium on full Transition1x wB97x/6-31G(d) data.
# Step 1: convert H5 splits to extxyz (if not already done)
# Step 2: fine-tune MACE-OFF23

set -e

H5=/home/energy/s242862/data/Transition1x.h5
TRAIN_XYZ=/home/energy/s242862/data/transition1x_train.xyz
VAL_XYZ=/home/energy/s242862/data/transition1x_val.xyz
TEST_XYZ=/home/energy/s242862/data/transition1x_test.xyz
RESULTS=/home/energy/s242862/mace_results

mkdir -p /home/energy/s242862/logs
mkdir -p ${RESULTS}

module load Python/3.13.5-GCCcore-14.3.0

# Add CUDA driver library to path - find it dynamically across nodes
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
LIBCUDA=$(find /usr -name "libcuda.so.1" 2>/dev/null | head -1)
if [ -n "$LIBCUDA" ]; then
    echo "Found libcuda at: $LIBCUDA"
    export LD_PRELOAD=$LIBCUDA
else
    echo "WARNING: libcuda.so.1 not found"
fi
export CUDA_VISIBLE_DEVICES=0

echo "GPU check:"
nvidia-smi || echo "nvidia-smi failed"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

pip install mace-torch --quiet --user

# Check that conversion has been done (run job_convert_extxyz.sh first)
if [ ! -f "${TRAIN_XYZ}" ] || [ ! -f "${VAL_XYZ}" ] || [ ! -f "${TEST_XYZ}" ]; then
    echo "ERROR: extxyz files not found. Run job_convert_extxyz.sh first."
    exit 1
fi

# Copy data to local scratch for fast I/O
SCRATCH=/tmp/mace_${SLURM_JOB_ID}
mkdir -p ${SCRATCH}
echo "Copying train/val/test xyz to local scratch ${SCRATCH}..."
cp ${TRAIN_XYZ} ${SCRATCH}/train.xyz
cp ${VAL_XYZ}   ${SCRATCH}/val.xyz
cp ${TEST_XYZ}  ${SCRATCH}/test.xyz
echo "Copy done."
TRAIN_XYZ=${SCRATCH}/train.xyz
VAL_XYZ=${SCRATCH}/val.xyz
TEST_XYZ=${SCRATCH}/test.xyz

echo "Starting MACE fine-tuning..."

mace_run_train \
    --name="mace_transition1x" \
    --foundation_model="mace_omol" \
    --train_file="${TRAIN_XYZ}" \
    --valid_file="${VAL_XYZ}" \
    --energy_key="energy" \
    --forces_key="forces" \
    --E0s="average" \
    --loss="huber" \
    --default_dtype="float32" \
    --batch_size=16 \
    --max_num_epochs=100 \
    --lr=1e-4 \
    --swa \
    --start_swa=80 \
    --ema \
    --ema_decay=0.99 \
    --results_dir="${RESULTS}" \
    --device=cuda \
    --num_workers=4
