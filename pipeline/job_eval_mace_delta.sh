#!/bin/bash
#SBATCH --job-name=mace_delta
#SBATCH --partition=sm3090_devel
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/mace_delta_%j.log

set -e

module load Python/3.13.5-GCCcore-14.3.0

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
LIBCUDA=$(find /usr -name "libcuda.so.1" 2>/dev/null | head -1)
if [ -n "$LIBCUDA" ]; then
    export LD_PRELOAD=$LIBCUDA
fi
export CUDA_VISIBLE_DEVICES=0

mkdir -p /home/energy/s242862/logs

echo "GPU check:"
nvidia-smi | head -14
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python3 /home/energy/s242862/pipeline/eval_mace_delta.py

echo "Done."
