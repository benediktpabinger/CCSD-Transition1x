#!/bin/bash
#SBATCH --job-name=gpu_diag
#SBATCH --partition=a100
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/gpu_diag_%j.log

find / -name "libcuda.so*" 2>/dev/null
echo "---LD_LIBRARY_PATH---"
echo $LD_LIBRARY_PATH
echo "---ldconfig---"
ldconfig -p 2>/dev/null | grep libcuda
