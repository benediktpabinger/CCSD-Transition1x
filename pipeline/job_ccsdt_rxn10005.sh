#!/bin/bash
#SBATCH --job-name=ccsdt_rxn10005
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ccsdt_rxn10005_%j.out
#SBATCH --error=logs/ccsdt_rxn10005_%j.err

source /etc/profile.d/modules.sh
module load Python/3.11.3-GCCcore-12.3.0

export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24

mkdir -p logs

cd $HOME
python3 pipeline/ccsdt_rxn10005_optts.py
