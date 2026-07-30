#!/bin/bash
#SBATCH --job-name=ccsdt_rxn8827
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ccsdt_rxn8827_%j.out
#SBATCH --error=logs/ccsdt_rxn8827_%j.err

source /etc/profile.d/modules.sh
module load Python/3.11.3-GCCcore-12.3.0

export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24

mkdir -p logs

cd $HOME
python3 pipeline/ccsdt_rxn8827_optts.py
