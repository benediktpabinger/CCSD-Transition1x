#!/bin/bash
#SBATCH --job-name=autocas
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=autocas_%j.out

RXN=${1:-rxn7949}
BASIS=${2:-def2-svp}
BOND_DIM=${3:-250}

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load Boost/1.82.0-GCC-12.3.0
module load GSL/2.7-GCC-12.3.0
module load FlexiBLAS/3.3.1-GCC-12.3.0
module load HDF5/1.14.0-gompi-2023a

echo "Running AutoCAS for $RXN  basis=$BASIS  bond_dim=$BOND_DIM"
python3 ~/pipeline/autocas_active_space.py $RXN \
    --basis $BASIS \
    --bond-dim $BOND_DIM \
    --threads $SLURM_CPUS_PER_TASK
