#!/bin/bash
#SBATCH --job-name=mp2_neb
#SBATCH --partition=xeon24el8
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/mp2_neb_%j.log

# Usage: sbatch job_neb.sh rxn0103
REACTION=${1:-rxn0103}
H5FILE=~/data/Transition1x.h5
OUTPUT=~/ccsd_neb_results/${REACTION}
N_IMAGES=10
BASIS=cc-pVDZ
NPROCS=1

mkdir -p logs
mkdir -p ${OUTPUT}

module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Reaction: ${REACTION}"
echo "Output:   ${OUTPUT}"

python ~/mp2_neb_warmstart.py \
    --h5file    ${H5FILE} \
    --reaction  ${REACTION} \
    --split     test \
    --n-images  ${N_IMAGES} \
    --basis     ${BASIS} \
    --nprocs    ${NPROCS} \
    --output    ${OUTPUT}
