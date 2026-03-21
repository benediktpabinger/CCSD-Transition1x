#!/bin/bash
#SBATCH --job-name=ccsd_tz
#SBATCH --partition=xeon24el8
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/ccsd_tz_%j.log

# Usage: sbatch job_ccsd_cineb_tz.sh rxn0103
REACTION=${1:-rxn0103}
H5FILE=~/data/Transition1x.h5
OUTPUT=~/ccsd_tz_results/${REACTION}
N_IMAGES=10
BASIS=cc-pVTZ
N_THREADS=8   # must match --cpus-per-task

mkdir -p logs
mkdir -p ${OUTPUT}

module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=${N_THREADS}
export MKL_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=${N_THREADS}

echo "Reaction: ${REACTION}"
echo "Output:   ${OUTPUT}"
echo "Threads:  ${N_THREADS}"
echo "Basis:    ${BASIS}"

python ~/ccsd_cineb_tz_pyscf.py \
    --h5file    ${H5FILE} \
    --reaction  ${REACTION} \
    --split     test \
    --n-images  ${N_IMAGES} \
    --basis     ${BASIS} \
    --n-threads ${N_THREADS} \
    --output    ${OUTPUT}
