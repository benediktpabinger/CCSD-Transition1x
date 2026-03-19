#!/bin/bash
#SBATCH --job-name=ccsd_neb
#SBATCH --partition=xeon24el8
#SBATCH --time=24:00:00              # CCSD NEB is expensive — 24h to be safe
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ccsd_neb_%j.log

# ============================================================
# EINSTELLUNGEN
# ============================================================
REACTION=${1:-rxn0103}   # sbatch job_neb.sh rxn0103
H5FILE=~/data/Transition1x.h5
OUTPUT=~/ccsd_neb_results/${REACTION}
N_IMAGES=10
BASIS=cc-pVDZ
NPROCS=1   # serial ORCA (same as working TZ single-point jobs)
# ============================================================

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

python ~/mp2_neb.py \
    --h5file    ${H5FILE} \
    --reaction  ${REACTION} \
    --split     test \
    --n-images  ${N_IMAGES} \
    --basis     ${BASIS} \
    --nprocs    ${NPROCS} \
    --output    ${OUTPUT}

# ============================================================
# PARALLEL MODE (for full test set — faster but needs more resources):
#
# Change SBATCH options to:
#   #SBATCH --ntasks=8          # n_images - 2 interior images
#   #SBATCH --cpus-per-task=1
#   #SBATCH --mem=64GB
#
# And replace the python call with:
#   mpirun -n 8 python ~/ccsd_neb.py ... --parallel --nprocs 1
# ============================================================
