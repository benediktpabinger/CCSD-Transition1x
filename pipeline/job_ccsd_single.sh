#!/bin/bash
#SBATCH --job-name=ccsd_neb
#SBATCH --partition=xeon24el8
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=~/logs/ccsd_neb_%j.log

# Usage: sbatch ~/pipeline/job_ccsd_single.sh rxn0103
# Runs CCSD/cc-pVDZ CI-NEB for a single reaction.
# Useful for testing or resubmitting a timed-out reaction.

REACTION=${1:-rxn0103}
OUTPUT=~/ccsd_pyscf_results/${REACTION}

mkdir -p ~/logs
mkdir -p ${OUTPUT}

module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Reaction:  ${REACTION}"
echo "Output:    ${OUTPUT}"

python ~/pipeline/ccsd_neb_pyscf.py \
    --h5file     ~/data/Transition1x.h5 \
    --reaction   ${REACTION} \
    --split      test \
    --n-images   10 \
    --basis      cc-pVDZ \
    --n-threads  8 \
    --neb-fmax   0.15 \
    --cineb-fmax 0.05 \
    --output     ${OUTPUT} \
    --skip-relax
