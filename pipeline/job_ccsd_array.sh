#!/bin/bash
#SBATCH --job-name=ccsd_neb
#SBATCH --partition=xeon24el8
#SBATCH --time=50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --array=1-287%80
#SBATCH --output=~/logs/ccsd_neb_%A_%a.log

# Usage: sbatch ~/pipeline/job_ccsd_array.sh
# Runs CCSD/cc-pVDZ CI-NEB for all 287 test set reactions.
# Reactions are read from ~/ccsd_dataset/test_reactions.txt (line N = array task N).
# Safe to resubmit — skips reactions that already have a 'converged' file,
# and skips endpoint relaxation if reactant.xyz/product.xyz already exist.

REACTION=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ~/ccsd_dataset/test_reactions.txt)
OUTPUT=~/ccsd_pyscf_results/${REACTION}

# Skip if already converged
if [ -f "${OUTPUT}/converged" ]; then
    echo "Reaction ${REACTION} already converged, skipping."
    exit 0
fi

mkdir -p ~/logs
mkdir -p ${OUTPUT}

module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Reaction:  ${REACTION}"
echo "Output:    ${OUTPUT}"
echo "Array job: ${SLURM_ARRAY_JOB_ID}, task: ${SLURM_ARRAY_TASK_ID}"

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
