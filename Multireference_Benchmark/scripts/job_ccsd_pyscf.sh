#!/bin/bash
#SBATCH --job-name=ccsd_pyscf
#SBATCH --partition=xeon24el8
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --output=/home/energy/s242862/logs/ccsd_pyscf_%j.log

# RHF -> CCSD -> CCSD(T) using PySCF, def2-TZVP
# Submit with: sbatch --export=REACTION=rxnXXXX job_ccsd_pyscf.sh

set -e

REACTION=${REACTION:-rxn0103}
TS_XYZ=/home/energy/s242862/orca_neb_results/${REACTION}/transition_state.xyz
R_XYZ=/home/energy/s242862/orca_neb_results/${REACTION}/reactant.xyz
P_XYZ=/home/energy/s242862/orca_neb_results/${REACTION}/product.xyz
OUTPUT=/home/energy/s242862/nevpt2_results/${REACTION}_ccsd_pyscf

mkdir -p /home/energy/s242862/logs
mkdir -p ${OUTPUT}

module load Python/3.13.5-GCCcore-14.3.0

export OMP_NUM_THREADS=8
export PYSCF_MAX_MEMORY=60000

SCRATCH=/tmp/ccsd_pyscf_${SLURM_JOB_ID}
mkdir -p ${SCRATCH}
cd ${SCRATCH}

echo "Reaction: $REACTION"
echo "Output:   $OUTPUT"

python3 /home/energy/s242862/Multireference_Benchmark/scripts/ccsd_pyscf.py \
    --reaction ${REACTION} \
    --ts_xyz   ${TS_XYZ} \
    --r_xyz    ${R_XYZ} \
    --p_xyz    ${P_XYZ} \
    --output   ${OUTPUT} \
    --basis    def2-tzvp

echo "Done. Results in ${OUTPUT}/"
