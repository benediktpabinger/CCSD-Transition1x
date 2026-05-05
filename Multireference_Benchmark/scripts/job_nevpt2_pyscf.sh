#!/bin/bash
#SBATCH --job-name=nevpt2_pyscf
#SBATCH --partition=xeon24el8
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --output=/home/energy/s242862/logs/nevpt2_pyscf_%j.log

# AVAS -> CASSCF -> NEVPT2 using PySCF.
# AVAS runs once at the TS; MOs are projected to reactant/product for a valid barrier.
# Submit with: sbatch --export=REACTION=rxnXXXX job_nevpt2_pyscf.sh

set -e

REACTION=${REACTION:-rxn0103}
TS_XYZ=/home/energy/s242862/orca_neb_results/${REACTION}/transition_state.xyz
R_XYZ=/home/energy/s242862/orca_neb_results/${REACTION}/reactant.xyz
P_XYZ=/home/energy/s242862/orca_neb_results/${REACTION}/product.xyz
OUTPUT=/home/energy/s242862/nevpt2_results/${REACTION}_pyscf_avas

mkdir -p /home/energy/s242862/logs
mkdir -p ${OUTPUT}

module load Python/3.13.5-GCCcore-14.3.0

export OMP_NUM_THREADS=8
export PYSCF_MAX_MEMORY=60000

SCRATCH=/tmp/nevpt2_pyscf_${SLURM_JOB_ID}
mkdir -p ${SCRATCH}
cd ${SCRATCH}

echo "Reaction: $REACTION"
echo "Output:   $OUTPUT"

python3 /home/energy/s242862/Multireference_Benchmark/scripts/nevpt2_pyscf.py \
    --reaction ${REACTION} \
    --ts_xyz   ${TS_XYZ} \
    --r_xyz    ${R_XYZ} \
    --p_xyz    ${P_XYZ} \
    --output   ${OUTPUT} \
    --basis    def2-tzvp \
    --avas_ao  'C 2pz' 'O 2pz' 'N 2p'

echo "Done. Results in ${OUTPUT}/"
