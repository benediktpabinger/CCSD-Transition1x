#!/bin/bash
#SBATCH --job-name=nevpt2_sp
#SBATCH --partition=xeon24el8
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --output=/home/energy/s242862/logs/nevpt2_%j.log

# CASSCF/NEVPT2/def2-TZVP single-point for rxn0103
# Uses ORCA AutoCAS to select active space automatically

set -e

REACTION=rxn0103
TS_XYZ=/home/energy/s242862/ccsd_neb_results/${REACTION}/transition_state.xyz
R_XYZ=/home/energy/s242862/ccsd_neb_results/${REACTION}/reactant.xyz
P_XYZ=/home/energy/s242862/ccsd_neb_results/${REACTION}/product.xyz
OUTPUT=/home/energy/s242862/nevpt2_results/${REACTION}

mkdir -p /home/energy/s242862/logs
mkdir -p ${OUTPUT}

module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0

ORCA_CMD=$(which orca)
echo "ORCA: $ORCA_CMD"
echo "Reaction: $REACTION"

python3 /home/energy/s242862/pipeline/nevpt2_sp.py \
    --reaction  ${REACTION} \
    --ts_xyz    ${TS_XYZ} \
    --r_xyz     ${R_XYZ} \
    --p_xyz     ${P_XYZ} \
    --output    ${OUTPUT} \
    --orca-cmd  ${ORCA_CMD} \
    --nprocs    8 \
    --nel       6 \
    --norb      6
