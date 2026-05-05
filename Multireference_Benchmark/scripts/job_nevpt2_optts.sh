#!/bin/bash
#SBATCH --job-name=nevpt2_optts
#SBATCH --partition=xeon24el8_week
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --output=/home/energy/s242862/logs/nevpt2_optts_%j.log

# CASSCF OptTS + NEVPT2/def2-TZVP
# Submit with: sbatch --export=REACTION=rxnXXXX job_nevpt2_optts.sh
# Step 1: Optimise TS geometry at CASSCF level (starting from wB97M-V TS)
# Step 2: NEVPT2 single-points on CASSCF-optimised TS, reactant, product

set -e

REACTION=${REACTION:-rxn0103}
OUTPUT=/home/energy/s242862/nevpt2_results/${REACTION}_optts
TS_XYZ=/home/energy/s242862/orca_neb_results/${REACTION}/transition_state.xyz
R_XYZ=/home/energy/s242862/orca_neb_results/${REACTION}/reactant.xyz
P_XYZ=/home/energy/s242862/orca_neb_results/${REACTION}/product.xyz

mkdir -p /home/energy/s242862/logs
mkdir -p ${OUTPUT}

module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0

ORCA_CMD=$(which orca)
echo "ORCA: $ORCA_CMD"
echo "Reaction: $REACTION"

# ── Step 1: CASSCF OptTS ────────────────────────────────────────────────────
cat > ${OUTPUT}/optts.inp << EOF
# CASSCF TS optimisation starting from wB97M-V saddle point
! CASSCF def2-TZVP TightSCF OptTS

%pal nprocs 8 end
%maxcore 4000

%casscf
  nel    6
  norb   6
  nroots 1
end

%geom
  calc_hess true
end

* xyzfile 0 1 ${TS_XYZ}
EOF

echo "Running CASSCF OptTS..."
${ORCA_CMD} ${OUTPUT}/optts.inp > ${OUTPUT}/optts.out 2>&1

# Extract optimised TS geometry
OPTTS_XYZ=${OUTPUT}/optts_optts.xyz
if [ ! -f ${OPTTS_XYZ} ]; then
    # ORCA names it differently sometimes
    OPTTS_XYZ=$(ls ${OUTPUT}/*optts*.xyz 2>/dev/null | head -1)
fi
if [ -z "${OPTTS_XYZ}" ]; then
    echo "ERROR: CASSCF OptTS geometry not found"
    exit 1
fi
echo "CASSCF OptTS geometry: ${OPTTS_XYZ}"

# ── Step 2: NEVPT2 single-points ────────────────────────────────────────────
python3 /home/energy/s242862/pipeline/nevpt2_sp.py \
    --reaction  ${REACTION}_optts \
    --ts_xyz    ${OPTTS_XYZ} \
    --r_xyz     ${R_XYZ} \
    --p_xyz     ${P_XYZ} \
    --output    ${OUTPUT} \
    --orca-cmd  ${ORCA_CMD} \
    --nprocs    8 \
    --nel       6 \
    --norb      6

echo "Done. Results in ${OUTPUT}/"
