#!/bin/bash
#SBATCH --job-name=mr_nevpt2_dftneb16
#SBATCH --partition=xeon24el8
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120GB
#SBATCH --array=0-15
#SBATCH --output=/home/energy/s242862/logs/mr_nevpt2_dftneb16_%A_%a.log

# NEVPT2/AVAS on R, TS, P at the ORCA (DFT) NEB geometry, for the 16
# reactions that don't yet have NEVPT2 at this geometry (they only have it
# at the separately-optimized CASSCF OptTS geometry, or not at all):
# rxn4113 (failed CASSCF convergence at P in the original NEB-geometry run),
# the 12 next-HIGH reactions, and the 3 recovered reactions
# (rxn5691/rxn1283/rxn0894). Uses mr_benchmark_nevpt2.py unmodified -- its
# default geometry source is already orca_neb_results/<rxn>, and it never
# re-optimizes the geometry (single points only).
#
# --max-cycle 2000 (looser than the script's default 1000) since these are
# exactly the reactions CASSCF struggles with -- that's why the more
# elaborate pruning/mc2step OptTS pipeline exists for them. Some fraction
# may still fail to converge; check logs and retry/diagnose individually
# rather than assume all 16 land cleanly.

set -e

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs /home/energy/s242862/nevpt2_results

REACTIONS=(
    rxn4113
    rxn7060 rxn8827 rxn1147 rxn10005
    rxn4518 rxn3107 rxn4522 rxn7936 rxn0101 rxn10054 rxn7957 rxn8837
    rxn5691 rxn1283 rxn0894
)
RXN=${REACTIONS[${SLURM_ARRAY_TASK_ID}]}

OUT=/home/energy/s242862/nevpt2_results/${RXN}_pyscf_avas/nevpt2_results.json
if [ -f "$OUT" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN} already done, skipping."
    exit 0
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN}"

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export BLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export PYSCF_MAX_MEMORY=100000

python3 /home/energy/s242862/pipeline/mr_benchmark_nevpt2.py ${RXN} --n-threads 24 --max-cycle 2000
