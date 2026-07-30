#!/bin/bash
#SBATCH --job-name=mr_ccsdt_nexthigh
#SBATCH --partition=xeon24el8
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120GB
#SBATCH --array=0-11
#SBATCH --output=/home/energy/s242862/logs/mr_ccsdt_nexthigh_%A_%a.log

# CCSD(T)/def2-TZVP on R, TS, P at the ORCA (DFT) NEB geometry, for the 12
# next-HIGH MR benchmark reactions that don't yet have CCSD(T) at this
# geometry (they only have CCSD(T) at the separately-optimized CASSCF OptTS
# geometry, or none at all). Uses the default geometry source in
# mr_benchmark_ccsdt.py (orca_neb_results/<rxn>), same script/settings as the
# original 10 High-MR reactions.

set -e

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs /home/energy/s242862/mr_benchmark/results

REACTIONS=(
    rxn7060 rxn8827 rxn1147 rxn10005
    rxn4518 rxn3107 rxn4522 rxn7936 rxn0101 rxn10054 rxn7957
    rxn8837
)
RXN=${REACTIONS[${SLURM_ARRAY_TASK_ID}]}

OUT=/home/energy/s242862/mr_benchmark/results/${RXN}_ccsdt.json
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

python3 /home/energy/s242862/pipeline/mr_benchmark_ccsdt.py ${RXN} --n-threads 24
