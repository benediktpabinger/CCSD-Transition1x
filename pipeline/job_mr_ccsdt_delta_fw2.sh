#!/bin/bash
#SBATCH --job-name=mr_ccsdt_fw2
#SBATCH --partition=xeon24el8
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120GB
#SBATCH --array=0-29
#SBATCH --output=/home/energy/s242862/logs/mr_ccsdt_fw2_%A_%a.log

# CCSD(T)/def2-TZVP on R, TS, P at MACE+delta (fw=2.0) NEB geometries (30 reactions).
# Tests whether the delta-corrected MLIP finds geometries that give the right
# barrier under a higher level of theory, independent of its own energetics.

set -e

REACTIONS=(
    rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn7945 rxn7937 rxn6196 rxn0346 rxn1150
    rxn9246 rxn4498 rxn1061 rxn4003 rxn4004 rxn4063 rxn4114 rxn4060 rxn1961 rxn1962
    rxn0896 rxn1154 rxn5690 rxn4513 rxn7955 rxn4519 rxn4500 rxn2553 rxn8829 rxn1155
)
RXN=${REACTIONS[${SLURM_ARRAY_TASK_ID}]}

OUT=/home/energy/s242862/mr_benchmark/results/${RXN}_ccsdt_delta_fw2.json
if [ -f "$OUT" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN} already done, skipping."
    exit 0
fi

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export BLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export PYSCF_MAX_MEMORY=100000

python3 /home/energy/s242862/pipeline/mr_benchmark_ccsdt.py ${RXN} --n-threads 12 \
    --geom-dir /home/energy/s242862/mace_delta_neb_results_fw2/${RXN} \
    --tag delta_fw2
