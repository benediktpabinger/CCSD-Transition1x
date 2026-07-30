#!/bin/bash
#SBATCH --job-name=mr_ccsdt_extra3
#SBATCH --partition=xeon24el8
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120GB
#SBATCH --array=0-2
#SBATCH --output=/home/energy/s242862/logs/mr_ccsdt_extra3_%A_%a.log

# CCSD(T)/def2-TZVP on R, TS, P at the ORCA (DFT) NEB geometry, for
# rxn5691/rxn1283/rxn0894 -- the 3 reactions previously marked "OptTS did
# not converge" and dropped from the 23-reaction benchmark, but which turn
# out to have complete, physically sane nevpt2_optts_results.json on disk
# (the "failed" docs appear stale). Uses the default geometry source in
# mr_benchmark_ccsdt.py (orca_neb_results/<rxn>).

set -e

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs /home/energy/s242862/mr_benchmark/results

REACTIONS=(rxn5691 rxn1283 rxn0894)
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
