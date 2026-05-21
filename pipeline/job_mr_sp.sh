#!/bin/bash
#SBATCH --job-name=mr_sp
#SBATCH --partition=xeon24el8
#SBATCH --time=1:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=/home/energy/s242862/mr_benchmark/logs/sp_%A_%a.log

# wB97X-D3/6-31G(d) single points on all MR benchmark NEB geometries.
# Run mr_benchmark_setup.py first to generate inputs, then:
#   sbatch --array=0-<N-1>%40 job_mr_sp.sh
# where N = number of sp.inp files (printed by setup script).

set -e

module load ORCA/5.0.4-gompi-2023a
mkdir -p /home/energy/s242862/mr_benchmark/logs

SP_BASE=/home/energy/s242862/mr_benchmark/orca_sp

mapfile -t ALL_INPUTS < <(find ${SP_BASE} -name 'sp.inp' | sort)
TOTAL=${#ALL_INPUTS[@]}

if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL} ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} >= total ${TOTAL}, exiting"
    exit 0
fi

# Each task processes 2 geometries sequentially (covers 1240 with 620 tasks)
for OFFSET in 0 1; do
    IDX=$(( SLURM_ARRAY_TASK_ID * 2 + OFFSET ))
    if [ ${IDX} -ge ${TOTAL} ]; then break; fi

    INP=${ALL_INPUTS[${IDX}]}
    DIR=$(dirname ${INP})
    echo "Task ${SLURM_ARRAY_TASK_ID} [${IDX}/${TOTAL}]: ${DIR}"

    cd ${DIR}
    if [ -f sp.out ] && grep -q "FINAL SINGLE POINT ENERGY" sp.out; then
        echo "Already done, skipping."
        continue
    fi

    $(which orca) sp.inp > sp.out 2>&1
    echo "Done."
done
