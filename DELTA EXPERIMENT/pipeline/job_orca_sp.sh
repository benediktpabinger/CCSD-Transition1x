#!/bin/bash
#SBATCH --job-name=delta_sp
#SBATCH --partition=xeon24el8
#SBATCH --time=1:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=/home/energy/s242862/delta_experiment/logs/sp_%A_%a.log
#SBATCH --array=0-999%40

set -e

module load ORCA/5.0.4-gompi-2023a

SP_BASE=/home/energy/s242862/delta_experiment/orca_sp
mkdir -p /home/energy/s242862/delta_experiment/logs

# Build flat list of all sp.inp files
mapfile -t ALL_INPUTS < <(find ${SP_BASE} -name 'sp.inp' | sort)
TOTAL=${#ALL_INPUTS[@]}

if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL} ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} >= total ${TOTAL}, exiting"
    exit 0
fi

INP=${ALL_INPUTS[${SLURM_ARRAY_TASK_ID}]}
DIR=$(dirname ${INP})

echo "Task ${SLURM_ARRAY_TASK_ID}/${TOTAL}: ${DIR}"

cd ${DIR}
if [ -f sp.out ] && grep -q "FINAL SINGLE POINT ENERGY" sp.out; then
    echo "Already done, skipping."
    exit 0
fi

$(which orca) sp.inp > sp.out 2>&1
echo "Done."
