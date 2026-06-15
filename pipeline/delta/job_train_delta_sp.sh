#!/bin/bash
#SBATCH --job-name=train_delta_sp
#SBATCH --partition=xeon24el8
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --array=0-1999
#SBATCH --output=/home/energy/s242862/logs/train_delta_sp_%A_%a.log

# wB97M-V/def2-TZVP SPs on 20 stratified T1x geometries per reaction (TS-biased).
# 5000 training reactions split across 3 submissions (OFFSET=0, 2000, 4000).
# Submit with: sbatch --export=OFFSET=0 job_train_delta_sp.sh  (reactions 1-2000)
#              sbatch --export=OFFSET=2000 job_train_delta_sp.sh  (reactions 2001-4000)
#              sbatch --export=OFFSET=4000 job_train_delta_sp.sh  (reactions 4001-5000, array 0-999)
# Run sample_train_reactions.py first to generate the reaction list.

set -e

export TMPDIR=/tmp

module load Python/3.13.5-GCCcore-14.3.0
module load ORCA/5.0.4-gompi-2023a
mkdir -p /home/energy/s242862/logs

OFFSET=${OFFSET:-0}
RXNLIST=/home/energy/s242862/ccsd_dataset/train_delta_rxns.txt
RXN=$(sed -n "$((SLURM_ARRAY_TASK_ID + OFFSET + 1))p" ${RXNLIST})

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN}"

python3 /home/energy/s242862/pipeline/delta/train_delta_sp.py ${RXN} --n-images 20 --n-threads 8
