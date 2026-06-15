#!/bin/bash
#SBATCH --job-name=casscf_optts_mr
#SBATCH --partition=xeon24el8_week
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120GB
#SBATCH --array=0-9
#SBATCH --output=/home/energy/s242862/logs/casscf_optts_mr_%A_%a.log

# CASSCF(AVAS) OptTS + NEVPT2/def2-TZVP for the 10 High MR benchmark reactions.
# AVAS auto-selects the active space at each TS; geometric eigenvector-following
# optimises the saddle point geometry; NEVPT2 SPs computed with projected MOs.
# Starting geometry: ORCA wB97M-V/def2-TZVP NEB transition state.

set -e

REACTIONS=(rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn7945 rxn7937 rxn6196 rxn0346 rxn1150)
RXN=${REACTIONS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/logs

module load Python/3.13.5-GCCcore-14.3.0

echo "Task ${SLURM_ARRAY_TASK_ID}: ${RXN}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export BLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export PYSCF_MAX_MEMORY=100000

python3 /home/energy/s242862/pipeline/mr_casscf_optts.py ${RXN} --n-threads 12
