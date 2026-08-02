#!/bin/bash
#SBATCH --job-name=bsfreq
#SBATCH --partition=xeon24el8
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_freq/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_freq/slurm_%A_%a.err

# Numerical BS-UKS frequencies at the 9 usable broken-symmetry TS geometries.
# One imaginary frequency = genuine first-order saddle point.
# 6N gradients per structure, ~90 s each -> ~1.7 h for 11 atoms.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn0346 rxn8827 rxn0894 rxn1147 rxn5691 rxn7949 rxn4518 rxn1320 rxn8837)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_freq
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_freq.py $RXN
RC=$?
echo "rc=$RC $(date)"
exit $RC
