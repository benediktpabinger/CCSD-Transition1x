#!/bin/bash
#SBATCH --job-name=uksstab
#SBATCH --partition=xeon24el8
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/uks_stab/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/uks_stab/slurm_%A_%a.err

# Stability analysis of the converged broken-symmetry UKS solutions from job
# 10687985. That job kept no orbitals (no chkfile, no mo_coeff dump), so each
# BS solution is re-converged by the identical Route-1 path first; the
# re-converged dE_BS / <S^2> are compared against the stored values.
# 63 of 72 rows qualify (dE_BS < 0, <S^2> > 0.05).
# Orbitals ARE saved this time: bs_<tag>.npz and bs2_<tag>.npz.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

H=/home/energy/s242862
mkdir -p $H/uks_stab

RXNS=(rxn4518 rxn7949 rxn8832 rxn1320 rxn8837 rxn0894 rxn4522 rxn5691 rxn0346
      rxn1147 rxn7957 rxn1283 rxn3107 rxn8885 rxn8827 rxn4113 rxn7060 rxn6196)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  start $(date)"
python3 $H/pipeline/uks_stability.py $RXN
RC=$?
echo "rc=$RC  finished $(date)"
exit $RC
