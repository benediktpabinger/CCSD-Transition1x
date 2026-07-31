#!/bin/bash
#SBATCH --job-name=gradmts
#SBATCH --partition=xeon24el8
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/grad_model_ts/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/grad_model_ts/slurm_%A_%a.err

# RKS and broken-symmetry UKS gradients at the model-predicted TS geometries.
# 18 externally unstable reactions x 4 geometry sources (RKS-ref, UMA-S, UMA-M,
# eSEN) = 72 calculations, all at wB97M-V/def2-TZVP (PySCF, grids 3, 1e-10).
# The RKS-ref rows are recomputed here rather than reused, so every row in the
# final table is produced by identical code.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

H=/home/energy/s242862
mkdir -p $H/grad_model_ts

RXNS=(rxn4518 rxn7949 rxn8832 rxn1320 rxn8837 rxn0894 rxn4522 rxn5691 rxn0346
      rxn1147 rxn7957 rxn1283 rxn3107 rxn8885 rxn8827 rxn4113 rxn7060 rxn6196)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  start $(date)"
python3 $H/pipeline/grad_at_model_ts.py $RXN
RC=$?
echo "rc=$RC  finished $(date)"
exit $RC
