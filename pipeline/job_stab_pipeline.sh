#!/bin/bash
#SBATCH --job-name=stabpipe
#SBATCH --partition=xeon24el8
#SBATCH --array=0-44
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=36:00:00
#SBATCH --output=/home/energy/s242862/stab_pipeline/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/stab_pipeline/slurm_%A_%a.err

# Complete stability pipeline over the full benchmark:
#   45 reactions (top-26 by N_FOD + mid-10 + low-10; rxn0896 is rank 11 and in
#   both top-26 and mid, hence 45 not 46)
#   x 4 geometry sources (RKS-ref, UMA-S, UMA-M, eSEN) = 180 calculations
# Per calculation: RKS + gradient -> RKS stability -> broken symmetry (Route 1,
# Route 2 fallback) + BS gradient -> stability OF the BS solution -> follow an
# internal instability once. Orbitals saved. No optimisation.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

H=/home/energy/s242862
mkdir -p $H/stab_pipeline

RXNS=(rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn7945 rxn7937 rxn6196 rxn0346 rxn1150 rxn0896 rxn4518 rxn3107 rxn8837 rxn7060 rxn5691 rxn1283 rxn8827 rxn4522 rxn7936 rxn1147 rxn0894 rxn0101 rxn10005 rxn10054 rxn7957 rxn1154 rxn5690 rxn4513 rxn7955 rxn4519 rxn4500 rxn2553 rxn8829 rxn1155 rxn9246 rxn4498 rxn1061 rxn4003 rxn4004 rxn4063 rxn4114 rxn4060 rxn1961 rxn1962)
GRPS=(high high high high high high high high high high high high high high high high high high high high high high high high high high mid mid mid mid mid mid mid mid mid low low low low low low low low low low)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
GRP=${GRPS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: $RXN ($GRP)  node $SLURM_NODELIST  $(date)"
python3 $H/pipeline/stability_pipeline.py $RXN
RC=$?
echo "rc=$RC  $(date)"
exit $RC
