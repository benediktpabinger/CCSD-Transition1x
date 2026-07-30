#!/bin/bash
#SBATCH --job-name=neb25
#SBATCH --partition=xeon24el8
#SBATCH --array=0-44
#SBATCH --nodes=1
# ORCA parallelises over MPI ranks -> ntasks, not cpus-per-task
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/orca_neb_omol25/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_neb_omol25/slurm_%A_%a.err

# Full NEB recomputation at the OMol25 level of theory.
#   set: top-26 by N_FOD + mid-10 + low-10 (union = 45; rxn0896 is rank 11
#        and belongs to both top-26 and mid, hence 45 not 46)
#   level: wB97M-V/def2-TZVPD, RIJ+COSX, TightSCF, DEFGRID3,
#          thresh 1e-12, tcut 1e-13, ORCA 5.0.4
#   endpoints ARE re-relaxed at this level (otherwise every barrier mixes
#   two levels of theory)
# Validated on rxn1320 (job 10686096): CI-NEB converged, 4 h 10 min,
# TS RMSD vs the def2-TZVP reference 0.0134 A, barrier shift +8.2 meV.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

H=/home/energy/s242862
RXNS=(rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn7945 rxn7937 rxn6196 rxn0346 rxn1150 rxn0896 rxn4518 rxn3107 rxn8837 rxn7060 rxn5691 rxn1283 rxn8827 rxn4522 rxn7936 rxn1147 rxn0894 rxn0101 rxn10005 rxn10054 rxn7957 rxn1154 rxn5690 rxn4513 rxn7955 rxn4519 rxn4500 rxn2553 rxn8829 rxn1155 rxn9246 rxn4498 rxn1061 rxn4003 rxn4004 rxn4063 rxn4114 rxn4060 rxn1961 rxn1962)
GRPS=(high high high high high high high high high high high high high high high high high high high high high high high high high high mid mid mid mid mid mid mid mid mid low low low low low low low low low low)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
GRP=${GRPS[$SLURM_ARRAY_TASK_ID]}

OUT=$H/orca_neb_omol25/$RXN
mkdir -p $OUT
export OMP_NUM_THREADS=1

echo "Task $SLURM_ARRAY_TASK_ID: $RXN ($GRP)  node $SLURM_NODELIST  start $(date)"

# already finished (e.g. the rxn1320 validation run)? do not redo
if [ -f "$OUT/converged" ]; then
  echo "SKIP: $OUT/converged exists"
  exit 0
fi

python3 $H/pipeline/orca_neb_omol25.py \
    --h5file   $H/data/Transition1x.h5 \
    --reaction $RXN \
    --split    test \
    --output   $OUT \
    --orca-cmd $(which orca) \
    --nprocs   12
RC=$?

echo "rc=$RC  finished $(date)"
[ -f $OUT/converged ] && echo "CONVERGED" || echo "NOT CONVERGED"
tail -2 $OUT/neb.log 2>/dev/null
exit $RC
