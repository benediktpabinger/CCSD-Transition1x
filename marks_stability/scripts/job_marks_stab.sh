#!/bin/bash
#SBATCH --job-name=marksstab
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
# ORCA parallelises over MPI processes, not threads -> ntasks, not cpus-per-task
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/s242862/marks_stab/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/marks_stab/slurm_%A_%a.err

# RKS stability of the reference transition states of the Marks benchmark
# (arXiv:2604.00405), at that paper's own level: wB97X-V/def2-TZVP.
#
# Note the functional: wB97X-V, NOT the wB97M-V this project trains against.
# The whole point is to measure the instability on Marks' level, so his
# reference geometries and his functional have to be used together.
#
# Geometries come from thegomeslab/fsm (Marks & Gomes' own FSM repository),
# which carries ts.xyz + chg + mult for every Baker and Sharada reaction at
# the same wB97X-V/def2-TZVP reference level.
#
# Per structure:
#   1) ref.inp    restricted (or unrestricted for open-shell) single point
#                 with EnGrad. Two jobs in one: E_RKS for the depth, and the
#                 gradient norm as a check that the geometry really is a
#                 stationary point at this level -- the repo never states the
#                 level of its ts.xyz, the companion paper does.
#   2) stab.inp   UKS with the stability analysis, restarting onto the broken
#                 solution when an external instability is found. Has to stand
#                 alone: ORCA allows no other RunTyp beside it.
# Open-shell structures get step 1 only; the RKS question does not apply to
# them, but their <S^2> is still logged.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
# Task list and per-process memory are set by the submitting script: the small
# structures run in minutes in a fraction of the memory, and asking for a
# 12 h / 24 GB slot for those only stops the backfill scheduler from ever
# finding a window for them on a partition with ~400 jobs queued.
LIST=${TASKLIST:-$H/marks_stab/tasks.txt}
MAXCORE=${MAXCORE:-2500}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
RID=$(echo "$LINE"  | awk '{print $1}')
CHG=$(echo "$LINE"  | awk '{print $2}')
MULT=$(echo "$LINE" | awk '{print $3}')
GEOM=$(echo "$LINE" | awk '{print $4}')

if [ -z "$RID" ] || [ ! -f "$GEOM" ]; then
  echo "bad task $SLURM_ARRAY_TASK_ID: '$LINE'"; exit 1
fi

W=$H/marks_stab/$RID
mkdir -p $W
cd $W
cp $GEOM start.xyz

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RID  chg=$CHG mult=$MULT  node $SLURM_NODELIST  $(date)"

if [ "$MULT" -eq 1 ]; then REF=RKS; else REF=UKS; fi

# 1) reference single point + gradient
cat > ref.inp <<EOF
! $REF wB97X-V def2-TZVP def2/J RIJCOSX TightSCF DEFGRID3 EnGrad
%pal nprocs 8 end
%maxcore $MAXCORE
%scf
  MaxIter 300
end
* xyzfile $CHG $MULT start.xyz
EOF
$ORCA ref.inp > ref.out 2> ref.err
grep -E 'FINAL SINGLE POINT ENERGY|Expectation value of <S\*\*2>|Norm of the Cartesian gradient' ref.out | tail -4
grep -E 'ORCA TERMINATED NORMALLY|finished by error|SCF NOT CONVERGED' ref.out | tail -2

# 2) stability analysis -- closed-shell singlets only
if [ "$MULT" -eq 1 ]; then
cat > stab.inp <<EOF
! UKS wB97X-V def2-TZVP def2/J RIJCOSX TightSCF DEFGRID3
%pal nprocs 8 end
%maxcore $MAXCORE
%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 300
end
* xyzfile $CHG $MULT start.xyz
EOF
$ORCA stab.inp > stab.out 2> stab.err
grep -E 'is unstable|is stable|UNSTABLE|STABLE|Expectation value of <S\*\*2>|FINAL SINGLE POINT ENERGY' \
     stab.out | tail -8
grep -E 'ORCA TERMINATED NORMALLY|finished by error|SCF NOT CONVERGED' stab.out | tail -2
else
echo "open-shell (mult=$MULT): RKS question not applicable, stability step skipped"
fi

echo "Finished $RID $(date)"
