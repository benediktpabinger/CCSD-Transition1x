#!/bin/bash
#SBATCH --job-name=cheapstab
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
# ORCA parallelises over MPI processes, not threads -> ntasks, not cpus-per-task
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/home/energy/s242862/cheap_stab/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/cheap_stab/slurm_%A_%a.err

# Does the instability survive at the level Transition1x was generated at?
#
# The plan is to develop the method -- density propagation along a NEB band, an
# <S^2> watchdog, multi-start -- at wB97X/6-31G(d) rather than at
# wB97M-V/def2-TZVP, because iteration there is far cheaper. Two reasons it is
# cheaper, and the second matters more than the basis:
#
#   6-31G(d) has roughly 110 basis functions against 250 for def2-TZVP
#   wB97X has no VV10 term, so CP-SCF works and Hessians are analytic. That
#   removes the 6N numerical gradients that cost 25-35 minutes per structure
#   and were run over a hundred times.
#
# But a cheap testbed is only a testbed if the phenomenon is present in it. The
# fraction of exact exchange controls the tendency to break symmetry and wB97X
# has a different profile from wB97M-V; the smaller basis moves the
# Coulson-Fischer point as well. So the same reactions need not be unstable.
#
# One single point with a stability analysis at each of the 45 reference
# transition states answers it, and says how deep the breaking goes where it
# does occur. Nothing is built on the cheap level before this is known.
#
# Reference numbers to compare against, at wB97M-V/def2-TZVP:
#   19 of 45 externally unstable, dE_BS from -648.5 to -1.3 meV

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
LIST=$H/cheap_stab_tasks.txt

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
RXN=$(echo "$LINE" | awk '{print $1}')
GEOM=$(echo "$LINE" | awk '{print $2}')
if [ -z "$RXN" ] || [ ! -f "$GEOM" ]; then
  echo "bad task $SLURM_ARRAY_TASK_ID: '$LINE'"; exit 1
fi

W=$H/cheap_stab/$RXN
mkdir -p $W
cd $W
cp $GEOM start.xyz

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

# 1) restricted, for the reference energy of the sheet we are comparing against
cat > rks.inp <<'EOF'
! RKS wB97X 6-31G(d) TightSCF
%pal nprocs 8 end
%maxcore 2500
%scf
  MaxIter 300
end
* xyzfile 0 1 start.xyz
EOF
$ORCA rks.inp > rks.out 2> rks.err
grep 'FINAL SINGLE POINT ENERGY' rks.out | tail -1

# 2) the stability analysis, which rotates into the broken solution if one is
#    lower. Has to stand alone -- ORCA allows no other RunTyp beside it.
cat > stab.inp <<'EOF'
! UKS wB97X 6-31G(d) TightSCF
%pal nprocs 8 end
%maxcore 2500
%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 300
end
* xyzfile 0 1 start.xyz
EOF
$ORCA stab.inp > stab.out 2> stab.err
grep -E 'is unstable|is stable|UNSTABLE|Expectation value of <S\*\*2>|FINAL SINGLE POINT ENERGY' \
     stab.out | tail -6
grep -E 'ORCA TERMINATED NORMALLY|finished by error' stab.out | tail -1
echo "Finished $(date)"
