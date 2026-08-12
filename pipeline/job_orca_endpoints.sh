#!/bin/bash
#SBATCH --job-name=endpoint
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
# ORCA parallelises over MPI processes, not threads -> ntasks, not cpus-per-task
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/s242862/orca_endpoint/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_endpoint/slurm_%A_%a.err

# Is the restricted solution stable at the reactant and product?
#
# The stability analysis has been run at 180 geometries and every one of them
# was a transition state. Nobody has ever looked at the endpoints, and the
# endpoints are what defines the path: a NEB interpolates between them, and the
# barrier is E(TS) minus E(reactant). If the restricted solution is unstable at
# a minimum, then that geometry was optimised on the wrong surface, the barrier
# has the wrong zero, and the reaction being modelled is not quite the reaction
# on the ground-state surface.
#
# The expected answer is that they are all stable. Reactants and products are
# ordinarily closed-shell minima and the symmetry breaking appears only where a
# bond is half broken; the broken-symmetry NEB bands support this directly,
# with <S^2> going to 0.000 at both ends. But expected is not checked, and this
# is the premise everything else rests on.
#
# One single point per structure with STABPerform. No gradient, no Hessian --
# the stability analysis alone answers it, and ORCA allows no other RunTyp
# beside it anyway.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
LIST=$H/endpoint_tasks.txt

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
LABEL=$(echo "$LINE" | awk '{print $1}')
GEOM=$(echo "$LINE" | awk '{print $2}')
if [ -z "$LABEL" ] || [ ! -f "$GEOM" ]; then
  echo "bad task $SLURM_ARRAY_TASK_ID: '$LINE'"; exit 1
fi

W=$H/orca_endpoint/$LABEL
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $LABEL  node $SLURM_NODELIST  $(date)"
echo "geometry: $GEOM"
cp $GEOM start.xyz

cat > sp.inp <<'EOF'
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF
%pal nprocs 12 end
%maxcore 3000
%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 300
end
* xyzfile 0 1 start.xyz
EOF

$ORCA sp.inp > sp.out 2> sp.err
echo "rc=$?"
echo ""
echo "--- stability ---"
grep -E 'is unstable|is stable|instability|STABILITY' sp.out | tail -6
echo "--- final ---"
grep -E 'FINAL SINGLE POINT ENERGY|Expectation value of <S\*\*2>' sp.out | tail -4
grep -E 'ORCA TERMINATED NORMALLY|finished by error' sp.out | tail -1
echo "Finished $(date)"
