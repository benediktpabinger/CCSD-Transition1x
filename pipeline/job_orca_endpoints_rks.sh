#!/bin/bash
#SBATCH --job-name=eprks
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=/home/energy/s242862/orca_endpoint/rks_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_endpoint/rks_%A_%a.err

# The restricted energy at each endpoint, so the size of the effect can be
# stated rather than only its existence.
#
# STABRestartUHFifUnstable rotates into the broken solution and writes only the
# energy after the restart, so the stability run alone says an endpoint is
# unstable but not by how much. <S^2> = 0.10 at rxn7937 might be worth a few
# meV or a few hundred, and the difference decides whether this matters for the
# reported barriers or is a curiosity.
#
# Run at every endpoint, not only the flagged ones: dE_BS = 0 where the
# restricted solution is already the ground state is the control, and it costs
# a few minutes per structure.
#
# Plain RKS, no stability analysis, no gradient.

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
cp $GEOM start.xyz

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $LABEL RKS  node $SLURM_NODELIST  $(date)"

cat > rks.inp <<'EOF'
! RKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF
%pal nprocs 12 end
%maxcore 3000
%scf
  MaxIter 300
end
* xyzfile 0 1 start.xyz
EOF

$ORCA rks.inp > rks.out 2> rks.err
echo "rc=$?"
grep -E 'FINAL SINGLE POINT ENERGY|ORCA TERMINATED NORMALLY|finished by error' \
     rks.out | tail -3
echo "Finished $(date)"
