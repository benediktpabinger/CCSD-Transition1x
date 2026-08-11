#!/bin/bash
#SBATCH --job-name=orcairc
#SBATCH --partition=xeon24el8
#SBATCH --array=0-3
#SBATCH --nodes=1
# ORCA parallelises over MPI processes, not threads -> ntasks, not cpus-per-task
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=/home/energy/s242862/orca_irc/irc_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_irc/irc_%A_%a.err

# The intrinsic reaction coordinate from all four contested saddles.
#
# This is the point of the whole exercise. rxn1147 and rxn7957 were decided by a
# judgement -- "this reactive bond already sits at its normal length, so the
# reaction is finished at that point" -- and that judgement is the last
# unautomated step in the argument. The IRC removes it: if a structure lies past
# the transition state, the path descending from the other saddle runs through
# it, and nobody has to decide what counts as a finished bond.
#
# Why ORCA and not the hand-written PySCF version: the IRC is defined as the
# steepest-descent path in mass-weighted coordinates, and writing that down with
# an Euler integrator is easy. Doing it without drifting off the path in a
# curved valley is not, and an unvalidated integrator is what the withdrawn
# endpoint test already was. ORCA's is tested code.
#
# Prerequisites, all now satisfied by job_orca_irc_freq.sh:
#   - the geometry is stationary for ORCA (0.010 eV/A at our saddles)
#   - the broken symmetry survives the displacements, shown by the Hessian
#     matching PySCF's to a mode overlap of 0.9991
#   - numfreq.hess exists at each of the four structures
#
# InitHess read avoids recomputing a Hessian that already took 25 minutes each.
# Direction both walks forwards and backwards from the same saddle.
#
# 60 steps per direction, not the 120 first written down: the rival structure
# sits 0.26 A from our saddle at rxn1147 and 0.20 A at rxn7957, so the path
# either runs through it within the first handful of steps or it does not. The
# remaining steps would only be spent walking the valley out, which does not
# enter the question.
# Follow_CoordType cartesian keeps the path in the coordinates the bond lengths
# are read off, rather than in ORCA's internal set.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862

RXNS=(rxn1147 rxn1147 rxn7957 rxn7957)
SRCS=(ours    UMA-S   ours    UMA-M)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
SRC=${SRCS[$SLURM_ARRAY_TASK_ID]}

W=$H/orca_irc/${RXN}_${SRC}
cd $W || { echo "no work dir $W"; exit 1; }

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN $SRC  node $SLURM_NODELIST  $(date)"

for f in start.xyz bs_start.gbw numfreq.hess; do
  if [ ! -f $f ]; then echo "MISSING $f - the freq job has to finish first"; exit 1; fi
done
cp numfreq.hess irc_start.hess

cat > irc.inp <<'EOF'
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF IRC MORead
%moinp "bs_start.gbw"
%pal nprocs 12 end
%maxcore 3000
%scf
  MaxIter 300
end
%irc
  MaxIter 60
  PrintLevel 1
  Direction both
  InitHess read
  Hess_Filename "irc_start.hess"
  Follow_CoordType cartesian
  Adapt_Scale_Displ true
end
* xyzfile 0 1 start.xyz
EOF

echo ""
echo "=== IRC ==="
$ORCA irc.inp > irc.out 2> irc.err
echo "irc rc=$?"

echo ""
echo "--- summary tables ---"
awk '/IRC PATH SUMMARY/{f=1} f{print} /ORCA TERMINATED|IRC calculation/{if(f&&NR>1)exit}' \
    irc.out | head -80

echo ""
echo "--- endpoints ---"
for t in IRC_F IRC_B; do
  if [ -f irc_$t.xyz ]; then echo "$t:"; head -2 irc_$t.xyz; fi
done
ls -la irc_IRC*.xyz irc_IRC*trj.xyz 2>/dev/null

echo ""
echo "--- did it finish ---"
grep -E 'ORCA TERMINATED NORMALLY|ORCA finished by error|THE IRC HAS CONVERGED|IRC did not converge' irc.out | tail -4
echo "Finished $(date)"
