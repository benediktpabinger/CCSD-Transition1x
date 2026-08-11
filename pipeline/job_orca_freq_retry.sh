#!/bin/bash
#SBATCH --job-name=freqretry
#SBATCH --partition=xeon24el8
#SBATCH --array=0-1
#SBATCH --nodes=1
# ORCA parallelises over MPI processes, not threads -> ntasks, not cpus-per-task
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=/home/energy/s242862/orca_irc/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_irc/slurm_%A_%a.err

# The two structures the repair runs produced and that have no frequency yet.
# Both were started from a model geometry after the optimisation from the
# reference had failed, and both converged where the earlier attempts had not.
#
#   rxn8885   -323.311979  <S^2> 1.028  converged, 277 steps
#             against our current structure at -323.296344, <S^2> 0.153, flagged
#             BS_LOST: the new one lies 425 meV lower and is 1.475 A away. It is
#             the second basin that was suspected. If it carries one imaginary
#             frequency on the reactive mode it replaces our current saddle --
#             and UMA-S, which sits 342 meV below the old point, would then be
#             84 meV above the new one, reversing that row of the verdict table.
#
#   rxn1283   -322.349375  <S^2> 1.004  converged, 139 steps
#             the first converged saddle for this reaction at all. It would
#             close one of the four unresolved cases.
#
# Same three stages as the IRC preparation: stability alone (ORCA allows no
# other RunTyp beside it), then the gradient, then NumFreq -- all reading the
# broken-symmetry orbitals made in stage 1a.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862

RXNS=(rxn8885 rxn1283)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
G=$H/bs_tsopt_retry/$RXN/ts_opt.xyz

W=$H/orca_irc/${RXN}_retry
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN retry  node $SLURM_NODELIST  $(date)"
echo "orca: $ORCA"
echo "geometry: $G"
if [ ! -f "$G" ]; then echo "MISSING GEOMETRY"; exit 1; fi
cp $G start.xyz
head -2 start.xyz

cat > bs_sp.inp <<'EOF'
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

echo ""
echo "=== Stage 1a: BS single point ==="
$ORCA bs_sp.inp > bs_sp.out 2> bs_sp.err
echo "stage1a rc=$?"
grep -E 'FINAL SINGLE POINT ENERGY|Expectation value of <S\*\*2>|is unstable|is stable|ORCA finished by error' bs_sp.out | tail -8
if [ ! -f bs_sp.gbw ]; then echo "NO GBW - stopping"; exit 1; fi
cp bs_sp.gbw bs_start.gbw

cat > engrad.inp <<'EOF'
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad MORead
%moinp "bs_start.gbw"
%pal nprocs 12 end
%maxcore 3000
%scf
  MaxIter 300
end
* xyzfile 0 1 start.xyz
EOF

echo ""
echo "=== Stage 1b: gradient ==="
$ORCA engrad.inp > engrad.out 2> engrad.err
echo "stage1b rc=$?"
grep 'Expectation value of <S\*\*2>' engrad.out | tail -1
awk '/CARTESIAN GRADIENT/{f=1;next} f&&NF>=6{
      for(i=4;i<=6;i++){v=$i<0?-$i:$i; if(v>m)m=v}}
     f&&/Difference to translation/{printf "  max |dE/dx| = %.6f Eh/Bohr = %.4f eV/A\n", m, m*51.42208; exit}' \
    engrad.out

cat > numfreq.inp <<'EOF'
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF NumFreq MORead
%moinp "bs_start.gbw"
%pal nprocs 12 end
%maxcore 3000
%scf
  MaxIter 300
end
%freq
  CentralDiff true
  Increment 0.005
end
* xyzfile 0 1 start.xyz
EOF

echo ""
echo "=== Stage 2: NumFreq ==="
$ORCA numfreq.inp > numfreq.out 2> numfreq.err
echo "stage2 rc=$?"
echo ""
echo "--- frequencies (lowest 12) ---"
awk '/VIBRATIONAL FREQUENCIES/{f=1} f{print} /NORMAL MODES/{if(f)exit}' \
    numfreq.out | head -24
echo ""
echo "--- S**2 of the last displaced point ---"
grep 'Expectation value of <S\*\*2>' numfreq.lastscf 2>/dev/null | tail -1
grep -E 'ORCA TERMINATED NORMALLY|ORCA finished by error' numfreq.out | tail -2
ls -la numfreq.hess 2>/dev/null && echo "hessian: $W/numfreq.hess"
echo "Finished $(date)"
