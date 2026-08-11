#!/bin/bash
#SBATCH --job-name=ircfreq
#SBATCH --partition=xeon24el8
#SBATCH --array=0-3
#SBATCH --nodes=1
# ORCA parallelises over MPI processes, not threads -> ntasks, not cpus-per-task
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=/home/energy/s242862/orca_irc/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_irc/slurm_%A_%a.err

# Stage 1 and 2 of the ORCA IRC chain: broken-symmetry single point, then a
# numerical frequency calculation. The IRC itself is a separate job, because
# this one has to answer two questions first and both can kill the plan.
#
#   1. Is our geometry stationary in ORCA's world?  It was optimised on the
#      PySCF surface, with a different grid and a different VV10 implementation.
#      A large gradient here means it is not a saddle for ORCA and the IRC would
#      start from a point that is not on any path. Stage 1b prints it -- it has
#      to be its own run, because ORCA allows no RunTyp but SinglePoint next to
#      a stability analysis.
#
#   2. Does the broken-symmetry solution survive the 6N displacements?  This is
#      what destroyed the BS-NEB: `BrokenSym` re-derives the broken guess at
#      every SCF and only 5 of 11 <S**2> profiles came out coherent. Here the
#      solution is made once with STABPerform and then read with MORead, so
#      every displaced point starts from the same broken orbitals. If <S**2>
#      still collapses on some displacements, the Hessian is contaminated and
#      the whole ORCA route is dead. The trace below shows it.
#
# It also delivers something we have never had: an independent check of the
# PySCF numerical Hessians. Two codes, two grids, two implementations of the
# same numerical differentiation. If ORCA reproduces the imaginary frequency,
# the PySCF Hessians behind every stage-2 and stage-3 verdict are validated.
#
#   task  reaction  structure  our PySCF imaginary frequency
#   0     rxn1147   ours       591 cm-1
#   1     rxn1147   UMA-S      591-ish, from freq_at_model
#   2     rxn7957   ours       677 cm-1
#   3     rxn7957   UMA-M      from freq_at_model
#
# NumHess/NumFreq is mandatory, not a preference: Calc_Hess alone takes the
# analytic CP-SCF route, which dies with "The CPSCF equations can not yet handle
# non-local correlation" for wB97M-V. VV10 has no analytic second derivatives in
# ORCA 5.0.4 either.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862

RXNS=(rxn1147 rxn1147 rxn7957 rxn7957)
SRCS=(ours    UMA-S   ours    UMA-M)
GEOM=($H/bs_tsopt_batch/rxn1147/ts_opt.xyz \
      $H/uma_neb_results/rxn1147/transition_state.xyz \
      $H/bs_tsopt_v2/rxn7957/ts_opt.xyz \
      $H/uma_m_neb_results/rxn7957/transition_state.xyz)

RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
SRC=${SRCS[$SLURM_ARRAY_TASK_ID]}
G=${GEOM[$SLURM_ARRAY_TASK_ID]}

W=$H/orca_irc/${RXN}_${SRC}
mkdir -p $W
cd $W

ORCA=$(which orca)     # full path is required for ORCA's MPI startup
echo "Task $SLURM_ARRAY_TASK_ID: $RXN $SRC  node $SLURM_NODELIST  $(date)"
echo "orca: $ORCA"
echo "geometry: $G"

if [ ! -f "$G" ]; then echo "MISSING GEOMETRY"; exit 1; fi
cp $G start.xyz
head -1 start.xyz

# ---------------------------------------------------------------------------
# Stage 1a: broken-symmetry single point.
# The stability analysis has to stand alone: ORCA refuses any RunTyp other than
# SinglePoint alongside STABPerform ("Only RunTyp == SinglePoint possible with
# Stability Analysis"), so the gradient cannot be asked for in the same input.
# ---------------------------------------------------------------------------
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

S2=$(grep 'Expectation value of <S\*\*2>' bs_sp.out | tail -1 | awk '{print $NF}')
echo "S2_after_stage1a = $S2"

if [ ! -f bs_sp.gbw ]; then echo "NO GBW - stopping"; exit 1; fi
cp bs_sp.gbw bs_start.gbw

# ---------------------------------------------------------------------------
# Stage 1b: gradient at the same geometry, reading those orbitals.
# This is the first question: our geometry was optimised on the PySCF surface,
# so is it stationary for ORCA at all?
# ---------------------------------------------------------------------------
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
echo "--- Cartesian gradient at the start geometry ---"
awk '/CARTESIAN GRADIENT/{f=1} f{print} /Difference to translation invariance/{if(f)exit}' \
    engrad.out | tail -20
echo "--- largest component ---"
awk '/CARTESIAN GRADIENT/{f=1;next} f&&NF==6{v=$4<0?-$4:$4; if(v>m)m=v;
     v=$5<0?-$5:$5; if(v>m)m=v; v=$6<0?-$6:$6; if(v>m)m=v}
     /Difference to translation/{if(f){printf "  max |dE/dx| = %.6f Eh/Bohr = %.4f eV/A\n", m, m*51.42208; exit}}' \
    engrad.out

# ---------------------------------------------------------------------------
# Stage 2: numerical frequencies, reading the broken-symmetry orbitals.
# No STABPerform here -- redoing the stability analysis at every displacement
# would be both ruinous and free to jump to a different solution.
# ---------------------------------------------------------------------------
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
echo "--- <S**2> over all displaced points ---"
grep 'Expectation value of <S\*\*2>' numfreq.out | awk '{print $NF}' > s2_trace.txt
NS=$(wc -l < s2_trace.txt)
echo "  n = $NS SCF solutions"
sort -g s2_trace.txt | head -3 | sed 's/^/  min /'
sort -g s2_trace.txt | tail -3 | sed 's/^/  max /'
awk '{s+=$1; n++} END{if(n)printf "  mean %.4f\n", s/n}' s2_trace.txt
awk '$1 < 0.3 {c++} END{printf "  collapsed below 0.3: %d\n", c+0}' s2_trace.txt

echo ""
echo "--- did it finish ---"
grep -E 'ORCA TERMINATED NORMALLY|ORCA finished by error' numfreq.out | tail -3
ls -la numfreq.hess 2>/dev/null && echo "hessian: $W/numfreq.hess"
echo "Finished $(date)"
