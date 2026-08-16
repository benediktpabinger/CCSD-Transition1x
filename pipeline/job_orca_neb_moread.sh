#!/bin/bash
#SBATCH --job-name=nebmoread
#SBATCH --partition=xeon24el8
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_neb_moread/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_neb_moread/slurm_%A_%a.err

# THE TEST OF THE DIAGNOSIS, not another baseline.
#
# Every band run so far -- 22 at wB97M-V/def2-TZVP and 2 at wB97X/6-31G(d) --
# stayed at <S^2> = 0 through the whole band phase.  The stated reason is that
# `BrokenSym 1,1` is stateless: it re-derives the broken guess at every single
# SCF instead of inheriting it, and the re-derivation lands back in the
# restricted minimum.
#
# If that reason is right, then removing BrokenSym and handing the band a
# broken set of orbitals through MORead should change the outcome, because
# ORCA's ordinary guess propagation then carries the state instead.  If the
# band still collapses, the diagnosis is wrong or incomplete and the fix has to
# be a different one -- which is worth knowing before anyone writes a custom
# NEB driver for it.
#
# Two stages, because the guess has to exist before it can be read:
#
#   0  single point at the reference TS geometry, same cheap level, with the
#      stability analysis, so the .gbw that comes out is genuinely broken
#   1  NEB-TS reading that .gbw, no BrokenSym anywhere
#
# Run on rxn8827 and rxn8837 because those two are the only reactions with a
# complete baseline at BOTH levels:
#
#   rxn8827  production: band restricted, TS-opt never broke
#            cheap:      band restricted, TS-opt broke (S^2 1.051)
#   rxn8837  production: band restricted, TS-opt broke (1.039)
#            cheap:      band restricted, TS-opt broke (1.076)
#
# So for both, "the band held the breaking" would be a new outcome that no run
# at either level has produced.  That is what makes this falsifiable.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn8827 rxn8837)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

W=$H/bs_uks_neb_moread/$RXN
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

cp $H/orca_neb_results/$RXN/reactant.xyz reactant.xyz
cp $H/orca_neb_results/$RXN/product.xyz product.xyz
cp $H/orca_neb_results/$RXN/transition_state.xyz ts_guess.xyz

# ---------------------------------------------------------------- stage 0
# The broken guess.  STABPerform cannot run alongside anything but a single
# point -- that restriction is why the whole pipeline is staged like this.
cat > bsguess.inp <<EOF
! UKS wB97X 6-31G(d) SP TightSCF SlowConv

%pal
  nprocs 8
end

%maxcore 3500

%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 500
end

* xyzfile 0 1 $W/ts_guess.xyz
EOF

if [ -s bsguess.gbw ] && grep -q 'ORCA TERMINATED NORMALLY' bsguess.out 2>/dev/null; then
  echo "stage 0 uebersprungen, bsguess.gbw existiert bereits"
else
  $ORCA bsguess.inp > bsguess.out 2> bsguess.err
  echo "stage 0 rc=$?"
fi
grep -E 'Expectation value of <S\*\*2>|UHF/UKS wavefunction is unstable|is stable' \
     bsguess.out | tail -4

S2=$(grep 'Expectation value of <S\*\*2>' bsguess.out | awk '{print $NF}' | tail -1)
echo "guess <S^2> = $S2"

# A restricted guess would make the whole test vacuous, so refuse to continue.
if awk -v s="$S2" 'BEGIN{exit !(s < 0.3)}'; then
  echo "ABORT: guess is not broken (<S^2> = $S2); nothing to propagate."
  exit 1
fi

# ------------------------------------------------------------- stage 0b
# MORead is rejected outright by the NEB module:
#
#   WARNING: NEB Calculation
#            MORead requested. This is not implemented.
#            Please instead use the NEB_RESTART_GBWNAME feature.
#
# That feature is better suited than MORead would have been.  It takes a
# BASENAME, not a file, and reads one set of orbitals PER IMAGE:
#
#   Basename of existing gbw files  ....  <basename>_im{NIm}.gbw
#
# So every image starts from broken orbitals, not just the first.  The
# endpoints get them too; they will collapse to the restricted solution there,
# which is correct -- beyond the Coulson-Fischer point the broken solution does
# not exist, and all 45 reactants are closed-shell.
#
# NImages counts the intermediate images, so the path carries NImages+2 points.
# A couple of spares are written; ORCA reads only what it needs.
for i in $(seq 0 11); do
  cp bsguess.gbw guess_im${i}.gbw
done
echo "stage 0b: $(ls guess_im*.gbw | wc -l) Startorbital-Dateien"

# ---------------------------------------------------------------- stage 1
# Identical to the baseline except the two things under test: no BrokenSym,
# and per-image broken orbitals handed in.
cat > neb.inp <<EOF
! UKS wB97X 6-31G(d) NEB-TS TightSCF SlowConv

%pal
  nprocs 8
end

%maxcore 3500

%scf
  MaxIter 500
end

%neb
  Product "$W/product.xyz"
  NImages 8
  MaxIter 500
  Preopt true
  PrintLevel 3
  NEB_Restart_GBWName "$W/guess"
end

* xyzfile 0 1 $W/reactant.xyz
EOF

$ORCA neb.inp > neb.out 2> neb.err
echo "stage 1 rc=$?"

echo ""
echo "--- did it converge ---"
grep -E 'THE NEB OPTIMIZATION HAS CONVERGED|HURRAY|ORCA TERMINATED NORMALLY|finished by error' \
     neb.out | tail -4

echo ""
echo "--- the number this run exists for: <S^2> split by phase ---"
awk '/NEB OPTIMIZATION HAS CONVERGED|TS OPTIMIZATION/ {conv=1}
     /Expectation value of <S\*\*2>/ {
       v=$NF
       if (v > 1.8) next
       if (conv) {a++; if (v>am) am=v; if (v>0.3) ac++}
       else      {b++; if (v>bm) bm=v; if (v>0.3) bc++}
     }
     END {
       printf "  band phase   n=%d  max=%.3f  >0.3: %d\n", b+0, bm+0, bc+0
       printf "  after band   n=%d  max=%.3f  >0.3: %d\n", a+0, am+0, ac+0
     }' neb.out

echo ""
echo "  baseline for comparison, same reaction, same level, WITH BrokenSym:"
awk '/NEB OPTIMIZATION HAS CONVERGED|TS OPTIMIZATION/ {conv=1}
     /Expectation value of <S\*\*2>/ {
       v=$NF
       if (v > 1.8) next
       if (conv) {a++; if (v>am) am=v} else {b++; if (v>bm) bm=v}
     }
     END {printf "  band phase   n=%d  max=%.3f\n  after band   n=%d  max=%.3f\n",
          b+0, bm+0, a+0, am+0}' $H/bs_uks_neb_cheap/$RXN/neb.out 2>/dev/null

echo "Finished $(date)"
