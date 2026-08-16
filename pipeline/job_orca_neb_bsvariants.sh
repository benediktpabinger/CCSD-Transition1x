#!/bin/bash
#SBATCH --job-name=nebbsvar
#SBATCH --partition=xeon24el8
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_neb_bsvar/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_neb_bsvar/slurm_%A_%a.err

# Two other ways of constructing the broken guess, since the guess is where
# the failure sits.
#
# The verfuegbar/genommen comparison showed, for the first time on both
# quantities at the same point, that at rxn8827 the top image carries a broken
# solution with <S^2> 0.762 and the band ran restricted there.  The images that
# get missed are the shallow ones at the edges of the diradical region -- 0.6
# to 0.8, while the ones that are found sit at 0.9 to 1.06.
#
# Neither variant here repairs the propagation; both are still re-derived at
# every SCF.  What they change is how the guess is built, and therefore which
# images it succeeds on:
#
#   b22      BrokenSym 2,2 instead of 1,1 -- decouple two pairs instead of one.
#            A weakly broken point may need a different flip count to be found
#            at all.
#   rot      Rotate {HOMO,LUMO,90} instead of BrokenSym -- mix the frontier
#            orbitals directly rather than swapping spins on atoms.  A
#            different corner of orbital space, aimed at the same solution.
#
# Everything else is identical to the cheap baseline, so a difference in the
# retroactive band measurement is attributable to the guess and nothing else.
#
#   bs_uks_neb_cheap/<rxn>     BrokenSym 1,1   reference
#   bs_uks_neb_bsvar/b22_<rxn> BrokenSym 2,2
#   bs_uks_neb_bsvar/rot_<rxn> Rotate
#
# rxn8827  the case with a restricted top over an available broken solution
# rxn1320  the production band broke nowhere; the cheap one took all four
# rxn8837  cheap band took 4 of 6 available.  CONTROL -- must not get worse.
#
# What refutes the idea: if both variants miss the same images as 1,1 does,
# the guess construction is not the lever and only propagation is left, which
# ORCA cannot do -- see the three segfaulted attempts at feeding orbitals in.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
CFG=(b22:rxn8827 b22:rxn1320 b22:rxn8837 rot:rxn8827 rot:rxn1320 rot:rxn8837)
ENTRY=${CFG[$SLURM_ARRAY_TASK_ID]}
VAR=${ENTRY%%:*}
RXN=${ENTRY##*:}

W=$H/bs_uks_neb_bsvar/${VAR}_${RXN}
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $VAR auf $RXN  node $SLURM_NODELIST  $(date)"

cp $H/orca_neb_results/$RXN/reactant.xyz reactant.xyz
cp $H/orca_neb_results/$RXN/product.xyz product.xyz

if [ "$VAR" = "b22" ]; then
  GUESS="  BrokenSym 2,2"
else
  # Rotate takes orbital INDICES, not labels -- "Rotate {HOMO, LUMO, ...}"
  # is rejected with "A number was expected".  The indices follow from the
  # electron count, so they are derived here rather than written down and
  # silently wrong for the other formula in the set.
  module load Python/3.13.5-GCCcore-14.3.0 2>/dev/null
  module load ASE 2>/dev/null
  HOMO=$(python3 -c "
Z = dict(H=1, C=6, N=7, O=8, F=9, S=16)
L = open('$W/reactant.xyz').read().split(chr(10))
n = int(L[0].split()[0])
e = sum(Z[l.split()[0]] for l in L[2:2+n] if l.split())
print(e // 2 - 1)")
  LUMO=$((HOMO + 1))
  echo "  Elektronenzaehlung: HOMO $HOMO, LUMO $LUMO"
  # 90 degrees is a full HOMO/LUMO mix in the alpha set only, the standard way
  # of building an open-shell singlet guess without BrokenSym
  GUESS="  Rotate {$HOMO, $LUMO, 90, 1, 1} end"
fi
echo "Guess-Konstruktion: $GUESS"

cat > neb.inp <<EOF
! UKS wB97X 6-31G(d) NEB-TS TightSCF SlowConv

%pal
  nprocs 8
end

%maxcore 3500

%scf
$GUESS
  MaxIter 500
end

%neb
  Product "$W/product.xyz"
  NImages 8
  MaxIter 500
  Preopt true
  PrintLevel 3
end

* xyzfile 0 1 $W/reactant.xyz
EOF

$ORCA neb.inp > neb.out 2> neb.err
RC=$?
echo "rc=$RC"

echo ""
echo "--- hat ORCA die Eingabe angenommen ---"
grep -iE "unrecognized|unknown|not implemented|error|aborting" neb.out | head -4
echo "  LBFGS-Zeilen: $(grep -cE '^ +LBFGS' neb.out)"
if [ "$(grep -cE '^ +LBFGS' neb.out)" = "0" ]; then
  echo "  ABBRUCH-VERDACHT: keine einzige Banditeration. Wenn oben eine"
  echo "  Meldung ueber ein unbekanntes Schluesselwort steht, kennt ORCA"
  echo "  diese Konstruktion nicht und die Variante faellt aus."
fi

echo ""
echo "--- Ausgang ---"
grep -E 'THE NEB OPTIMIZATION HAS CONVERGED|HURRAY|ORCA TERMINATED NORMALLY|kill-11' \
     neb.out | tail -3
echo "  Bildorbitale: $(ls neb_im*.gbw 2>/dev/null | wc -l) von 10"

echo ""
echo "  Die Bandphase steht nicht im Log. Auswertung ueber neb_im*.gbw gegen"
echo "  bs_uks_neb_perimage/$RXN/s2_before.txt -- dieselben Geometrien, also"
echo "  ist verfuegbar gegen genommen direkt vergleichbar."
echo "Finished $(date)"
