#!/bin/bash
#SBATCH --job-name=nebci
#SBATCH --partition=xeon24el8
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_nebci/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_nebci/slurm_%A_%a.err

# Stop asking the band to do the saddle optimisation.
#
# Tonight's numbers say the same thing in every run: the band force is at or
# under its criterion while the climbing-image force is not, and the CI
# criterion is an order of magnitude tighter.
#
#   B_rxn8837      Fp 0.048            FCI 0.00169   crit 0.020 / 0.002
#   B_rxn8827      Fp 0.014  under     FCI 0.0176
#   16img rxn8827  Fp 0.030            FCI 0.00224
#
# Four of five runs are held up by the climbing image, not by the path.  And
# the two runs cancelled earlier today spent 122 and 132 iterations oscillating
# in exactly that regime.
#
# NEB-TS tries to converge the saddle inside the band optimisation.  NEB-CI
# stops once the climbing image is identified and leaves it there.  Handing
# that image to a dedicated broken-symmetry TS optimisation does the same work
# with a method built for it -- and it is what the earlier from-model runs did
# successfully, only from a worse starting point.
#
#   1  NEB-CI, same level and path as the baseline, BrokenSym as before
#   2  single point with STABPerform at the climbing image -> broken orbitals
#   3  OptTS + Freq from those orbitals
#
# Stage 3 is analytic here: wB97X carries no VV10, so the Hessian and the
# frequencies come out of the same run.
#
# rxn8827  baseline NEB-TS: band restricted at the top, result on the RKS-TS
# rxn1320  baseline NEB-TS: never converged, no climbing image in 133 steps
# rxn8837  baseline NEB-TS: worked.  CONTROL -- the split must not break it.
#
# What would refute the idea: if NEB-CI also fails to produce a climbing image
# for rxn1320, the problem is the band after all and not the tolerance.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn8827 rxn1320 rxn8837)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

W=$H/bs_uks_nebci/$RXN
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

cp $H/orca_neb_results/$RXN/reactant.xyz reactant.xyz
cp $H/orca_neb_results/$RXN/product.xyz product.xyz

# ------------------------------------------------------------------ 1
cat > neb.inp <<EOF
! UKS wB97X 6-31G(d) NEB-CI TightSCF SlowConv

%pal
  nprocs 8
end

%maxcore 3500

%scf
  BrokenSym 1,1
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
echo "1: rc=$?  LBFGS $(grep -cE '^ +LBFGS' neb.out)"
grep -E 'THE NEB OPTIMIZATION HAS CONVERGED|HURRAY|finished by error|kill-11' \
     neb.out | tail -3

CI=$(ls *NEB-CI_converged.xyz 2>/dev/null | head -1)
if [ -z "$CI" ]; then
  echo "ABBRUCH: kein Climbing Image erzeugt -- das Band ist das Problem,"
  echo "         nicht die Toleranz. Genau das widerlegt die Idee."
  exit 3
fi
echo "Climbing Image: $CI"

# ------------------------------------------------------------------ 2
cat > bs.inp <<EOF
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

* xyzfile 0 1 $W/$CI
EOF

$ORCA bs.inp > bs.out 2> bs.err
S2=$(grep 'Expectation value of <S\*\*2>' bs.out | awk '{print $NF}' | tail -1)
echo "2: <S^2> am Climbing Image = $S2"
if awk -v s="$S2" 'BEGIN{exit !(s < 0.3)}'; then
  echo "   HINWEIS: das Climbing Image traegt keine gebrochene Loesung."
  echo "            Stufe 3 laeuft dann restringiert -- auch das ist ein"
  echo "            Ergebnis, siehe die Gipfel-Analyse in Kapitel 3a."
fi

# ------------------------------------------------------------------ 3
cat > tsopt.inp <<EOF
! UKS wB97X 6-31G(d) OptTS Freq TightSCF SlowConv MORead

%moinp "$W/bs.gbw"

%pal
  nprocs 8
end

%maxcore 3500

%geom
  Calc_Hess true
  MaxIter 200
end

%scf
  MaxIter 500
end

* xyzfile 0 1 $W/$CI
EOF

$ORCA tsopt.inp > tsopt.out 2> tsopt.err
echo "3: rc=$?"

echo ""
echo "--- Ergebnis ---"
echo "  konvergiert : $(grep -c 'THE OPTIMIZATION HAS CONVERGED' tsopt.out)"
echo "  Zyklen      : $(grep -c 'GEOMETRY OPTIMIZATION CYCLE' tsopt.out)"
echo "  <S^2>       : $(grep 'Expectation value of <S\*\*2>' tsopt.out \
      | awk '{print $NF}' | tail -1)"
# only the LAST frequency block -- OptTS prints the initial Hessian as well,
# and mixing them makes a converged saddle look like a higher-order one
python3 - tsopt.out <<'EOF'
import re, sys
t = open(sys.argv[1], errors='replace').read()
i = t.rfind('VIBRATIONAL FREQUENCIES')
if i < 0:
    print('  Frequenzen  : keine')
else:
    fr = [float(m.group(1)) for m in
          re.finditer(r'^\s*\d+:\s+(-?\d+\.\d+)\s+cm', t[i:], re.M)]
    im = [v for v in fr if v < -1.0]
    print('  imaginaer   : %d   %s' % (len(im), ' '.join('%.1f' % v for v in im[:5])))
EOF
echo "Finished $(date)"
