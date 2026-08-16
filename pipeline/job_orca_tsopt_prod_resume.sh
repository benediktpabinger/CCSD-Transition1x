#!/bin/bash
#SBATCH --job-name=tsprod
#SBATCH --partition=xeon24el8
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_nebci_prod/resume_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_nebci_prod/resume_%A_%a.err

# Stage 3 of the production NEB-CI split, restarted with a numerical Hessian.
#
# The first attempt died after one optimisation cycle in all three reactions:
#
#   ORCA_CPSCF: The CPSCF equations can not yet handle non-local correlation
#   ORCA finished by error termination in SCF Hessian
#
# `%geom Calc_Hess true` builds an ANALYTIC starting Hessian, and that needs
# CPSCF, which cannot handle the VV10 term in wB97M-V.  The final frequencies
# were already set to NumFreq; the starting Hessian was not, and it is the one
# that runs first.
#
# The band phases survive -- they took ten to twenty hours each and are not
# repeated.  This picks up the converged climbing images and redoes only the
# saddle optimisation.
#
#   rxn0346  <S^2> at the CI 0.600      old NEB-TS gradient 2.553 eV/A
#   rxn8827                  0.370                          1.074
#   rxn6196                  0.177                          0.683
#
# All three climbing images carry a broken solution, which at this level is not
# a given: the old NEB-TS runs had a restricted top image for rxn8827.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn0346 rxn8827 rxn6196)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

SRC=$H/bs_uks_nebci_prod/$RXN
W=$SRC
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

CI=$(ls $SRC/*NEB-CI_converged.xyz 2>/dev/null | head -1)
if [ -z "$CI" ] || [ ! -s $SRC/bs.gbw ]; then
  echo "ABBRUCH: Climbing Image oder Orbitale fehlen"
  exit 2
fi
echo "Climbing Image: $(basename $CI)"
echo "S2 dort: $(grep 'Expectation value of <S\*\*2>' $SRC/bs.out | awk '{print $NF}' | tail -1)"

cat > tsopt2.inp <<EOI
! UKS wB97M-V def2-TZVP def2/J RIJCOSX OptTS NumFreq TightSCF SlowConv MORead

%moinp "$SRC/bs.gbw"

%pal
  nprocs 12
end

%maxcore 4500

%geom
  Calc_Hess true
  NumHess true
  MaxIter 200
end

%scf
  MaxIter 500
end

* xyzfile 0 1 $CI
EOI

$ORCA tsopt2.inp > tsopt2.out 2> tsopt2.err
echo "rc=$?"

echo ""
echo "--- Ergebnis ---"
echo "  konvergiert : $(grep -c 'THE OPTIMIZATION HAS CONVERGED' tsopt2.out)"
echo "  Zyklen      : $(grep -c 'GEOMETRY OPTIMIZATION CYCLE' tsopt2.out)"
echo "  <S^2>       : $(grep 'Expectation value of <S\*\*2>' tsopt2.out | awk '{print $NF}' | tail -1)"
python3 - tsopt2.out <<'EOP'
import re, sys
t = open(sys.argv[1], errors='replace').read()
i = t.rfind('VIBRATIONAL FREQUENCIES')
if i < 0:
    print('  Frequenzen  : keine')
else:
    fr = [float(m.group(1)) for m in
          re.finditer(r'^\s*\d+:\s+(-?\d+\.\d+)\s+cm', t[i:], re.M)]
    im = [v for v in fr if v < -1.0]
    print('  imaginaer   : %d   %s' % (len(im), ' '.join('%.1f' % v for v in im[:4])))
EOP
echo "Finished $(date)"
