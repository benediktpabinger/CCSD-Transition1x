#!/bin/bash
#SBATCH --job-name=freqnew
#SBATCH --partition=xeon24el8
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/energy/s242862/freq_newts/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/freq_newts/slurm_%A_%a.err

# Frequencies for the transition states that NEB-TS produced without any.
#
# NEB-TS writes a converged structure but computes no Hessian, so the runs from
# the model path and the 16-image bands are stuck at stage 1: a structure
# exists and nothing is known about it.  The NEB-CI runs do not have this
# problem because their third stage was an explicit OptTS with Freq, and that
# is exactly where 7 of 7 came from.
#
# Two stages, because a frequency calculation on the wrong sheet is worthless:
#
#   1  single point with STABPerform -> the ground-state orbitals at this
#      geometry, and <S^2> recorded
#   2  Freq reading those orbitals
#
# wB97X carries no VV10, so stage 2 is analytic and takes minutes rather than
# the 6N displacements a numerical Hessian would need.
#
# The count of imaginary modes is taken from ORCA's own output afterwards, not
# from a re-diagonalised Hessian: the projection in sweep_summary.py leaves
# residual rotations near -24 cm-1 on these structures and its threshold sits
# at -20, which turns first-order saddles into second-order ones.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
# every NEB-TS structure from last night that has no Hessian
CFG=(modelpath:rxn5690 modelpath:rxn8827 \
     neb16:rxn1320 neb16:rxn8837 \
     nebci:rxn5691 nebci:rxn8827 modelpath:rxn5691)
ENTRY=${CFG[$SLURM_ARRAY_TASK_ID]}
SET=${ENTRY%%:*}
RXN=${ENTRY##*:}

case $SET in
  modelpath) SRC=$H/bs_uks_neb_modelpath/$RXN ;;
  neb16)     SRC=$H/bs_uks_neb16/$RXN ;;
  nebci)     SRC=$H/bs_uks_nebci/$RXN ;;
esac

W=$H/freq_newts/${SET}_${RXN}
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $SET / $RXN  node $SLURM_NODELIST  $(date)"

TS=$(ls $SRC/*NEB-TS_converged.xyz 2>/dev/null | head -1)
[ -z "$TS" ] && TS=$(ls $SRC/*NEB-CI_converged.xyz 2>/dev/null | head -1)
if [ -z "$TS" ]; then
  echo "ABBRUCH: keine konvergierte Struktur in $SRC"
  exit 2
fi
cp $TS start.xyz
echo "Struktur: $(basename $TS)"

# ------------------------------------------------------------------ 1
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

* xyzfile 0 1 $W/start.xyz
EOF

$ORCA bs.inp > bs.out 2> bs.err
S2=$(grep 'Expectation value of <S\*\*2>' bs.out | awk '{print $NF}' | tail -1)
echo "1: <S^2> = $S2"
if [ ! -s bs.gbw ]; then
  echo "ABBRUCH: Stufe 1 hat keine Orbitale erzeugt"
  exit 3
fi

# ------------------------------------------------------------------ 2
cat > freq.inp <<EOF
! UKS wB97X 6-31G(d) Freq TightSCF SlowConv MORead

%moinp "$W/bs.gbw"

%pal
  nprocs 8
end

%maxcore 3500

%scf
  MaxIter 500
end

* xyzfile 0 1 $W/start.xyz
EOF

$ORCA freq.inp > freq.out 2> freq.err
echo "2: rc=$?"

echo ""
echo "--- Ergebnis ---"
echo "  <S^2> nach Freq: $(grep 'Expectation value of <S\*\*2>' freq.out \
      | awk '{print $NF}' | tail -1)"
python3 - freq.out <<'EOF'
import re, sys
t = open(sys.argv[1], errors='replace').read()
i = t.rfind('VIBRATIONAL FREQUENCIES')
if i < 0:
    print('  keine Frequenzen')
else:
    fr = [float(m.group(1)) for m in
          re.finditer(r'^\s*\d+:\s+(-?\d+\.\d+)\s+cm', t[i:], re.M)]
    im = [v for v in fr if v < -1.0]
    print('  n_imag = %d   %s' % (len(im), ' '.join('%.1f' % v for v in im[:5])))
    print('  niedrigste reelle: %s'
          % ' '.join('%.1f' % v for v in sorted(v for v in fr if v > 1)[:3]))
EOF
echo "  Hesse: $(ls *.hess 2>/dev/null | tr '\n' ' ')"
echo "Finished $(date)"
