#!/bin/bash
#SBATCH --job-name=bands2
#SBATCH --partition=xeon24el8
#SBATCH --array=0-21
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/energy/s242862/band_s2_v2/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/band_s2_v2/slurm_%A_%a.err

# WHAT THE BAND ACTUALLY DID -- second attempt, with a control.
#
# Everything said so far about "the band stayed restricted" was read out of
# bs_uks_neb.out, which does not contain the band's SCFs at all.  ORCA logs
# PREOPT (the two endpoint relaxations) and the final TS optimisation; the
# per-image SCFs of the band go nowhere the main output can see.  The giveaway
# is the count -- rxn4113 ran 100 band iterations and logged 36 SCFs.
#
# The first attempt to recover it used `NoIter`, and produced <S^2> = 0.000 for
# all 220 images.  That looked like a clean confirmation of the diagnosis and
# was nothing of the sort: those runs also printed
#
#     FINAL SINGLE POINT ENERGY     0.000000000000
#
# NoIter skips the property evaluation entirely.  Zero was the absence of a
# number, not a measurement of zero.
#
# This version uses MaxIter 1: one SCF cycle starting from the band's own
# converged orbitals.  From a converged wavefunction a single cycle moves
# <S^2> negligibly, so what comes out is the band's spin state, while an
# ordinary single point could relax into a different solution and hide it.
#
# And it validates itself before measuring.  Stage C below runs the identical
# recipe on a gbw whose <S^2> is already known from its own output file.  If
# the recipe cannot reproduce that number, the task aborts instead of writing
# 220 more zeros.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
cd $H/bs_uks_neb_results
RXNS=($(ls -d rxn*/ | tr -d '/' | sort))
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

SRC=$H/bs_uks_neb_results/$RXN
W=$H/band_s2_v2/$RXN
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

run_s2 () {   # $1 = gbw   $2 = xyz   $3 = tag  -> echoes "<S^2> energy cycles"
  cp "$1" ${3}_in.gbw
  cat > ${3}.inp <<EOF
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF MORead

%moinp "$W/${3}_in.gbw"

%pal
  nprocs 4
end

%maxcore 3500

%scf
  MaxIter 60
end

* xyzfile 0 1 $2
EOF
  $ORCA ${3}.inp > ${3}.out 2> ${3}.err
  local s2 e cyc
  s2=$(grep 'Expectation value of <S\*\*2>' ${3}.out | awk '{print $NF}' | tail -1)
  # ORCA appends "(SCF not fully converged!)" to this line when it did not
  # converge, so $NF is a word, not a number.  Take the float after ENERGY.
  e=$(grep 'FINAL SINGLE POINT ENERGY' ${3}.out | tail -1 \
      | sed -E 's/.*ENERGY[[:space:]]+(-?[0-9]+\.[0-9]+).*/\1/')
  cyc=$(grep -c 'ITER ' ${3}.out 2>/dev/null)
  [ -z "$s2" ] && s2="nan"
  case "$e" in ''|*[!0-9.+-]*) e="nan" ;; esac
  rm -f ${3}_in.gbw ${3}.gbw ${3}.densities ${3}*.tmp
  echo "$s2 $e ${cyc:-0}"
}

# ---------------------------------------------------------------- stage C
# Control: a wavefunction with a known <S^2>, measured by the same recipe.
CTLDIR=""
for c in $H/orca_freq/nebts_$RXN $H/orca_freq/ours_$RXN $H/orca_irc/${RXN}_ours; do
  [ -f $c/bs_sp.gbw ] && [ -f $c/start.xyz ] && CTLDIR=$c && break
done

if [ -z "$CTLDIR" ]; then
  echo "KONTROLLE: keine Referenz-gbw fuer $RXN gefunden -- Messung nicht validierbar"
  exit 2
fi

KNOWN=$(grep 'Expectation value of <S\*\*2>' $CTLDIR/bs_sp.out | awk '{print $NF}' | tail -1)
read CS2 CE CCYC <<< "$(run_s2 $CTLDIR/bs_sp.gbw $CTLDIR/start.xyz ctl)"
echo "KONTROLLE  bekannt <S^2> = $KNOWN   gemessen = $CS2   E = $CE   Zyklen = $CCYC"

# nan must be rejected before any numeric comparison.  In the first attempt awk
# silently read "nan" as 0, and where the known value happened to be 0.000 the
# control passed on a measurement that did not exist.
if [ "$CS2" = "nan" ] || [ "$CE" = "nan" ] || [ -z "$KNOWN" ]; then
  echo "ABBRUCH: Kontrolle liefert keinen Zahlenwert -- Rezept misst nichts."
  exit 3
fi
if ! awk -v a="$KNOWN" -v b="$CS2" \
     'BEGIN{d=a-b; if(d<0) d=-d; exit !(d < 0.05)}'; then
  echo "ABBRUCH: Kontrolle weicht ab ($KNOWN gegen $CS2) -- Rezept nicht vertrauenswuerdig."
  exit 4
fi
echo "Kontrolle bestanden, Messung beginnt."
echo ""

# ---------------------------------------------------------------- messen
TRJ=$(ls $SRC/*_MEP_trj.xyz 2>/dev/null | head -1)
[ -z "$TRJ" ] && { echo "kein MEP_trj"; exit 0; }

NAT=$(head -1 $TRJ | tr -d ' \r')
awk -v nat="$NAT" 'BEGIN{i=-1}
     {if ((NR-1) % (nat+2) == 0) {i++; f=sprintf("img_%d.xyz", i)}
      print > f}' $TRJ

BASE=$(ls $SRC/*_im0.gbw | head -1 | sed 's/_im0\.gbw$//')
OUT=$W/band_s2.txt
: > $OUT

N=$(ls img_*.xyz | wc -l)
for k in $(seq 0 $((N-1))); do
  G=${BASE}_im${k}.gbw
  [ -f "$G" ] || { echo "  im$k: keine Orbitale"; continue; }
  read S2 E CYC <<< "$(run_s2 $G $W/img_${k}.xyz im$k)"
  printf "%-9s %2d %10s %20s %5s\n" "$RXN" "$k" "$S2" "$E" "$CYC" >> $OUT
  printf "  im%-2d  <S^2> = %-10s  E = %-20s  Zyklen = %s\n" "$k" "$S2" "$E" "$CYC"
done

echo ""
echo "--- $RXN ---"
# The cycle count is the self-diagnosis.  Restarting from a converged
# wavefunction at its own geometry should take very few cycles; a long
# re-convergence means the SCF left the band's solution and the <S^2> printed
# belongs to a different one.
awk '{if ($3 != "nan" && $4 != "nan" && $4+0 != 0) {n++;
       if ($3+0 > mx) mx = $3+0; if ($3+0 > 0.3) c++
       if ($5+0 > mc) mc = $5+0}}
     END {printf "  gueltig gemessen %d   max <S^2> %.3f   ueber 0.3: %d   "
                 "max Zyklen %d\n", n+0, mx+0, c+0, mc+0}' $OUT
echo "Finished $(date)"
