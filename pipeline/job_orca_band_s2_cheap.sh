#!/bin/bash
#SBATCH --job-name=bands2c
#SBATCH --partition=xeon24el8
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/energy/s242862/band_s2_cheap/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/band_s2_cheap/slurm_%A_%a.err

# What the CHEAP bands did, image by image -- the second half of a comparison
# whose first half already exists.
#
# The per-image job crashed, but its stage 2 finished first and produced
# something the project did not have: for every image of the cheap baseline
# path, whether a broken solution EXISTS there, found with a stability
# analysis.
#
#   rxn8827   6 of 10 images   max <S^2> 0.947
#   rxn1320   4 of 10          max 1.001
#   rxn8837   6 of 10          max 1.062
#
# That is availability, not use.  rxn1320 in particular is the reaction filed
# all evening as "the band broke nowhere", and a broken solution turns out to
# exist at four of its ten geometries.
#
# This job supplies the other number: what the band's own wavefunction was at
# those same images.  Together they separate two questions that were run
# together until now:
#
#   does the sheet exist here?      stage 2 of the per-image job, with STABPerform
#   did the band take it?           this job, from the band's own orbitals
#
# Recipe identical to job_orca_band_s2.sh -- MaxIter is not capped, MORead from
# the stored image orbitals, one SCF cycle from a converged wavefunction -- only
# the level changes to the cheap one and the control comes from the per-image
# run, whose <S^2> values are known per image.
#
# rxn1320 and rxn7949 were cancelled mid-optimisation; their orbitals describe
# the band at that moment, which is still worth having, and the output says so.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn8827 rxn8837 rxn1320 rxn7949)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

SRC=$H/bs_uks_neb_cheap/$RXN
W=$H/band_s2_cheap/$RXN
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"
if [ ! -f $SRC/neb.out ] || ! grep -q 'TOTAL RUN TIME' $SRC/neb.out; then
  echo "HINWEIS: dieser Lauf wurde abgebrochen -- die Orbitale beschreiben"
  echo "         das Band im Moment des Abbruchs, nicht ein konvergiertes."
fi

run_s2 () {   # $1 gbw  $2 xyz  $3 tag -> "<S^2> energy cycles"
  cp "$1" ${3}_in.gbw
  cat > ${3}.inp <<EOF
! UKS wB97X 6-31G(d) TightSCF MORead

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
  e=$(grep 'FINAL SINGLE POINT ENERGY' ${3}.out | tail -1 \
      | sed -E 's/.*ENERGY[[:space:]]+(-?[0-9]+\.[0-9]+).*/\1/')
  cyc=$(grep -c 'ITER ' ${3}.out 2>/dev/null)
  [ -z "$s2" ] && s2="nan"
  case "$e" in ''|*[!0-9.+-]*) e="nan" ;; esac
  rm -f ${3}_in.gbw ${3}.gbw ${3}.densities ${3}*.tmp
  echo "$s2 $e ${cyc:-0}"
}

# ---------------------------------------------------------------- Kontrolle
# The per-image run wrote one orbital set per image at this very level, with
# the <S^2> recorded alongside.  Reproducing one of them is the check that the
# recipe measures anything at all -- the guard that caught NoIter and MaxIter 1.
PI=$H/bs_uks_neb_perimage/$RXN
if [ ! -f $PI/s2_before.txt ] || [ ! -f $PI/guess_im0.gbw ]; then
  echo "ABBRUCH: keine Referenzwerte aus dem bildweisen Lauf fuer $RXN"
  exit 2
fi
KIDX=$(awk '$2 != "nan" {print $1; exit}' $PI/s2_before.txt)
KNOWN=$(awk -v k="$KIDX" '$1+0 == k {print $2}' $PI/s2_before.txt)
read CS2 CE CC <<< "$(run_s2 $PI/guess_im${KIDX}.gbw $PI/img_${KIDX}.xyz ctl)"
echo "KONTROLLE Bild $KIDX: bekannt $KNOWN, gemessen $CS2, E $CE, Zyklen $CC"

if [ "$CS2" = "nan" ] || [ "$CE" = "nan" ]; then
  echo "ABBRUCH: Kontrolle liefert keinen Zahlenwert."
  exit 3
fi
if ! awk -v a="$KNOWN" -v b="$CS2" \
     'BEGIN{d=a-b; if(d<0) d=-d; exit !(d < 0.05)}'; then
  echo "ABBRUCH: Kontrolle weicht ab ($KNOWN gegen $CS2)."
  exit 4
fi
echo "Kontrolle bestanden."
echo ""

# ---------------------------------------------------------------- messen
TRJ=$(ls $SRC/*_MEP_trj.xyz 2>/dev/null | head -1)
[ -z "$TRJ" ] && { echo "kein MEP_trj"; exit 0; }
NAT=$(head -1 $TRJ | tr -d ' \r')
awk -v nat="$NAT" 'BEGIN{i=-1}
    {if ((NR-1) % (nat+2) == 0) {i++; f=sprintf("img_%d.xyz", i)}
     print > f}' $TRJ

BASE=$(ls $SRC/neb_im0.gbw | head -1 | sed 's/_im0\.gbw$//')
OUT=$W/band_s2.txt
: > $OUT
N=$(ls img_*.xyz | wc -l)
for k in $(seq 0 $((N-1))); do
  G=${BASE}_im${k}.gbw
  [ -f "$G" ] || { echo "  im$k: keine Orbitale"; continue; }
  read S2 E CYC <<< "$(run_s2 $G $W/img_${k}.xyz im$k)"
  printf "%-9s %2d %10s %20s %5s\n" "$RXN" "$k" "$S2" "$E" "$CYC" >> $OUT
  printf "  im%-2d  genommen <S^2> = %-10s  Zyklen %s\n" "$k" "$S2" "$CYC"
done

echo ""
echo "--- verfuegbar gegen genommen, $RXN ---"
paste <(awk '{print $1, $2}' $PI/s2_before.txt) \
      <(awk '{print $3}' $OUT) 2>/dev/null \
  | awk '{printf "  im%-2s  verfuegbar %-9s  genommen %-9s  %s\n",
          $1, $2, $3, (($2+0 > 0.3 && $3+0 <= 0.3) ? "<-- Blatt vorhanden, nicht genommen" : "")}'
echo "Finished $(date)"
