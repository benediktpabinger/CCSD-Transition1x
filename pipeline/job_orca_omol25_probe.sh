#!/bin/bash
#SBATCH --job-name=om25probe
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --output=/home/energy/s242862/orca_om25/slurm_%A_%a.out

# Stichprobe: verschiebt sich der gemessene Modellfehler, wenn man auf dem
# Niveau rechnet, gegen das OMol25 trainiert ist?
#
# Bisher rechnen wir wB97M-V/def2-TZVP mit ORCA-Standardgitter und lockeren
# Integralschwellen. OMol25 ist def2-TZVP*D* mit DEFGRID3 und engen Schwellen.
# Der gemessene Energiefehler liegt bei 8 meV, die dokumentierte
# Basisverschiebung von dE_BS bei <=10 meV -- die beiden Zahlen sind gleich
# gross, und solange das so ist, ist der Modellfehler nicht von der
# Basissatzdifferenz zu trennen.
#
# Eine Aufgabe = eine (Reaktion, Modell). Vier ORCA-Laeufe darin:
#   1  ts_sp     Einzelpunkt am Modell-TS, STABPerform waehlt die Flaeche
#   2  ts_engrad Gradient von genau diesen Orbitalen  -> Stufe-1-Restkraft
#   3  r_sp      Einzelpunkt am Modell-Edukt          -> Barriere
#   4  p_sp      Einzelpunkt am Modell-Produkt        -> Reaktionsenergie
#
# Niveau woertlich aus omol25_settings.sh, dort gegen das OMol25-Protokoll
# validiert (identische Zustaende, Energien auf ~1e-8 Ha).

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }

H=/home/energy/s242862
TASKFILE=${TASKFILE:-$H/om25_probe_tasks.txt}
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $TASKFILE)
[ -n "$TASK" ] || { echo "ABBRUCH: keine Aufgabe fuer Index $SLURM_ARRAY_TASK_ID"; exit 3; }
RXN=$(echo "$TASK" | cut -d: -f1)
MOD=$(echo "$TASK" | cut -d: -f2)

case "$MOD" in
  UMA-S) DIR=uma_neb_results ;;
  UMA-M) DIR=uma_m_neb_results ;;
  eSEN)  DIR=esen_neb_results ;;
  *) echo "ABBRUCH: unbekanntes Modell $MOD"; exit 4 ;;
esac

W=$H/orca_om25/${RXN}_${MOD}
mkdir -p $W && cd $W
echo "Aufgabe: $RXN $MOD    $(date)"

METHOD='wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3'
HEAD='%pal nprocs 12 end
%maxcore 3500'
SCFCOMMON='  Thresh 1e-12
  TCut   1e-13
  MaxIter 300'

run_sp () {          # $1 = tag, $2 = Quellstruktur
  local tag=$1 src=$2
  [ -f "$src" ] || { echo "  $tag: Struktur fehlt ($src)"; return 1; }
  [ -f "$W/${tag}.out" ] && { echo "  $tag: liegt vor, uebersprungen"; return 0; }
  cp "$src" ${tag}.xyz
  cat > ${tag}.inp <<EOF
! UKS $METHOD
$HEAD
%scf
$SCFCOMMON
  STABPerform true
  STABRestartUHFifUnstable true
end
* xyzfile 0 1 ${tag}.xyz
EOF
  $ORCA ${tag}.inp > ${tag}.out 2>&1
  grep -q "ORCA TERMINATED NORMALLY" ${tag}.out || { echo "  $tag: ABBRUCH"; tail -3 ${tag}.out; return 1; }
  echo "  $tag: E = $(grep 'FINAL SINGLE POINT ENERGY' ${tag}.out | tail -1 | awk '{print $NF}')   <S^2> = $(grep 'Expectation value of <S\*\*2>' ${tag}.out | tail -1 | awk '{print $NF}')"
}

run_sp ts_sp $H/$DIR/$RXN/transition_state.xyz || exit 6
run_sp r_sp  $H/$DIR/$RXN/reactant.xyz         || exit 7
run_sp p_sp  $H/$DIR/$RXN/product.xyz          || exit 8

# Gradient auf genau der Loesung, die ts_sp gefunden hat
if [ ! -f "$W/ts_engrad.out" ]; then
  cp ts_sp.gbw ts_start.gbw
  cat > ts_engrad.inp <<EOF
! UKS $METHOD EnGrad MORead
%moinp "ts_start.gbw"
$HEAD
%scf
$SCFCOMMON
end
* xyzfile 0 1 ts_sp.xyz
EOF
  $ORCA ts_engrad.inp > ts_engrad.out 2>&1
  grep -q "CARTESIAN GRADIENT" ts_engrad.out || { echo "  ts_engrad: ABBRUCH"; exit 9; }
fi
awk '/CARTESIAN GRADIENT/{f=1;next} f&&NF>=6{for(i=4;i<=6;i++){v=$i<0?-$i:$i; if(v>m)m=v}}
     f&&/Difference to translation/{printf "  max|F| = %.4f eV/A\n", m*51.42208; exit}' ts_engrad.out

echo "FERTIG $RXN $MOD    $(date)"
