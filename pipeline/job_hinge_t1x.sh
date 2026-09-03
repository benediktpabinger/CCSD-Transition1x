#!/bin/bash
#SBATCH --job-name=hingeT1x
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --output=/home/energy/s242862/orca_hinge_t1x/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_hinge_t1x/slurm_%A_%a.err

# Der Hinge-Test an den LABEL-Geometrien.
#
# Gleiche Rechnung wie job_hinge_omol25.sh, andere Geometrie: nicht der
# Uebergangszustand unseres eigenen NEB, sondern der aus Transition1x selbst
# (wB97x/6-31G(d), Gruppe transition_state im H5, extrahiert von
# pipeline/extract_t1x_ts.py). Das ist die Struktur, auf der die Modelle
# trainiert sind.
#
# Folge fuer die Auswertung: F_RKS traegt hier zusaetzlich den Niveauwechsel
# wB97x/6-31G(d) -> wB97M-V/def2-TZVPD. Die stabilen Reaktionen messen diesen
# Anteil, weil dort beide Flaechen zusammenfallen.
#
# Frage: Der Uebergangszustand ist auf der RESTRINGIERTEN Flaeche optimiert.
# Wie gross ist die Restkraft dort auf der Flaeche, die an diesem Punkt der
# Grundzustand ist? Beide Kraefte an derselben Kerngeometrie, nur die
# elektronische Loesung unterscheidet sich.
#
# Neu gegenueber der alten Fassung (results/hinge_rows.csv):
#   Geometrie   frueher orca_neb_results/  (def2-TZVP-NEB, RKS)
#               jetzt   orca_neb_omol25/   (def2-TZVPD-NEB, RKS, gleiches
#                                           Niveau wie die Kraefte)
#   Kraefte     frueher PySCF wB97M-V/def2-TZVP, grids 3
#               jetzt   ORCA 5.0.4, OMol25-Niveau, wie die Audit-Tabelle
#
# Drei ORCA-Laeufe je Reaktion:
#   1  rks_sp      RKS + EnGrad            -> E_RKS, F_RKS
#   2  uks_sp      UKS + STABPerform       -> E_BS, <S^2>
#   3  uks_engrad  EnGrad auf den Orbitalen von uks_sp (MORead) -> F_BS
# Der RKS-Lauf bewusst OHNE Stabilitaetsanalyse: die restringierte Loesung ist
# dort gewollt, auch wo sie nicht der Grundzustand ist.
#
# Schreibt nur nach orca_hinge_t1x/. t1x_ts/ wird nur gelesen.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }

H=/home/energy/s242862
TASKFILE=${TASKFILE:-$H/t1x_ts_tasks.txt}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $TASKFILE)
[ -n "$LINE" ] || { echo "ABBRUCH: keine Aufgabe fuer Index $SLURM_ARRAY_TASK_ID"; exit 3; }
RXN=$(echo "$LINE" | cut -d: -f1)
SRC=$(echo "$LINE" | cut -d: -f2)     # absoluter Pfad zur Startstruktur
NEL=$(echo "$LINE" | cut -d: -f3)     # Elektronenzahl aus dem H5

W=$H/orca_hinge_t1x/$RXN
mkdir -p $W && cd $W
echo "Aufgabe: $RXN   Struktur $SRC   $(date)"
[ -f "$SRC" ] || { echo "ABBRUCH: Struktur fehlt ($SRC)"; exit 6; }
cp "$SRC" ts.xyz

METHOD='wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3'
HEAD='%pal nprocs 12 end
%maxcore 3500'
SCFCOMMON='  Thresh 1e-12
  TCut   1e-13
  MaxIter 300'

# ---- 1  RKS mit Gradient, keine Stabilitaetsanalyse
if [ ! -f "$W/rks_sp.out" ]; then
  cat > rks_sp.inp <<EOF
! RKS $METHOD EnGrad
$HEAD
%scf
$SCFCOMMON
end
* xyzfile 0 1 ts.xyz
EOF
  $ORCA rks_sp.inp > rks_sp.out 2>&1
fi
grep -q "ORCA TERMINATED NORMALLY" rks_sp.out || { echo "  rks_sp: ABBRUCH"; tail -3 rks_sp.out; exit 7; }

# ---- 2  UKS mit Stabilitaetsanalyse
if [ ! -f "$W/uks_sp.out" ]; then
  cat > uks_sp.inp <<EOF
! UKS $METHOD
$HEAD
%scf
$SCFCOMMON
  STABPerform true
  STABRestartUHFifUnstable true
end
* xyzfile 0 1 ts.xyz
EOF
  $ORCA uks_sp.inp > uks_sp.out 2>&1
fi
grep -q "ORCA TERMINATED NORMALLY" uks_sp.out || { echo "  uks_sp: ABBRUCH"; tail -3 uks_sp.out; exit 8; }

# ---- 3  Gradient auf genau der Loesung, die uks_sp gefunden hat
if [ ! -f "$W/uks_engrad.out" ]; then
  cp uks_sp.gbw uks_start.gbw
  cat > uks_engrad.inp <<EOF
! UKS $METHOD EnGrad MORead
%moinp "uks_start.gbw"
$HEAD
%scf
$SCFCOMMON
end
* xyzfile 0 1 ts.xyz
EOF
  $ORCA uks_engrad.inp > uks_engrad.out 2>&1
fi
grep -q "CARTESIAN GRADIENT" uks_engrad.out || { echo "  uks_engrad: ABBRUCH"; tail -3 uks_engrad.out; exit 9; }

NEL_OUT=$(grep -m1 "Number of Electrons" rks_sp.out | awk '{print $NF}')
if [ -n "$NEL" ] && [ "$NEL_OUT" != "$NEL" ]; then
  echo "  ABBRUCH: Elektronenzahl widerspricht sich -- Taskfile $NEL, ORCA $NEL_OUT"
  exit 10
fi

echo "  E_RKS = $(grep 'FINAL SINGLE POINT ENERGY' rks_sp.out | tail -1 | awk '{print $NF}')"
echo "  E_BS  = $(grep 'FINAL SINGLE POINT ENERGY' uks_sp.out | tail -1 | awk '{print $NF}')"
echo "  <S^2> = $(grep 'Expectation value of <S\*\*2>' uks_sp.out | tail -1 | awk '{print $NF}')"
for f in rks_sp uks_engrad; do
  awk -v tag=$f '/CARTESIAN GRADIENT/{g=1;next} g&&NF>=6{for(i=4;i<=6;i++){v=$i<0?-$i:$i; if(v>m)m=v}}
       g&&/Difference to translation/{printf "  max|F| %-11s = %.4f eV/A\n", tag, m*51.42208; exit}' $f.out
done
echo "FERTIG $RXN   $(date)"
