#!/bin/bash
#SBATCH --job-name=rotchk
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --output=/home/energy/s242862/orca_rot_check/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_rot_check/slurm_%A_%a.err

# Findet das OMol25-Symmetriebruch-Protokoll an den Audit-Geometrien dieselbe
# Loesung wie unsere Stabilitaetsanalyse?
#
# Die Master-Tabelle results/omol25_model_geoms.csv beruht auf ts_sp-Laeufen mit
# STABPerform. OMol25 bricht die Symmetrie anders: 20-Grad-Rotation zwischen
# HOMO und LUMO im Beta-Raum. In omol25_settings.sh wurden beide Wege an 26
# Reaktionen an den REFERENZ-Uebergangszustaenden verglichen und stimmten
# ueberein (26/26, ~1e-8 Ha). Die Master-Tabelle steht aber an den
# MODELLgeometrien -- dort ist die Aequivalenz nie geprueft worden.
#
# Dieser Lauf schliesst die Luecke: derselbe Punkt, dieselbe Stufe, nur der
# andere Weg. Ein UKS-Einzelpunkt je Zeile, 135 insgesamt.
#
# Niveau woertlich wie in job_orca_omol25_probe.sh (ts_sp), einziger
# Unterschied: der %scf-Block. Statt
#     STABPerform true / STABRestartUHFifUnstable true
# steht dort die Rotationszeile, woertlich aus job_omol25_settings.sh:
#     Rotate {$HOMO, $LUMO, 20, 1, 1} end
# KEINE Stabilitaetsanalyse -- das ist der Punkt des Tests.
#
# Schreibt ausschliesslich nach orca_rot_check/. orca_om25/ und
# orca_rks_sheet/ werden nur gelesen bzw. gar nicht angefasst.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }

H=/home/energy/s242862
TASKFILE=${TASKFILE:-$H/rot_check_tasks.txt}
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $TASKFILE)
[ -n "$TASK" ] || { echo "ABBRUCH: keine Aufgabe fuer Index $SLURM_ARRAY_TASK_ID"; exit 3; }

RXN=$(echo "$TASK" | cut -d: -f1)
MOD=$(echo "$TASK" | cut -d: -f2)
HOMO=$(echo "$TASK" | cut -d: -f3)
LUMO=$(echo "$TASK" | cut -d: -f4)
NEL=$(echo "$TASK" | cut -d: -f5)

case "$MOD" in
  UMA-S) DIR=uma_neb_results ;;
  UMA-M) DIR=uma_m_neb_results ;;
  eSEN)  DIR=esen_neb_results ;;
  *) echo "ABBRUCH: unbekanntes Modell $MOD"; exit 4 ;;
esac

W=$H/orca_rot_check/${RXN}_${MOD}
mkdir -p $W && cd $W
echo "Aufgabe: $RXN $MOD   HOMO=$HOMO LUMO=$LUMO NEL=$NEL   $(date)"

SRC=$H/$DIR/$RXN/transition_state.xyz
[ -f "$SRC" ] || { echo "ABBRUCH: Struktur fehlt ($SRC)"; exit 6; }

if [ ! -f "$W/ts_rot.out" ]; then
  cp "$SRC" ts_rot.xyz            # nur lesen aus dem Modellverzeichnis
  cat > ts_rot.inp <<EOF
! UKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
%pal nprocs 12 end
%maxcore 3500
%scf
  Thresh 1e-12
  TCut   1e-13
  MaxIter 300
  Rotate {$HOMO, $LUMO, 20, 1, 1} end
end
* xyzfile 0 1 ts_rot.xyz
EOF
  $ORCA ts_rot.inp > ts_rot.out 2>&1
fi

grep -q "ORCA TERMINATED NORMALLY" ts_rot.out || { echo "  ABBRUCH: nicht normal beendet"; tail -5 ts_rot.out; exit 7; }

# Elektronenzahl gegenpruefen: die Rotationsindizes muessen zu NEL passen
NEL_OUT=$(grep -m1 "Number of Electrons" ts_rot.out | awk '{print $NF}')
if [ "$NEL_OUT" != "$NEL" ]; then
  echo "  ABBRUCH: Elektronenzahl widerspricht sich -- Taskfile $NEL, ORCA $NEL_OUT"
  exit 8
fi
EXP_HOMO=$(( NEL_OUT / 2 - 1 ))
if [ "$EXP_HOMO" != "$HOMO" ]; then
  echo "  ABBRUCH: HOMO-Index passt nicht zu NEL=$NEL_OUT -- erwartet $EXP_HOMO, benutzt $HOMO"
  exit 9
fi

echo "  E     = $(grep 'FINAL SINGLE POINT ENERGY' ts_rot.out | tail -1 | awk '{print $NF}')"
echo "  <S^2> = $(grep 'Expectation value of <S\*\*2>' ts_rot.out | tail -1 | awk '{print $NF}')"
echo "  SCF   = $(grep -o 'SCF CONVERGED AFTER *[0-9]* CYCLES' ts_rot.out | tail -1)"
echo "FERTIG $RXN $MOD   $(date)"
