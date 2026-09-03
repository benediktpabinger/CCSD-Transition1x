#!/bin/bash
#SBATCH --job-name=rkssheet
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --output=/home/energy/s242862/orca_rks_sheet/slurm_%A_%a.out

# Auf welcher Flaeche sitzt das MLIP bei den Energie-Ausreissern?
#
# orca_om25/<rxn>_<Modell>/ts_sp.out liefert bereits E_BS am Modell-TS: der
# Lauf hat UKS gerechnet und ist dort auf die gebrochene Loesung konvergiert
# (<S^2> ~ 1). Was fehlt, ist die restringierte Loesung am selben Punkt.
#
# Hier genau ein Lauf je Zeile: RKS statt UKS, sonst wortgleiche Einstellungen
# wie in job_orca_omol25_probe.sh, damit E_RKS und E_BS ohne Umrechnung
# nebeneinander stehen. Keine Stabilitaetsanalyse -- die restringierte Loesung
# ist hier gewollt, auch wenn sie nicht der Grundzustand ist.
#
# Der Edukt-Nullpunkt wird nicht neu gerechnet: r_sp.out hat in allen
# betroffenen Zeilen <S^2> = 0, ist also bereits die restringierte Loesung.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }

H=/home/energy/s242862
TASKFILE=${TASKFILE:-$H/rks_sheet_tasks.txt}
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

W=$H/orca_rks_sheet/${RXN}_${MOD}
mkdir -p $W && cd $W
echo "Aufgabe: $RXN $MOD    $(date)"

SRC=$H/$DIR/$RXN/transition_state.xyz
[ -f "$SRC" ] || { echo "ABBRUCH: Struktur fehlt ($SRC)"; exit 6; }

if [ ! -f "$W/ts_rks.out" ]; then
  cp "$SRC" ts_rks.xyz
  cat > ts_rks.inp <<EOF
! RKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
%pal nprocs 12 end
%maxcore 3500
%scf
  Thresh 1e-12
  TCut   1e-13
  MaxIter 300
end
* xyzfile 0 1 ts_rks.xyz
EOF
  $ORCA ts_rks.inp > ts_rks.out 2>&1
fi

grep -q "ORCA TERMINATED NORMALLY" ts_rks.out || { echo "  ABBRUCH"; tail -5 ts_rks.out; exit 7; }
echo "  E_RKS = $(grep 'FINAL SINGLE POINT ENERGY' ts_rks.out | tail -1 | awk '{print $NF}')"
grep -o "SCF CONVERGED AFTER *[0-9]* CYCLES" ts_rks.out | tail -1 | sed 's/^/  /'
echo "FERTIG $RXN $MOD    $(date)"
