#!/bin/bash
#SBATCH --job-name=epsp
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --output=/home/energy/s242862/orca_ep/slurm_%A_%a.out

# Energien an den Endpunkten, die das Modell selbst relaxiert hat.
#
# Zweck: der Barrierenfehler bei EINGEFRORENER Geometrie. Das Modell meldet
# eine Barriere aus seinen eigenen R und TS; hier kommt dieselbe Differenz
# aus DFT an genau denselben, unveraenderten Strukturen. Was uebrig bleibt,
# ist reiner Energiefehler -- die Geometrie faellt heraus, weil sie auf
# beiden Seiten dieselbe ist.
#
# Nur ein Einzelpunkt je Struktur, kein Gradient und keine Frequenz. Die
# Rezeptur ist die aus job_orca_grad_gap.sh: STABPerform waehlt die
# Grundzustandsloesung, STABRestartUHFifUnstable startet noetigenfalls als
# UHF neu. Am TS liegt dieselbe Rechnung schon vor (orca_freq/<rxn>_<Modell>).
#
# Eigenes Ausgabeverzeichnis orca_ep/, damit kein vorhandener Auswerter die
# Endpunkte versehentlich als TS-Kandidaten einsammelt (vgl. Anhang A.4).

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }

H=/home/energy/s242862
TASKFILE=${TASKFILE:-$H/ep_tasks.txt}
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $TASKFILE)
[ -n "$TASK" ] || { echo "ABBRUCH: keine Aufgabe fuer Index $SLURM_ARRAY_TASK_ID"; exit 3; }

RXN=$(echo "$TASK" | cut -d: -f1)
MOD=$(echo "$TASK" | cut -d: -f2)
END=$(echo "$TASK" | cut -d: -f3)          # R oder P

case "$MOD" in
  UMA-S) DIR=uma_neb_results ;;
  UMA-M) DIR=uma_m_neb_results ;;
  eSEN)  DIR=esen_neb_results ;;
  *) echo "ABBRUCH: unbekanntes Modell $MOD"; exit 4 ;;
esac
case "$END" in
  R) SRCNAME=reactant.xyz ;;
  P) SRCNAME=product.xyz ;;
  *) echo "ABBRUCH: unbekannter Endpunkt $END"; exit 4 ;;
esac

SRC=$H/$DIR/$RXN/$SRCNAME
W=$H/orca_ep/${RXN}_${MOD}_${END}
echo "Aufgabe: $RXN $MOD $END"
echo "Quelle : $SRC"

[ -f "$SRC" ] || { echo "ABBRUCH: Struktur fehlt"; exit 6; }
# nie ueberschreiben -- eine stille Ersetzung waere in den Tabellen unsichtbar
[ -f "$W/bs_sp.out" ] && { echo "uebersprungen: liegt bereits vor"; exit 0; }

mkdir -p $W && cd $W
cp "$SRC" start.xyz

cat > bs_sp.inp <<EOF
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF
%pal nprocs 8 end
%maxcore 4500
%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 300
end
* xyzfile 0 1 start.xyz
EOF

$ORCA bs_sp.inp > bs_sp.out 2>&1
grep -q "ORCA TERMINATED NORMALLY" bs_sp.out || { echo "ABBRUCH: nicht normal beendet"; tail -4 bs_sp.out; exit 7; }
E=$(grep 'FINAL SINGLE POINT ENERGY' bs_sp.out | tail -1 | awk '{print $NF}')
# eine Energie von exakt 0 heisst: es wurde nichts gerechnet
awk -v e="$E" 'BEGIN{exit !(e+0 == 0)}' && { echo "ABBRUCH: Energie ist null"; exit 8; }

echo "  E      = $E"
echo "  <S^2>  = $(grep 'Expectation value of <S\*\*2>' bs_sp.out | awk '{print $NF}' | tail -1)"
echo "  Neustart: $(grep -c 'SCF ITERATIONS (restarted)' bs_sp.out)"
echo "FERTIG $RXN $MOD $END"
