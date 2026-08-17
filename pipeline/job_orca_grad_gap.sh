#!/bin/bash
#SBATCH --job-name=gradgap
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --output=/home/energy/s242862/grad_gap/slurm_%A_%a.out

# The control group of the force-error table had one model per reaction --
# UMA-M 18x, UMA-S 6x, eSEN 2x -- while the multireference group had all
# three.  The per-model rows therefore rested on 2 to 18 structures on the
# left and about 15 on the right.  This job fills the 52 missing pairs so
# both sides have the same shape.
#
# The recipe is copied verbatim from the pairs that already exist
# (orca_freq/rxn0101_UMA-M): a single point with stability analysis to obtain
# the ground-state orbitals, then EnGrad reading exactly those orbitals, so
# the gradient sits on the same surface the model is being judged against.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }

H=/home/energy/s242862
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $H/grad_gap_tasks.txt)
[ -n "$TASK" ] || { echo "ABBRUCH: keine Aufgabe fuer Index $SLURM_ARRAY_TASK_ID"; exit 3; }
RXN=${TASK%%:*}
MOD=${TASK##*:}
case "$MOD" in
  UMA-S) DIR=uma_neb_results ;;
  UMA-M) DIR=uma_m_neb_results ;;
  eSEN)  DIR=esen_neb_results ;;
  *) echo "ABBRUCH: unbekanntes Modell $MOD"; exit 4 ;;
esac
SRC=$H/$DIR/$RXN/transition_state.xyz
W=$H/orca_freq/${RXN}_${MOD}
echo "Aufgabe: $RXN $MOD"
echo "Quelle : $SRC"

[ -f "$SRC" ] || { echo "ABBRUCH: Modellstruktur fehlt"; exit 6; }
# never overwrite a gradient that already exists -- the table is built from
# these directories and a silent replacement would be invisible
[ -f "$W/engrad.out" ] && { echo "uebersprungen: engrad.out existiert bereits"; exit 0; }

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
grep -q "ORCA TERMINATED NORMALLY" bs_sp.out || { echo "ABBRUCH: bs_sp nicht normal beendet"; tail -4 bs_sp.out; exit 7; }
[ -f bs_sp.gbw ] || { echo "ABBRUCH: keine Orbitale aus bs_sp"; exit 8; }
cp bs_sp.gbw bs_start.gbw

cat > engrad.inp <<EOF
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad MORead
%moinp "bs_start.gbw"
%pal nprocs 8 end
%maxcore 4500
%scf
  MaxIter 300
end
* xyzfile 0 1 start.xyz
EOF
$ORCA engrad.inp > engrad.out 2>&1
grep -q "ORCA TERMINATED NORMALLY" engrad.out || { echo "ABBRUCH: engrad nicht normal beendet"; tail -4 engrad.out; exit 9; }
grep -q "CARTESIAN GRADIENT" engrad.out || { echo "ABBRUCH: kein Gradient in der Ausgabe"; exit 10; }

echo "  <S^2> im Einzelpunkt: $(grep '<S\*\*2>' bs_sp.out | tail -1 | awk '{print $NF}')"
echo "FERTIG $RXN $MOD"
