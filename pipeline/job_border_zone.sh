#!/bin/bash
#SBATCH --job-name=borderzone
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --output=/home/energy/bepab/logs/border_zone_%A_%a.out
#SBATCH --error=/home/energy/bepab/logs/border_zone_%A_%a.err

# Grenzzonen-Pruefung der <S^2>-Schwelle (docs/paper_decisions.md).
#
# Dieselben drei ORCA-Laeufe wie job_hinge_t1x.sh, an den gespeicherten
# Transition1x-Uebergangszustaenden der Reaktionen, deren OMol25-Label ein
# kleines <S^2> traegt (Taskliste aus pipeline/mk_border_tasks.py):
#   1  rks_sp      RKS + EnGrad                 -> E_RKS, F_RKS
#   2  uks_sp      UKS + STABPerform            -> E_BS, <S^2>
#   3  uks_engrad  EnGrad auf den Orbitalen von uks_sp (MORead) -> F_BS
#
# Partition xeon24el8, nicht el9: ORCA/5.0.4-gompi-2023a gibt es nur in den
# el8-Modulbaeumen, und die Admins halten die el8-Knoten dafuer vorerst am
# Leben (Steen Lysgaard, 2026-10-28; docs/paper_decisions.md D15).
#
# Ein ganzer xeon24-Knoten je Array-Task (CLAUDE.md), darauf ZWEI Reaktionen
# gleichzeitig mit je 12 Kernen: Task i rechnet die Taskfile-Zeilen 2i+1 und
# 2i+2. Damit die beiden mpirun nicht dieselben 12 Kerne binden, wird das
# Core-Binding von OpenMPI abgeschaltet.
#
# Schreibt nur nach ~/orca_border_zone/<rxn>/. Neustart ueberspringt fertige
# Laeufe (Pruefung auf vorhandene .out-Dateien).
#
#   N=$(wc -l < ~/t1x_ts_border_tasks.txt); sbatch --array=0-$(( (N+1)/2 - 1 ))%18 pipeline/job_border_zone.sh

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_oversubscribe=true

H=/home/energy/bepab
TASKFILE=${TASKFILE:-$H/t1x_ts_border_tasks.txt}
OUTROOT=$H/orca_border_zone
mkdir -p $OUTROOT

METHOD='wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3'
HEAD='%pal nprocs 12 end
%maxcore 3500'
SCFCOMMON='  Thresh 1e-12
  TCut   1e-13
  MaxIter 300'

run_one () {                       # $1 = Zeile "rxn:xyz:nel"
  local LINE=$1
  local RXN=$(echo "$LINE" | cut -d: -f1)
  local SRC=$(echo "$LINE" | cut -d: -f2)
  local NEL=$(echo "$LINE" | cut -d: -f3)
  local W=$OUTROOT/$RXN
  mkdir -p $W && cd $W || return 2
  echo "Aufgabe: $RXN   Struktur $SRC   $(date)"
  [ -f "$SRC" ] || { echo "  ABBRUCH $RXN: Struktur fehlt ($SRC)"; return 6; }
  cp "$SRC" ts.xyz

  if [ ! -f rks_sp.out ] || ! grep -q "ORCA TERMINATED NORMALLY" rks_sp.out; then
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
  grep -q "ORCA TERMINATED NORMALLY" rks_sp.out || { echo "  ABBRUCH $RXN rks_sp"; tail -3 rks_sp.out; return 7; }

  if [ ! -f uks_sp.out ] || ! grep -q "ORCA TERMINATED NORMALLY" uks_sp.out; then
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
  grep -q "ORCA TERMINATED NORMALLY" uks_sp.out || { echo "  ABBRUCH $RXN uks_sp"; tail -3 uks_sp.out; return 8; }

  if [ ! -f uks_engrad.out ] || ! grep -q "CARTESIAN GRADIENT" uks_engrad.out; then
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
  grep -q "CARTESIAN GRADIENT" uks_engrad.out || { echo "  ABBRUCH $RXN uks_engrad"; tail -3 uks_engrad.out; return 9; }

  local NEL_OUT=$(grep -m1 "Number of Electrons" rks_sp.out | awk '{print $NF}')
  if [ -n "$NEL" ] && [ "$NEL_OUT" != "$NEL" ]; then
    echo "  ABBRUCH $RXN: Elektronenzahl widerspricht sich -- Taskfile $NEL, ORCA $NEL_OUT"
    return 10
  fi
  echo "  $RXN E_RKS = $(grep 'FINAL SINGLE POINT ENERGY' rks_sp.out | tail -1 | awk '{print $NF}')"
  echo "  $RXN E_BS  = $(grep 'FINAL SINGLE POINT ENERGY' uks_sp.out | tail -1 | awk '{print $NF}')"
  echo "  $RXN <S^2> = $(grep 'Expectation value of <S\*\*2>' uks_sp.out | tail -1 | awk '{print $NF}')"
  rm -f *.densities *_property.txt uks_start.gbw
  echo "FERTIG $RXN   $(date)"
}

I=$SLURM_ARRAY_TASK_ID
L1=$(sed -n "$((2*I + 1))p" $TASKFILE)
L2=$(sed -n "$((2*I + 2))p" $TASKFILE)
[ -n "$L1" ] || { echo "ABBRUCH: keine Aufgabe fuer Index $I"; exit 3; }

( run_one "$L1" ) > $OUTROOT/task_${SLURM_ARRAY_JOB_ID}_${I}_a.log 2>&1 &
P1=$!
if [ -n "$L2" ]; then
  ( run_one "$L2" ) > $OUTROOT/task_${SLURM_ARRAY_JOB_ID}_${I}_b.log 2>&1 &
  P2=$!
fi
wait $P1; R1=$?
R2=0; [ -n "$L2" ] && { wait $P2; R2=$?; }
cat $OUTROOT/task_${SLURM_ARRAY_JOB_ID}_${I}_a.log
[ -n "$L2" ] && cat $OUTROOT/task_${SLURM_ARRAY_JOB_ID}_${I}_b.log
echo "ENDE Task $I   rc_a=$R1 rc_b=$R2   $(date)"
[ $R1 -eq 0 ] && [ $R2 -eq 0 ]
