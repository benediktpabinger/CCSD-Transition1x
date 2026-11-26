#!/bin/bash
#SBATCH --job-name=label_all
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=180G
#SBATCH --output=/home/energy/bepab/logs/label_all_%A_%a.out
#SBATCH --error=/home/energy/bepab/logs/label_all_%A_%a.err

# Label check at every Transition1x transition state (D46), the protocol of job_border_zone.sh
# without the RKS run, which the label check does not use:
#   1  uks_sp      UKS + STABPerform                               -> lambda_min, <S^2>
#   2  uks_engrad  EnGrad on the uks_sp orbitals (MORead)          -> E, F (compared with OMol25)
# The energy must come from uks_engrad: VV10 is non-self-consistent in uks_sp (0.5-1 meV off).
# ORCA 6.0.1 on el9 ($HOME/opt/orca6.env, D16).
#
# One whole xeon24 node per array task; task i works on BLOCK lines of the task file, two
# 12-core ORCA runs at a time (worker a: even lines, worker b: odd lines of the block).
# Restart skips finished runs.
#
#   N=$(wc -l < ~/t1x_ts_label_all_tasks.txt); sbatch --array=0-$(( (N+39)/40 - 1 ))%12 pipeline/job_label_all.sh

source /etc/profile
source $HOME/opt/orca6.env
ORCA=$ORCADIR/orca
[ -x "$ORCA" ] || { echo "ABORT: orca not executable ($ORCA)"; exit 5; }
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_oversubscribe=true

H=/home/energy/bepab
TASKFILE=${TASKFILE:-$H/t1x_ts_label_all_tasks.txt}
OUTROOT=${OUTROOT:-$H/orca6_label_all}
BLOCK=${BLOCK:-40}
mkdir -p $OUTROOT

METHOD='wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3'
HEAD='%pal nprocs 12 end
%maxcore 3500'
SCFCOMMON='  Thresh 1e-12
  TCut   1e-13
  MaxIter 300'

run_one () {                       # $1 = line "rxn:xyz:nel"
  local RXN=$(echo "$1" | cut -d: -f1) SRC=$(echo "$1" | cut -d: -f2) NEL=$(echo "$1" | cut -d: -f3)
  local W=$OUTROOT/$RXN
  mkdir -p $W && cd $W || return 2
  [ -f "$SRC" ] || { echo "  ABORT $RXN: structure missing ($SRC)"; return 6; }
  cp "$SRC" ts.xyz
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
  grep -q "ORCA TERMINATED NORMALLY" uks_sp.out || { echo "  ABORT $RXN uks_sp"; tail -3 uks_sp.out; return 8; }
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
  grep -q "CARTESIAN GRADIENT" uks_engrad.out || { echo "  ABORT $RXN uks_engrad"; tail -3 uks_engrad.out; return 9; }
  local NEL_OUT=$(grep -m1 "Number of Electrons" uks_sp.out | awk '{print $NF}')
  [ -z "$NEL" ] || [ "$NEL_OUT" = "$NEL" ] || { echo "  ABORT $RXN: electrons $NEL vs ORCA $NEL_OUT"; return 10; }
  rm -f *.densities *.densitiesinfo *_property.txt *.property.txt uks_start.gbw uks_sp.gbw uks_engrad.gbw
  echo "DONE $RXN  <S^2> $(grep 'Expectation value of <S\*\*2>' uks_engrad.out | tail -1 | awk '{print $NF}')  $(date +%T)"
}

worker () {                        # $1 = parity 0/1
  local RC=0 n
  for (( n = $1; n < BLOCK; n += 2 )); do
    local L=$(sed -n "$((SLURM_ARRAY_TASK_ID * BLOCK + n + 1))p" $TASKFILE)
    [ -n "$L" ] || break
    ( run_one "$L" ) || RC=1
  done
  return $RC
}

echo "Task $SLURM_ARRAY_TASK_ID: lines $((SLURM_ARRAY_TASK_ID * BLOCK + 1))..$(((SLURM_ARRAY_TASK_ID + 1) * BLOCK)) of $TASKFILE  $(date)"
worker 0 > $OUTROOT/task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_a.log 2>&1 &
P1=$!
worker 1 > $OUTROOT/task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_b.log 2>&1 &
P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
cat $OUTROOT/task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_{a,b}.log
echo "END task $SLURM_ARRAY_TASK_ID  rc_a=$R1 rc_b=$R2  $(date)"
[ $R1 -eq 0 ] && [ $R2 -eq 0 ]
