#!/bin/bash
#SBATCH --job-name=marksPk
#SBATCH --nodes=1
#SBATCH --output=/home/energy/s242862/marks_stab/packed_%j.out
#SBATCH --error=/home/energy/s242862/marks_stab/packed_%j.err

# Same calculation as job_marks_stab.sh, but packing several structures into one
# whole-node allocation instead of asking for a separate 8-core slot each.
#
# xeon24el8 is saturated -- one idle core on the whole partition -- while
# xeon32el9_4096 and xeon40el8_768 each have a completely idle node that only
# accepts whole-node jobs. Nothing here needs a whole node on its own (the
# 10-atom control took three minutes), so one node runs NPAR structures at
# CORES cores each, in a pool.
#
# Level of theory is wB97X-V/def2-TZVP throughout: Marks' functional, not the
# wB97M-V this project trains against.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
LIST=${TASKLIST:?TASKLIST not set}
NPAR=${NPAR:-4}
CORES=${CORES:-8}
MAXCORE=${MAXCORE:-1800}

# Several independent ORCA/OpenMPI jobs share this node. Left to itself each
# would bind its ranks starting from core 0 and they would all pile onto the
# same cores, so binding is turned off and the kernel does the placement.
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_oversubscribe=1

ORCA=$(command -v orca)
# The ORCA module is an el8 build. On an el9 node the module load fails, $ORCA
# comes back empty, every ORCA call silently does nothing, and an empty stab.out
# contains no "UNSTABLE" line -- which read as "stable" until this guard was
# added. Never let a missing binary look like a result: die here instead.
if [ -z "$ORCA" ]; then
  echo "FATAL: no orca on $SLURM_NODELIST ($(cat /etc/redhat-release 2>/dev/null))." >&2
  echo "FATAL: ORCA/5.0.4-gompi-2023a is an el8 build; submit to an el8 partition." >&2
  exit 1
fi

echo "packed job $SLURM_JOB_ID on $SLURM_NODELIST: NPAR=$NPAR CORES=$CORES"
echo "list $LIST  ($(wc -l < $LIST) structures)  $(date)"

run_one() {
  local rid=$1 chg=$2 mult=$3 geom=$4
  local W=$H/marks_stab/$rid
  mkdir -p $W; cd $W || return 1
  cp $geom start.xyz

  local REF=RKS
  [ "$mult" -eq 1 ] || REF=UKS

  cat > ref.inp <<EOF
! $REF wB97X-V def2-TZVP def2/J RIJCOSX TightSCF DEFGRID3 EnGrad
%pal nprocs $CORES end
%maxcore $MAXCORE
%scf
  MaxIter 300
end
* xyzfile $chg $mult start.xyz
EOF
  $ORCA ref.inp > ref.out 2> ref.err < /dev/null

  if [ "$mult" -eq 1 ]; then
    cat > stab.inp <<EOF
! UKS wB97X-V def2-TZVP def2/J RIJCOSX TightSCF DEFGRID3
%pal nprocs $CORES end
%maxcore $MAXCORE
%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 300
end
* xyzfile $chg $mult start.xyz
EOF
    $ORCA stab.inp > stab.out 2> stab.err < /dev/null
    # A verdict is only reported when ORCA actually finished. Anything else is
    # FAILED, never "stable" by absence of the UNSTABLE line.
    local verdict
    if ! grep -q 'ORCA TERMINATED NORMALLY' stab.out 2>/dev/null; then
      verdict="FAILED (stability run did not terminate)"
    elif grep -q 'UNSTABLE HF/KS wave function' stab.out; then
      verdict=UNSTABLE
    else
      verdict=stable
    fi
    echo "  done $rid  $verdict  $(date +%H:%M:%S)"
  else
    if grep -q 'ORCA TERMINATED NORMALLY' ref.out 2>/dev/null; then
      echo "  done $rid  open-shell (no stability step)  $(date +%H:%M:%S)"
    else
      echo "  done $rid  FAILED (reference run did not terminate)  $(date +%H:%M:%S)"
    fi
  fi
}

# The list is read on descriptor 9, not stdin, and every ORCA call gets
# </dev/null: ORCA inherits the loop's stdin otherwise and swallows the rest of
# the task list, so the loop quietly stops after the first few structures.
n=0
while read -r -u 9 rid chg mult geom; do
  [ -z "$rid" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done
  run_one "$rid" "$chg" "$mult" "$geom" &
  n=$((n+1))
done 9< "$LIST"
wait

echo "packed job finished: $n structures  $(date)"
