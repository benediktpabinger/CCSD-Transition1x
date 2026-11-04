#!/bin/bash
#SBATCH --job-name=pilotchk
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/bepab/logs/pilotchk_%j.out
#SBATCH --error=/home/energy/bepab/logs/pilotchk_%j.err

# Hessian and downhill check of the lower pilot barriers (pipeline/pilot_check.py).
# WORKERS processes on one node share the structures of MODEL. Finished ones are
# skipped, so a rerun only does the rest. SETS="outdir:root outdir:root ..." runs
# several check sets one after the other (D35); without it OUTDIR and ROOT are used.
#
#   sbatch --export=ALL,MODEL=esen pipeline/job_pilot_check.sh

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0

MODEL=${MODEL:?MODEL not set}
WORKERS=${WORKERS:-3}
OUTDIR=${OUTDIR:-$HOME/neb_pilot/check}
ROOT=${ROOT:-$HOME/neb_pilot}
SETS=${SETS:-$OUTDIR:$ROOT}
export OMP_NUM_THREADS=$(( 24 / WORKERS ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
[ "$MODEL" = "mace-omol" ] && export PYTHONPATH=$HOME/venvs/e3nn044${PYTHONPATH:+:$PYTHONPATH}

cd $HOME/Transition1x
RC=0
for S in $SETS; do
  PIDS=()
  for (( W = 0; W < WORKERS; W++ )); do
    python3 pipeline/pilot_check.py --model $MODEL --outdir "${S%%:*}" --root "${S#*:}" &
    PIDS+=($!)
  done
  for P in "${PIDS[@]}"; do wait $P || RC=1; done
done
echo "ENDE $MODEL rc=$RC $(date)"
exit $RC
