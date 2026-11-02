#!/bin/bash
#SBATCH --job-name=omol_neb
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/bepab/logs/omol_neb_%A_%a.out
#SBATCH --error=/home/energy/bepab/logs/omol_neb_%A_%a.err

# CI-NEB with one OMol25 model on a block of reactions (pipeline/omol_neb.py).
# Default: a whole xeon24el9 node with WORKERS processes of 24/WORKERS threads
# each. Array task i works on the reactions i*BLOCK .. (i+1)*BLOCK-1 of RXNS.
# Finished reactions (result.json) are skipped, so a rerun only does the rest.
#
#   sbatch --export=ALL,MODEL=uma-s,RXNS=$HOME/list.txt,BLOCK=50,WORKERS=4 --array=0-3%18 pipeline/job_omol_neb.sh
#
# GPU:   add  -p a100el9 --ntasks=1 --cpus-per-task=8 --mem=48G --gres=gpu:1
#        and  DEVICE=cuda,WORKERS=1  in --export.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0

MODEL=${MODEL:?MODEL not set}
RXNS=${RXNS:?RXNS not set}
BLOCK=${BLOCK:-50}
WORKERS=${WORKERS:-4}
DEVICE=${DEVICE:-cpu}
OUTROOT=${OUTROOT:-$HOME/neb_omol}
PRECISION=${PRECISION:-default}     # default, float32 or float64
CINEB_FMAX=${CINEB_FMAX:-0.05}
STEPS=${STEPS:-500}
START_BAND=${START_BAND:-t1x}
I=${SLURM_ARRAY_TASK_ID:-0}
EXTRA=""
[ -n "$CONTINUE_FROM" ] && EXTRA="--continue-from $CONTINUE_FROM"

# mace-torch 0.3.15 needs e3nn 0.4.4, fairchem e3nn >= 0.5 (see job_model_sp_ts.sh)
[ "$MODEL" = "mace-omol" ] && export PYTHONPATH=$HOME/venvs/e3nn044${PYTHONPATH:+:$PYTHONPATH}
if [ "$DEVICE" = "cuda" ]; then
  export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
else
  export OMP_NUM_THREADS=$(( 24 / WORKERS ))
fi
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p $HOME/logs "$OUTROOT"
echo "Task $I: $MODEL on $DEVICE, block $(( I * BLOCK ))+$BLOCK of $RXNS, $WORKERS workers, $OMP_NUM_THREADS threads, $(hostname) $(date)"

cd $HOME/Transition1x
PIDS=()
for (( W = 0; W < WORKERS; W++ )); do
  python3 pipeline/omol_neb.py --model $MODEL --rxns $RXNS --start $(( I * BLOCK )) --count $BLOCK \
      --worker $W --nworkers $WORKERS --device $DEVICE --outroot "$OUTROOT" \
      --precision $PRECISION --cineb-fmax $CINEB_FMAX --steps $STEPS --start-band $START_BAND $EXTRA &
  PIDS+=($!)
done
RC=0
for P in "${PIDS[@]}"; do wait $P || RC=1; done
echo "ENDE Task $I   rc=$RC   $(date)"
exit $RC
