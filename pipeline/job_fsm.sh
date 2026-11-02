#!/bin/bash
#SBATCH --job-name=fsm
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/bepab/logs/fsm_%j.out
#SBATCH --error=/home/energy/bepab/logs/fsm_%j.err

# Freezing-string search with Sella refinement on the model (pipeline/fsm_search.py).
# WORKERS processes on one node share the reactions. Finished ones are skipped.
# Sella and JAX in ~/venvs/sella, ML-FSM from source in ~/opt/mlfsm_src.
#
#   sbatch --export=ALL,MODEL=esen pipeline/job_fsm.sh

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0

MODEL=${MODEL:-esen}
WORKERS=${WORKERS:-3}
OUTROOT=${OUTROOT:-$HOME/neb_fsm}
export OMP_NUM_THREADS=$(( 24 / WORKERS ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export JAX_PLATFORMS=cpu
export PYTHONPATH=$HOME/venvs/sella:$HOME/opt/mlfsm_src/src${PYTHONPATH:+:$PYTHONPATH}
[ "$MODEL" = "mace-omol" ] && export PYTHONPATH=$HOME/venvs/e3nn044:$PYTHONPATH

cd $HOME/Transition1x
EXTRA=""
[ -n "$RXNS" ] && EXTRA="--rxns $RXNS"
PIDS=()
for (( W = 0; W < WORKERS; W++ )); do
  python3 pipeline/fsm_search.py --model $MODEL --outroot "$OUTROOT" $EXTRA &
  PIDS+=($!)
done
RC=0
for P in "${PIDS[@]}"; do wait $P || RC=1; done
echo "ENDE $MODEL rc=$RC $(date)"
exit $RC
