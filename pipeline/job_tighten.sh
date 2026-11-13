#!/bin/bash
#SBATCH --job-name=tighten
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/home/energy/bepab/logs/tighten_%j.out
#SBATCH --error=/home/energy/bepab/logs/tighten_%j.err

# D43: Sella on the model from each converged freezing-string saddle down to 0.01 eV/A
# (pipeline/tighten_saddles.py). WORKERS processes share the structures of MODEL; finished
# ones are skipped, so several jobs per model can run side by side.
#
#   sbatch --export=ALL,MODEL=esen pipeline/job_tighten.sh

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0

MODEL=${MODEL:?MODEL not set}
WORKERS=${WORKERS:-3}
export OMP_NUM_THREADS=$(( 24 / WORKERS ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export JAX_PLATFORMS=cpu
export PYTHONPATH=$HOME/venvs/sella${PYTHONPATH:+:$PYTHONPATH}
[ "$MODEL" = "mace-omol" ] && export PYTHONPATH=$HOME/venvs/e3nn044:$PYTHONPATH

cd $HOME/Transition1x
PIDS=()
for (( W = 0; W < WORKERS; W++ )); do
  python3 pipeline/tighten_saddles.py --model $MODEL &
  PIDS+=($!)
done
RC=0
for P in "${PIDS[@]}"; do wait $P || RC=1; done
echo "ENDE $MODEL rc=$RC $(date)"
