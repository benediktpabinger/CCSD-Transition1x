#!/bin/bash
#SBATCH --job-name=model_sp_cpu
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/home/energy/bepab/logs/model_sp_cpu_%A_%a.out
#SBATCH --error=/home/energy/bepab/logs/model_sp_cpu_%A_%a.err

# CPU-Variante von job_model_sp_ts.sh fuer Zeiten, in denen alle
# GPU-Partitionen voll sind: Modell-Einzelpunkte (pipeline/model_sp_ts.py,
# --device cpu) auf einem ganzen xeon24el9-Knoten. Die kleinen
# Transition1x-Molekuele brauchen auf 24 Threads unter einer Sekunde je
# Struktur. Gleiche MODELS/SPINS/TAG/RXNS-Logik wie die GPU-Variante; die
# Ausgabe-JSONL ist neustart-sicher (fertige Reaktionen werden uebersprungen),
# darf aber nie von zwei Jobs gleichzeitig beschrieben werden -- also NICHT
# denselben TAG wie einen laufenden GPU-Job verwenden.
#
#   sbatch --export=ALL,MODELS="uma-s uma-m mace-omol",SPINS="1",TAG=allcpu --array=0-2 pipeline/job_model_sp_ts_cpu.sh

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
mkdir -p $HOME/logs $HOME/model_sp_ts

MODELS=(${MODELS:-uma-s uma-m mace-omol})
SPINS=(${SPINS:-1})
TAG=${TAG:-allcpu}
I=${SLURM_ARRAY_TASK_ID:-0}
NS=${#SPINS[@]}
MODEL=${MODELS[$(( I / NS ))]}
SPIN=${SPINS[$(( I % NS ))]}
OUT=$HOME/model_sp_ts/${TAG}_${MODEL}_spin${SPIN}.jsonl
[ -n "$MODEL" ] || { echo "ABBRUCH: kein Modell fuer Index $I"; exit 3; }
[ "$MODEL" = "mace-omol" ] && export PYTHONPATH=$HOME/venvs/e3nn044${PYTHONPATH:+:$PYTHONPATH}
echo "Task $I: $MODEL spin $SPIN (CPU, $OMP_NUM_THREADS threads) -> $OUT   $(date)"

cd $HOME/Transition1x
python3 pipeline/model_sp_ts.py --model $MODEL --spin $SPIN --device cpu ${RXNS:+--rxns $RXNS} --out $OUT
echo "ENDE Task $I   rc=$?   $(date)"
