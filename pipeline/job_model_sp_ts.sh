#!/bin/bash
#SBATCH --job-name=model_sp
#SBATCH --partition=a100el9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/energy/bepab/logs/model_sp_%A_%a.out
#SBATCH --error=/home/energy/bepab/logs/model_sp_%A_%a.err

# Modell-Einzelpunkte an den gespeicherten Transition1x-Uebergangszustaenden
# (pipeline/model_sp_ts.py), ein Array-Task je (Modell, Spin):
#   MODELS/SPINS als Leerzeichen-Listen per --export; Task i nimmt das
#   i-te Paar aus dem kartesischen Produkt (Modelle aussen, Spins innen).
# RXNS: optionale Reaktionsliste (Spin-Test an den 45 Ankern); sonst alle.
#
#   Spin-Test:  sbatch --export=ALL,MODELS="uma-s uma-m mace-omol",SPINS="0 1 2",RXNS=$HOME/anchors45.txt,TAG=spintest --array=0-8 pipeline/job_model_sp_ts.sh
#   Vollauf:    sbatch --export=ALL,MODELS="uma-s uma-m mace-omol",SPINS="1",TAG=all --array=0-2 pipeline/job_model_sp_ts.sh

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
LIBCUDA=$(find /usr -name "libcuda.so.1" 2>/dev/null | head -1)
[ -n "$LIBCUDA" ] && export LD_PRELOAD=$LIBCUDA
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p $HOME/logs $HOME/model_sp_ts

MODELS=(${MODELS:-uma-s uma-m mace-omol})
SPINS=(${SPINS:-1})
TAG=${TAG:-all}
I=${SLURM_ARRAY_TASK_ID:-0}
NS=${#SPINS[@]}
MODEL=${MODELS[$(( I / NS ))]}
SPIN=${SPINS[$(( I % NS ))]}
OUT=$HOME/model_sp_ts/${TAG}_${MODEL}_spin${SPIN}.jsonl
[ -n "$MODEL" ] || { echo "ABBRUCH: kein Modell fuer Index $I"; exit 3; }
# mace-torch 0.3.15 braucht e3nn 0.4.4, fairchem 2.20 e3nn >= 0.5 (0.6.0 in
# ~/.local). Fuer MACE liegt e3nn 0.4.4 getrennt in ~/venvs/e3nn044 und wird
# per PYTHONPATH vorangestellt; alles andere kommt weiter aus ~/.local.
[ "$MODEL" = "mace-omol" ] && export PYTHONPATH=$HOME/venvs/e3nn044${PYTHONPATH:+:$PYTHONPATH}
echo "Task $I: $MODEL spin $SPIN -> $OUT   $(date)"; nvidia-smi -L 2>/dev/null | head -1

cd $HOME/Transition1x
python3 pipeline/model_sp_ts.py --model $MODEL --spin $SPIN ${RXNS:+--rxns $RXNS} --out $OUT
echo "ENDE Task $I   rc=$?   $(date)"
