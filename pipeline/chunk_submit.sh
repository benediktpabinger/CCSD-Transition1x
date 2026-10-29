#!/bin/bash
# Reicht ein grosses SLURM-Array in Stuecken ein, damit das Konto-Limit von
# 100 eingereichten Elementen (MaxSubmit) nie ueberschritten wird: sobald
# weniger als MAXQ Elemente in der Queue stehen, kommt das naechste Stueck.
# Der Zustand (naechster Startindex) liegt in STATEFILE, deshalb kann das
# Skript jederzeit abgebrochen und neu gestartet werden.
#
#   chunk_submit.sh <statefile> <ntasks_total> <chunk> "<sbatch-optionen>" <jobscript>
#   z. B.
#   chunk_submit.sh ~/logs/s2pos.state 639 46 "--job-name=s2pos --export=ALL,TASKFILE=$HOME/t1x_ts_s2pos_tasks.txt,OUTROOT=$HOME/orca6_s2pos --output=$HOME/logs/s2pos_%A_%a.out --error=$HOME/logs/s2pos_%A_%a.err" pipeline/job_border_zone.sh
#
# Endet, wenn alle Stuecke eingereicht sind UND die Queue leer ist.
STATE=$1; NTOT=$2; CHUNK=$3; OPTS=$4; SCRIPT=$5
MAXQ=${MAXQ:-54}          # 54 + 46 = 100
[ -f "$STATE" ] || echo 0 > "$STATE"
while true; do
  NEXT=$(cat "$STATE")
  NQ=$(squeue -u "$USER" -h -r -o '%A' | wc -l)
  if [ "$NEXT" -lt "$NTOT" ] && [ "$NQ" -le "$MAXQ" ]; then
    LAST=$(( NEXT + CHUNK - 1 )); [ "$LAST" -ge "$NTOT" ] && LAST=$(( NTOT - 1 ))
    OUT=$(sbatch $OPTS --array=${NEXT}-${LAST}%18 "$SCRIPT" 2>&1)
    echo "$(date '+%F %T') tasks ${NEXT}-${LAST}: $OUT"
    if echo "$OUT" | grep -q "^Submitted"; then echo $(( LAST + 1 )) > "$STATE"; else sleep 600; continue; fi
    sleep 30; continue
  fi
  if [ "$NEXT" -ge "$NTOT" ] && [ "$NQ" -eq 0 ]; then echo "$(date '+%F %T') alle $NTOT Tasks durch"; break; fi
  sleep 300
done
