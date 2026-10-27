#!/bin/bash
#SBATCH --job-name=om25tsscan
#SBATCH --partition=xeon40el9
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --output=/home/energy/bepab/logs/omol25_ts_scan_%j.out
#SBATCH --error=/home/energy/bepab/logs/omol25_ts_scan_%j.err

# Der volle OMol25-Trainingssplit noch einmal, wie in
# pipeline/job_omol25_train_scan.sh (Job 10818210, Sep 2026, 5 h), diesmal mit
# pipeline/omol25_stream_scan_ts.py: behalten wird der Uebergangszustand JEDER
# Transition1x-Reaktion (~10 000 Zeilen), nicht nur die 45.
#
# Drei Stufen gleichzeitig, alles im Job, nichts auf dem Login-Knoten:
#   1  LADEN     train.tar.gz (455 GiB) in $NPARTS Stuecken ueber HTTP-Range,
#                $CONC gleichzeitig, jedes Stueck gegen seine Byte-Zahl geprueft
#   2  FUEGEN    der Reihe nach zusammenhaengen und entpacken, jedes Stueck
#                sofort loeschen -- auf Platte liegt nie der ganze Split
#   3  SCANNEN   jede fertige aselmdb lesen und loeschen
# Geloescht werden NUR die heruntergeladenen Dateien unter $D/train und $P.
#
# Braucht ~/t1x_ts_all.json aus pipeline/omol25_dump_ts_all.py
# (Vorprobe pipeline/job_omol25_ts_scan_test.sh, Job 11011077).
# Schreibt ~/omol25_ts_all.jsonl und ~/omol25_census_ts_all.json.

D=$HOME/omol25_probe
URL=https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train.tar.gz
TOTAL=488754888740
NPARTS=40
CONC=8

[ -s $HOME/t1x_ts_all.json ] || { echo "ABBRUCH: ~/t1x_ts_all.json fehlt"; exit 2; }

P=$D/parts
rm -rf $D/train $P $D/train_dl_rc
mkdir -p $D/train $P
cd $D

CHUNK=$(( (TOTAL + NPARTS - 1) / NPARTS ))
echo "START $(date)  $(hostname)"
echo "  $TOTAL Bytes, $NPARTS Stuecke zu je $CHUNK, $CONC gleichzeitig"

get_part() {
  local i=$1
  local s=$(( i * CHUNK ))
  local e=$(( s + CHUNK - 1 ))
  [ $e -ge $TOTAL ] && e=$(( TOTAL - 1 ))
  local want=$(( e - s + 1 ))
  local f=$P/part$i
  local try
  for try in 1 2 3 4 5 6; do
    if [ "$(stat -c%s $f 2>/dev/null)" = "$want" ]; then
      touch $f.done
      echo "  Stueck $i bereit ($want Bytes, Versuch $try)"
      return 0
    fi
    curl -s --fail --max-time 10800 -r ${s}-${e} -o $f "$URL"
    echo "  Stueck $i Versuch $try: curl=$? Groesse=$(stat -c%s $f 2>/dev/null)/$want"
    sleep 5
  done
  echo "ABBRUCH: Stueck $i nach 6 Versuchen unvollstaendig"
  touch $P/FEHLER
  return 1
}

# ---- 1  Laden; Nebenlaeufigkeit ueber 'wait -n' begrenzen
(
  running=0
  for i in $(seq 0 $((NPARTS - 1))); do
    [ -f $P/FEHLER ] && break
    get_part $i &
    running=$((running + 1))
    if [ $running -ge $CONC ]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
  echo "ALLE STUECKE GELADEN $(date)"
) > $D/parts.log 2>&1 &
DL=$!

# ---- 2  Der Reihe nach zusammenfuegen und entpacken, jedes Stueck sofort weg
(
  (
    for i in $(seq 0 $((NPARTS - 1))); do
      while [ ! -f $P/part$i.done ]; do
        [ -f $P/FEHLER ] && exit 1
        sleep 5
      done
      cat $P/part$i
      rm -f $P/part$i $P/part$i.done
    done
  ) | tar xz -C train
  echo $? > $D/train_dl_rc
) >> $D/parts.log 2>&1 &
UN=$!

# ---- 3  Der Scanner; train_dl_rc ist sein Halt-Signal
source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
python $HOME/Transition1x/pipeline/omol25_stream_scan_ts.py $D/train $D/train_dl_rc
RC=$?

wait $DL $UN
echo "ENDE $(date)   scan rc=$RC   entpacken rc=$(cat $D/train_dl_rc 2>/dev/null)"
rm -rf $D/train $P
