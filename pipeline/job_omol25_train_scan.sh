#!/bin/bash
#SBATCH --job-name=omol25scan
#SBATCH --partition=xeon40el8_768
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --output=/home/energy/s242862/omol25_train_scan_%j.out
#SBATCH --error=/home/energy/s242862/omol25_train_scan_%j.err

# Der volle OMol25-Trainingssplit, ohne ihn je ganz auf Platte zu haben.
#
# train.tar.gz sind 455 GiB. Die Heimatquote gibt das nicht her und braucht es
# nicht: tar entpackt der Reihe nach, also kann ein zweiter Prozess jede
# fertige aselmdb sofort scannen und wieder loeschen.
#
# WARUM ALS SLURM-JOB   Auf dem Login-Knoten wurde der Lauf abgeraeumt. Der
# Knoten hat ausserdem nur geteilte Kerne; hier laufen die vier Scanprozesse
# ungestoert, und zwoelf Stunden Laufzeit sind zugesagt statt geduldet.
# Rechenknoten haben Netz nach draussen, geprueft.
#
# DREI STUFEN, ALLE GLEICHZEITIG
#
#   1  LADEN     $NPARTS Stuecke ueber HTTP-Range, $CONC davon gleichzeitig.
#      Eine einzelne Verbindung liefert rund 23 MB/s, mehrere zusammen ein
#      Vielfaches -- gedrosselt wird pro Verbindung. Der zweite Grund ist die
#      Haltbarkeit: ein einziger HTTP-Strom ueber fuenf Stunden ist ein
#      Einzelpunkt, an dem alles verloren geht. Jedes Stueck wird gegen seine
#      erwartete Byte-Zahl geprueft und bei Abbruch neu geholt.
#
#   2  FUEGEN    Ein Leser haengt die Stuecke der Reihe nach aneinander und
#      loescht jedes sofort nach dem Lesen. Auf Platte liegen deshalb nie mehr
#      als die gerade laufenden Stuecke, nicht die vollen 455 GiB.
#
#   3  SCANNEN   pipeline/omol25_stream_scan.py liest jede fertige aselmdb und
#      loescht sie.
#
# Gesucht wird: die fuenf Summenformeln der 45 Label-Geometrien und die 45
# Reaktionsnamen. In OMol25 heisst Transition1x data_id 'trans1x', und die
# Herkunft traegt die Reaktion im Klartext:
#     trans1x/t1x_rxn9839_1010_943_0_1/orca.tar.zst
#
# Schreibt ~/omol25_hits_train.jsonl und ~/omol25_census_train.json.

D=/home/energy/s242862/omol25_probe
URL=https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train.tar.gz
TOTAL=488754888740
NPARTS=40
CONC=8

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

# ---- 1  Laden. Die Nebenlaeufigkeit ueber 'wait -n' begrenzen, NICHT ueber
#         jobs in einer Kommandosubstitution -- die laeuft in einer Subshell
#         und zaehlt dort nichts, dann starten alle 40 auf einmal.
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

# ---- 3  Der Scanner. train_dl_rc ist sein Halt-Signal.
source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
python /home/energy/s242862/omol25_stream_scan.py $D/train $D/train_dl_rc
RC=$?

wait $DL $UN
echo "ENDE $(date)   scan rc=$RC   entpacken rc=$(cat $D/train_dl_rc 2>/dev/null)"
rm -rf $D/train $P
