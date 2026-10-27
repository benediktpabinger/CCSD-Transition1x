#!/bin/bash
#SBATCH --job-name=om25tstest
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=00:40:00
#SBATCH --output=/home/energy/bepab/logs/omol25_ts_scan_test_%j.out

# Vorprobe fuer pipeline/job_omol25_ts_scan.sh, bevor der fuenfstuendige
# Download laeuft:
#   1  alle Uebergangszustaende aus Transition1x.h5 nach ~/t1x_ts_all.json
#   2  die Zuordnung des neuen Scanners an der alten Trefferdatei der 45
#      Reaktionen (~/omol25_hits_train.jsonl, alle Bilder) gegen
#      results/omol25_label_compare.csv pruefen
# Loescht nichts.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
cd $HOME/Transition1x
echo "START $(date)  $(hostname)"
python pipeline/omol25_dump_ts_all.py || exit 1
echo
python pipeline/omol25_stream_scan_ts.py --test $HOME/omol25_hits_train.jsonl
echo "ENDE $(date)  rc=$?"
