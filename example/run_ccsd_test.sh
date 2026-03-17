#!/bin/bash
#SBATCH --job-name=ccsd_test
#SBATCH --partition=xeon24el8
#SBATCH --time=05:00:00
#SBATCH --ntasks=4
#SBATCH --mem=16GB
#SBATCH --output=ccsd_test_%j.log
#SBATCH --mail-type=END,FAIL

# Lade benötigte Module
module purge
module load Python/3.13.5-GCCcore-14.3.0
module load Psi4/1.9.1-foss-2024a

# Wechsel zum Arbeitsverzeichnis
cd $SLURM_SUBMIT_DIR

# Zeige geladene Module
echo "Loaded modules:"
module list

echo ""
echo "==========================================="
echo "Starting CCSD test calculation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Date: $(date)"
echo "==========================================="
echo ""

# Führe das Python-Script aus
# ACHTUNG: Passen Sie die Reaktion und Parameter an!

# Zuerst verfügbare Reaktionen anzeigen (optional)
# python ccsd_test.py --list-reactions

# Beispiel: Eine Reaktion berechnen, limitiert auf 5 Konfigurationen zum Testen
python ccsd_test.py \
    --h5file data/Transition1x.h5 \
    --reaction "C2H4+H" \
    --max-configs 5 \
    --basis cc-pVDZ

echo ""
echo "==========================================="
echo "Job finished: $(date)"
echo "==========================================="
