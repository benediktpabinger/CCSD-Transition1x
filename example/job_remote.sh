#!/bin/bash
#SBATCH --job-name=ccsd_rxn1961
#SBATCH --partition=xeon24el8
#SBATCH --array=0-9                  # 10 Jobs (0 bis 9)
#SBATCH --time=02:00:00              # max 2 Stunden pro Job (~27 Configs x 2 min)
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ccsd_%A_%a.log # Log-Datei pro Job

# --- Setup ---
mkdir -p logs

# gompi (OpenMPI) must be loaded before ORCA so shared libraries are available
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

# Prevent OpenMP from spawning extra threads that conflict with MPI (ORCA exit 125)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Berechne Config-Bereich für diesen Job ---
# 268 Configs aufgeteilt in 10 Jobs → je ~27
TOTAL=268
N_JOBS=10
CHUNK=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))   # = 27 (aufgerundet)

START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))

echo "Job ${SLURM_ARRAY_TASK_ID}: Configs ${START} bis $((END-1))"

# --- CCSD Berechnung ---
python ~/ccsd_slurm.py \
    --h5file ~/data/Transition1x.h5 \
    --reaction rxn1961 \
    --split test \
    --start-config ${START} \
    --end-config ${END} \
    --basis cc-pVDZ
