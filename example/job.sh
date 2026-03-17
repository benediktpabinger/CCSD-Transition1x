#!/bin/bash
#SBATCH --job-name=ccsd_job
#SBATCH --partition=xeon24el8
#SBATCH --array=0-9                  # 10 Jobs (0 bis 9)
#SBATCH --time=02:00:00              # max 2 Stunden pro Job
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/ccsd_%A_%a.log # Log-Datei pro Job

# ============================================================
# EINSTELLUNGEN — hier anpassen
# ============================================================
REACTION=${1:-rxn1961}   # Reaktion als Argument: sbatch job.sh rxn1234
                         # oder Standard rxn1961 wenn kein Argument
H5FILE=~/data/Transition1x.h5
SPLIT=test
BASIS=cc-pVDZ
N_JOBS=10
# ============================================================

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

# --- Anzahl Configs automatisch aus dem HDF5 lesen ---
TOTAL=$(python3 -c "
from transition1x import Dataloader
dl = Dataloader('${H5FILE}', '${SPLIT}')
print(sum(1 for c in dl if c['rxn'] == '${REACTION}'))
")
echo "Reaction: ${REACTION} — ${TOTAL} configs total"

# --- Berechne Config-Bereich für diesen Job ---
CHUNK=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))

echo "Job ${SLURM_ARRAY_TASK_ID}: Configs ${START} bis $((END-1))"

# --- CCSD Berechnung ---
python ~/ccsd_slurm.py \
    --h5file ${H5FILE} \
    --reaction ${REACTION} \
    --split ${SPLIT} \
    --start-config ${START} \
    --end-config ${END} \
    --basis ${BASIS}
