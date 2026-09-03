#!/bin/bash
#SBATCH --job-name=neb25res
#SBATCH --partition=xeon24el8
#SBATCH --array=0-4
#SBATCH --nodes=1
# ORCA parallelisiert ueber MPI-Raenge -> ntasks, nicht cpus-per-task
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home/energy/s242862/orca_neb_omol25/resume_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_neb_omol25/resume_%A_%a.err

# Fortsetzung der fuenf OMol25-NEBs, die in die 24-h-Wanduhr gelaufen sind.
# Keine ist abgestuerzt; neb.db liegt jeweils vor, das Band wird von dort
# warmgestartet statt aus dem H5 neu interpoliert.
#
#   rxn10054    65 Schritte   fmax 0.0510
#   rxn7949     78 Schritte   fmax 0.0580
#   rxn3107    100 Schritte   fmax 0.0682
#   rxn5690     52 Schritte   fmax 0.0694
#   rxn0894    101 Schritte   fmax 0.1433
#
# Ziel ist CI-NEB bis fmax 0.05, wie im Erstlauf. Niveau unveraendert:
# wB97M-V/def2-TZVPD, RIJCOSX, TightSCF, DEFGRID3, Thresh 1e-12, TCut 1e-13.
# Die Endpunkte werden nicht neu relaxiert, reactant.xyz und product.xyz
# stehen aus dem Erstlauf und sind auf demselben Niveau entstanden.
#
# Diese fuenf sind genau die Reaktionen, bei denen der UMA-M-TS RKS-instabil
# ist und noch kein OMol25-TS vorliegt -- die Luecke fuer Figur 4.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

H=/home/energy/s242862
RXNS=(rxn10054 rxn7949 rxn3107 rxn5690 rxn0894)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
OUT=$H/orca_neb_omol25/$RXN

export OMP_NUM_THREADS=1

echo "Task $SLURM_ARRAY_TASK_ID: $RXN   node $SLURM_NODELIST   start $(date)"

if [ -f "$OUT/converged" ]; then
  echo "SKIP: $OUT/converged liegt bereits vor"
  exit 0
fi
if [ ! -f "$OUT/neb.db" ]; then
  echo "ABBRUCH: $OUT/neb.db fehlt, ein Warmstart ist nicht moeglich"
  exit 3
fi

# Sicherungskopie, falls der Warmstart das Band verschlechtert
cp "$OUT/neb.db" "$OUT/neb_before_resume.db"
cp "$OUT/neb.log" "$OUT/neb_before_resume.log" 2>/dev/null

python3 $H/pipeline/orca_neb_omol25.py \
    --h5file   $H/data/Transition1x.h5 \
    --reaction $RXN \
    --split    test \
    --output   $OUT \
    --orca-cmd $(which orca) \
    --nprocs   12 \
    --resume
RC=$?

echo "rc=$RC   ende $(date)"
[ -f $OUT/converged ] && echo "CONVERGED" || echo "NOT CONVERGED"
tail -2 $OUT/neb.log 2>/dev/null
exit $RC
