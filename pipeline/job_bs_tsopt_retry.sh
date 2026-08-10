#!/bin/bash
#SBATCH --job-name=bsretry
#SBATCH --partition=xeon24el8
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=36:00:00
#SBATCH --output=/home/energy/s242862/bs_tsopt_retry/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_tsopt_retry/slurm_%A_%a.err

# Second attempt at the five transition states that did not come out right.
#
# tight      the saddle found is real but belongs to a different motion -- its
#            imaginary mode does not stretch the reactive bonds. Restart from
#            the RKS reference with a tight trust radius so the optimiser cannot
#            bolt out of the basin before the reaction mode takes over.
#   rxn1320  mode fraction 0.00; the breaking C-H ran from 1.98 to 3.36 A
#   rxn4518  mode fraction 0.03 at 89 cm-1
#   rxn5691  bond rate 0.01 at 102 cm-1
#
# frommodel  the saddle may be in the wrong basin. Restart from the model
#            geometry with the deepest broken-symmetry solution, as was done
#            successfully for rxn4113.
#   rxn8885  optimised from S^2 0.507 down to 0.153 while a model geometry is
#            71x more strongly broken
#   rxn1283  never converged; factor 46x
#
# Results go to bs_tsopt_retry/ so the earlier runs stay intact for comparison.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn1320  rxn4518  rxn5691  rxn8885    rxn1283)
MODES=(tight   tight    tight    frommodel  frommodel)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_tsopt_retry
echo "Task $SLURM_ARRAY_TASK_ID: $RXN ($MODE)  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_tsopt_retry.py $RXN $MODE
RC=$?
echo "rc=$RC $(date)"
exit $RC
