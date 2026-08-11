#!/bin/bash
#SBATCH --job-name=frommodel
#SBATCH --partition=xeon24el8
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/home/energy/s242862/bs_tsopt_frommodel/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_tsopt_frommodel/slurm_%A_%a.err

# Search the second basin systematically instead of by accident.
#
# Every transition-state optimisation in this project started from the RKS
# reference geometry -- the very point we know is not the ground state. The
# optimiser then finds whichever saddle lies downhill of that point, so the
# starting geometry decides the answer. That is a systematic bias, not bad luck.
#
# Starting from a model geometry instead has been tried three times and found
# something all three times:
#
#   rxn4113   a second basin 0.93 A away, fully broken (-1940 meV) where the
#             reference is barely broken (-8 meV). Found only because UMA-M
#             happened to point there. Without that accident our saddle would
#             have been confirmed and the one model that was right would have
#             been counted as the worst failure -- which is exactly what the
#             RMSD table did.
#   rxn8885   1.475 A away, 425 meV lower, <S^2> 1.028 against 0.153. If it
#             holds up it moves UMA-S from 342 meV below our point to 84 above.
#   rxn1283   the first converged saddle for that reaction at all.
#
# The indicator is the `Faktor` column of the working document: strongest
# symmetry breaking over all four geometries divided by the value at the
# reference. A high factor means a model geometry sits in a far more strongly
# broken region, where a second saddle can hide.
#
#   task  rxn       factor  why this one
#   0     rxn8827     39x   currently counted for us (+20 meV, no rival)
#   1     rxn5690     25x   borderline case, dE_BS only -1.3 meV at the reference
#   2     rxn0894     21x   currently counted for us (+68 meV)
#   3     rxn8837     12x   currently our clearest win (+1034 meV)
#   4     rxn4522     12x   we have no saddle at all here, ran into walltime
#   5     rxn5691     11x   tight failed; the models already hold this one
#   6     rxn1320      1x   wrong saddle, mode fraction 0.00; tight failed
#   7     rxn4518      5x   wrong saddle, mode fraction 0.03; tight failed
#
# rxn8827, rxn0894 and rxn8837 are on our side of the ledger today. This run can
# take them away, and that is the point of running it.
#
# Output goes to bs_tsopt_frommodel/ so the tight attempts for rxn1320, rxn4518
# and rxn5691 in bs_tsopt_retry/ stay intact for comparison.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000
export TSOPT_OUT=/home/energy/s242862/bs_tsopt_frommodel

RXNS=(rxn8827 rxn5690 rxn0894 rxn8837 rxn4522 rxn5691 rxn1320 rxn4518)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p $TSOPT_OUT
echo "Task $SLURM_ARRAY_TASK_ID: $RXN (frommodel)  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_tsopt_retry.py $RXN frommodel
RC=$?
echo "rc=$RC $(date)"
exit $RC
