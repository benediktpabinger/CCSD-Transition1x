#!/bin/bash
#SBATCH --job-name=freqfneb
#SBATCH --partition=xeon24el8
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_freq_fromneb/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_freq_fromneb/slurm_%A_%a.err

# Frequencies for the two structures obtained by restarting the TS optimisation
# from the ORCA BS-NEB transition state, after the original run aborted at the
# RKS reference where <S^2> was below the (wrong) 0.3 gate.
#
#   rxn4113  <S^2> 0.97 at the NEB geometry vs 0.14 at the RKS reference;
#            the optimisation stayed within 0.008 A of the NEB structure
#   rxn6196  <S^2> 0.23; stayed within 0.134 A
#
# Confirming one imaginary frequency would put both into the confirmed set and,
# for rxn4113, establish that the second basin holds a genuine saddle point
# rather than merely a lower region.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000
export BSFREQ_SRC=bs_tsopt_fromneb
export BSFREQ_OUT=bs_freq_fromneb

RXNS=(rxn4113 rxn6196)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_freq_fromneb
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_freq2.py $RXN
RC=$?
echo "rc=$RC $(date)"
exit $RC
