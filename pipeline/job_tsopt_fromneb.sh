#!/bin/bash
#SBATCH --job-name=tsfromneb
#SBATCH --partition=xeon24el8
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_tsopt_fromneb/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_tsopt_fromneb/slurm_%A_%a.err

# The three reactions the earlier TS-optimisation batches refused to start,
# because the broken-symmetry solution at the RKS reference geometry fell below
# the <S^2> > 0.3 gate:
#
#   rxn4113  <S^2> 0.14 at the reference, dE_BS only -8.4 meV
#   rxn6196  <S^2> 0.22
#   rxn5690  <S^2> 0.07
#
# The ORCA BS-NEB found strongly broken paths for all three anyway -- rxn4113
# runs at <S^2> ~ 1.0 across six consecutive images and its TS sits 0.93 A from
# the RKS reference, with the reactive C1-C3 contact no longer a bond at all.
# Reading: the RKS reference sits at the edge of the broken-symmetry region
# while the real saddle lies further along.
#
# Starting the same optimisation from the NEB TS tests that, and supplies the
# cross-check these three currently lack -- they are the only large deviations
# in the set without an independent confirmation.
#
# The gate is 0.05 here, not 0.3. The frequency job settled that question:
# rxn3107 (0.18) and rxn8885 (0.15) were flagged as failures by the 0.3 gate and
# both proved to be genuine transition states with exactly one imaginary
# frequency. The script checks the sign of lambda_min_ext instead, which is the
# criterion that actually decides whether a broken-symmetry solution exists.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn4113 rxn6196 rxn5690)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_tsopt_fromneb
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_tsopt_fromneb.py $RXN
RC=$?
echo "rc=$RC $(date)"
exit $RC
