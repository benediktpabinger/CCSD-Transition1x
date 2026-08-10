#!/bin/bash
#SBATCH --job-name=tsnull
#SBATCH --partition=xeon24el8
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/s242862/tsopt_null/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/tsopt_null/slurm_%A_%a.err

# Null measurement for the TS-optimisation route (method B), which produced the
# 13 confirmed broken-symmetry transition states.
#
# Four reactions whose RKS reference TS is externally STABLE, so there is no
# broken-symmetry solution to move onto. Whatever displacement comes out is the
# method's own noise: PySCF against an ORCA-optimised starting geometry, with
# different convergence criteria.
#
#   rxn7945  N_FOD 0.9033   also a NEB control, so the two routes are comparable
#   rxn1150  N_FOD 0.8466   likewise
#   rxn7936  N_FOD 0.7271   likewise
#   rxn0101  N_FOD 0.7132   lambda_min_ext +0.0714, the most clearly stable case
#
# The NEB control gave 0.095 and 0.669 A on the first two. If method B stays
# near 0.02 A, the 13 confirmed results carry their stated RMSDs. If it also
# lands near 0.3 A, both routes have a resolution problem and the geometric
# claim has to be weakened -- the energies and the stability finding are
# unaffected either way, since neither involves an optimisation.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn7945 rxn1150 rxn7936 rxn0101)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/tsopt_null
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/tsopt_null.py $RXN
RC=$?
echo "rc=$RC $(date)"
exit $RC
