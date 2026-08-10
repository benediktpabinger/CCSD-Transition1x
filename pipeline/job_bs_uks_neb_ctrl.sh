#!/bin/bash
#SBATCH --job-name=bsnebctl
#SBATCH --partition=xeon24el8
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=72G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_neb_results/slurm_ctrl_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_neb_results/slurm_ctrl_%A_%a.err

# Control: the same BS-UKS NEB on three reactions whose RKS reference TS is
# externally STABLE. All three sit in the high-N_FOD group, so FOD flags them as
# multireference while the orbital-Hessian criterion does not -- the sharpest
# available test of the two diagnostics against each other.
#
#   rxn7945  N_FOD 0.9033  lambda_min_ext +0.00434
#   rxn1150  N_FOD 0.8466  lambda_min_ext +0.00498
#   rxn7936  N_FOD 0.7271  lambda_min_ext +0.00814
#
# Expected: BrokenSym finds nothing to break, <S^2> stays 0 at every image, and
# the NEB-TS reproduces the RKS reference TS. A large deviation would mean the
# BS-NEB setup itself perturbs the result, which would undermine the 18-reaction
# batch; agreement makes that batch's deviations attributable to the broken
# symmetry rather than to the method change.

source /etc/profile
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0

ORCA=/home/modules/software/ORCA/5.0.4-gompi-2023a/bin/orca
export TMPDIR=/tmp

RXNS=(rxn7945 rxn1150 rxn7936)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_uks_neb_results
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

python3 /home/energy/s242862/pipeline/bs_uks_neb.py $RXN \
        --nprocs 8 --maxcore 8000 --orca-path $ORCA
RC=$?
echo "NEB rc=$RC $(date)"

if [ $RC -eq 0 ]; then
    python3 /home/energy/s242862/neb_images_engrad.py $RXN \
            --nprocs 8 --maxcore 8000 --orca-path $ORCA
    echo "engrad rc=$? $(date)"
fi

echo "rc=$RC $(date)"
exit $RC
