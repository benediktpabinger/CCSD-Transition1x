#!/bin/bash
#SBATCH --job-name=bsneb18
#SBATCH --partition=xeon24el8
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=72G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_neb_results/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_neb_results/slurm_%A_%a.err

# Broken-symmetry UKS NEB-TS for the 18 remaining reactions whose RKS reference
# transition state is externally unstable.  rxn8837 is already done and is the
# template: it converged cleanly and its NEB-TS agrees with the PySCF BS TS
# optimisation to 0.003 A.
#
# Settings copied verbatim from that successful run:
#   ! UKS wB97M-V def2-TZVP NEB-TS TightSCF SlowConv
#   %scf BrokenSym 1,1  MaxIter 500
#   %neb NImages 8  MaxIter 500  Preopt true
#
# ORCA runs three phases: endpoint pre-optimisation, NEB then CI-NEB, and a full
# saddle-point optimisation of the climbing image. The last phase is why this
# route is preferred over the ASE-driven one, which stops at the highest band
# image -- a band maximum sits beside the true saddle by construction.
#
# BrokenSym is applied at every single SCF: high-spin triplet first, then a beta
# flip. Stateless and robust, at the cost of falling back to RKS where the
# diradical character is weak -- which is physically correct at the band ends,
# past the Coulson-Fischer point the broken-symmetry solution does not exist.
#
# ORCA parallelises over MPI ranks, so SLURM gets ntasks, not cpus-per-task, and
# the binary needs an absolute path. rxn8837 converged with nprocs 8, so MPI is
# not a problem for ORCA's own NEB module -- unlike the ASE-driven variant,
# where ORCA restarts per single point and COSX grids can differ between calls.
#
# Partition limit is 2-02:00:00; 2 days requested.

source /etc/profile
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0

ORCA=/home/modules/software/ORCA/5.0.4-gompi-2023a/bin/orca
export TMPDIR=/tmp

RXNS=(rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn6196 rxn0346 rxn4518 rxn3107 \
      rxn7060 rxn5691 rxn1283 rxn8827 rxn4522 rxn1147 rxn0894 rxn7957 rxn5690)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_uks_neb_results
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

python3 /home/energy/s242862/pipeline/bs_uks_neb.py $RXN \
        --nprocs 8 --maxcore 8000 --orca-path $ORCA
RC=$?
echo "NEB rc=$RC $(date)"

# Energies, forces and <S^2> per final band image -> images_bs.extxyz.
# ORCA's NEB writes geometries and energies but no gradients, so this adds one
# EnGrad single point per image. Ten points against the hundreds the NEB itself
# ran, so the extra cost is small.
if [ $RC -eq 0 ]; then
    python3 /home/energy/s242862/neb_images_engrad.py $RXN \
            --nprocs 8 --maxcore 8000 --orca-path $ORCA
    echo "engrad rc=$? $(date)"
fi

echo "rc=$RC $(date)"
exit $RC
