#!/bin/bash
#SBATCH --job-name=nebcheap
#SBATCH --partition=xeon24el8
#SBATCH --array=0-3
#SBATCH --nodes=1
# ORCA parallelises over MPI processes, not threads -> ntasks, not cpus-per-task
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_neb_cheap/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_neb_cheap/slurm_%A_%a.err

# The BS-NEB baseline at wB97X/6-31G(d).
#
# This is NOT a fix. BrokenSym is still stateless -- it re-derives the broken
# guess at every SCF -- the images are still relaxed independently, and
# STABPerform still refuses to run alongside anything but a single point. The
# same failure is expected, only faster.
#
# It is run anyway because it is the baseline. A repaired NEB that keeps the
# broken solution across the band proves nothing unless the unrepaired one is
# shown to lose it at the same level of theory, on the same reactions, with
# everything else held fixed. Without that, a run that works is indistinguish-
# able from a run that got lucky.
#
# Everything is copied from the production input except the level of theory:
# same NImages, same MaxIter, same Preopt, same BrokenSym 1,1. Results go to
# bs_uks_neb_cheap/ so bs_uks_neb_results/ stays intact for comparison.
#
# Three failures and one control, all unstable at this level:
#
#   rxn7949  cheap dE_BS -837 meV   production: no converged structure at all
#   rxn1320  cheap dE_BS -543       production: structure, but <S^2> = 0 over
#                                   the whole band -- an RKS run in disguise
#   rxn8827  cheap dE_BS -167       production: same, band entirely RKS
#   rxn8837  cheap dE_BS -507       CONTROL: production band held the breaking
#                                   over 19 of 29 images and landed 0.003 A
#                                   from our structure. If this one fails here
#                                   too, the cheap setup is at fault rather
#                                   than the method.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn7949 rxn1320 rxn8827 rxn8837)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

W=$H/bs_uks_neb_cheap/$RXN
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

cp $H/orca_neb_results/$RXN/reactant.xyz reactant.xyz
cp $H/orca_neb_results/$RXN/product.xyz product.xyz

cat > neb.inp <<EOF
! UKS wB97X 6-31G(d) NEB-TS TightSCF SlowConv

%pal
  nprocs 8
end

%maxcore 3500

%scf
  BrokenSym 1,1
  MaxIter 500
end

%neb
  Product "$W/product.xyz"
  NImages 8
  MaxIter 500
  Preopt true
  PrintLevel 3
end

* xyzfile 0 1 $W/reactant.xyz
EOF

$ORCA neb.inp > neb.out 2> neb.err
echo "rc=$?"

echo ""
echo "--- did it converge ---"
grep -E 'THE NEB OPTIMIZATION HAS CONVERGED|HURRAY|ORCA TERMINATED NORMALLY|finished by error' \
     neb.out | tail -4
ls -la *NEB-CI_converged.xyz *NEB-TS_converged.xyz 2>/dev/null

echo ""
echo "--- <S**2> over the run, high-spin reference separated out ---"
grep 'Expectation value of <S\*\*2>' neb.out | awk '{print $NF}' > s2_all.txt
awk '$1 > 1.8 {h++} $1 <= 1.8 {b++; s+=$1; if($1>mx)mx=$1}
     END {printf "  high-spin refs %d   band values %d   max %.3f   mean %.3f\n",
          h+0, b+0, mx+0, (b?s/b:0)}' s2_all.txt
awk '$1 <= 1.8 && $1 > 0.3 {c++} END {printf "  band values above 0.3: %d\n", c+0}' s2_all.txt

echo "Finished $(date)"
