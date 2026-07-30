#!/bin/bash
#SBATCH --job-name=neb25_test
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
# ORCA parallelises over MPI ranks -> ntasks, not cpus-per-task
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/orca_neb_omol25/slurm_test_%j.out
#SBATCH --error=/home/energy/s242862/orca_neb_omol25/slurm_test_%j.err

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

H=/home/energy/s242862
RXN=${1:-rxn1320}
OUT=$H/orca_neb_omol25/$RXN
mkdir -p $OUT

# ASE spawns one ORCA per image sequentially; each ORCA uses the 12 MPI ranks.
export OMP_NUM_THREADS=1

echo "TEST: $RXN  node $SLURM_NODELIST  ntasks=$SLURM_NTASKS  start $(date)"
echo "orca: $(which orca)"

python3 $H/pipeline/orca_neb_omol25.py \
    --h5file   $H/data/Transition1x.h5 \
    --reaction $RXN \
    --split    test \
    --output   $OUT \
    --orca-cmd $(which orca) \
    --nprocs   12

RC=$?
echo "rc=$RC  finished $(date)"

echo ""
echo "=== endpoint relaxation ==="
tail -3 $OUT/relax_r.log 2>/dev/null
tail -3 $OUT/relax_p.log 2>/dev/null
echo "=== NEB progress ==="
head -3 $OUT/neb.log 2>/dev/null
tail -3 $OUT/neb.log 2>/dev/null
echo "=== converged? ==="
[ -f $OUT/converged ] && echo YES || echo NO
exit $RC
