#!/bin/bash
#SBATCH --job-name=neb_diag
#SBATCH --partition=xeon24el8
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/s242862/neb_diag/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/neb_diag/slurm_%A_%a.err

# Why do 15 of 45 NEBs stall at a force noise floor (~0.05-0.11 eV/A) at the
# OMol25 settings, when the def2-TZVP reference run converged the same
# reactions in 6-11 steps?
#
# rxn4004 is the cleanest probe: old run 7 steps to fmax 0.0409, new run
# pinned at 0.1099 for 20+ steps. N_FOD 0.0095 -- lowest MR, so this is not a
# multireference effect.
#
# Variant A: def2-TZVP + DEFGRID3 + tight thresholds  -> isolates the BASIS
# Variant B: def2-TZVPD + default grid, no thresholds -> isolates GRID/THRESH
# Both keep MPI (nprocs 12), so if BOTH converge the cause is the combination,
# and if NEITHER does it is the parallelisation.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

H=/home/energy/s242862
RXN=rxn4004
V=$SLURM_ARRAY_TASK_ID
W=$H/neb_diag/v$V
mkdir -p $W; cd $W
export OMP_NUM_THREADS=1

if [ "$V" = "0" ]; then
  TAG="A_TZVP_grid3"
  SIMPLE='wB97M-V def2-TZVP def2/J RIJCOSX TightSCF DEFGRID3 EnGrad'
  BLOCKS='%pal nprocs 12 end\n%maxcore 3000\n%scf\n  maxiter 200\n  Thresh 1e-12\n  TCut   1e-13\nend'
else
  TAG="B_TZVPD_defaultgrid"
  SIMPLE='wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF EnGrad'
  BLOCKS='%pal nprocs 12 end\n%maxcore 3000\n%scf\n  maxiter 200\nend'
fi

echo "Variant $V ($TAG)  node $SLURM_NODELIST  start $(date)"
echo "simpleinput: $SIMPLE"

# patched copy of the driver, so the production script stays untouched
sed -e "s|^SIMPLEINPUT = .*|SIMPLEINPUT = ('$SIMPLE')|" \
    $H/pipeline/orca_neb_omol25.py > $W/driver.py
python3 - "$W/driver.py" "$BLOCKS" <<'PY'
import sys, re
p, blocks = sys.argv[1], sys.argv[2].replace('\\n', '\n')
s = open(p).read()
s = re.sub(r"    blocks = \(\n(?:.*\n)*?    \)\n",
           "    blocks = '''" + blocks + "'''\n", s, count=1)
open(p, 'w').write(s)
print('driver patched; blocks now:')
print(blocks)
PY

python3 $W/driver.py \
    --h5file   $H/data/Transition1x.h5 \
    --reaction $RXN \
    --split    test \
    --output   $W \
    --orca-cmd $(which orca) \
    --nprocs   12
RC=$?

echo ""
echo "=== $TAG  rc=$RC  $(date) ==="
[ -f $W/converged ] && echo "CONVERGED" || echo "NOT CONVERGED"
echo "--- neb.log ---"
cat $W/neb.log 2>/dev/null
echo "--- Referenz: alter TZVP-Lauf ---"
cat $H/orca_neb_results/$RXN/neb.log 2>/dev/null
exit $RC
