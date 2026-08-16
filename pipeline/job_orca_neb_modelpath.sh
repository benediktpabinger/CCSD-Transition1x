#!/bin/bash
#SBATCH --job-name=nebmpath
#SBATCH --partition=xeon24el8
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_neb_modelpath/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_neb_modelpath/slurm_%A_%a.err

# Start the DFT band from the model's band instead of from a straight line.
#
# Every band in this project begins as an interpolation between reactant and
# product.  The diradical region sits somewhere in the middle of that line, and
# which images land on which sheet is left to chance -- which is what the
# retroactive measurement showed: 14 of 19 bands break somewhere, but only 11
# at the image that becomes the transition state.
#
# UMA-M has a converged band for every reaction, all images, in neb.db.  Using
# it as the starting path costs nothing and changes the starting position
# completely.
#
# Not circular: this uses a model prediction, not our own structure -- exactly
# what someone doing the calculation for the first time would have.  The
# objection that ruled out feeding in our BS-TS as a guess does not apply.
#
# It also tests the chapter's own recommendation.  Section 2.5 says: model as a
# starting point, DFT as the refinement, established at 6 of 7 for single
# geometries.  At band level it has never been tried, and that is the form in
# which people actually use these models.
#
# rxn8827  model band converged (fmax 0.049).  Our baseline: band restricted at
#          the top, result on the RKS-TS.
# rxn1320  model band converged (0.045).  Our baseline: never converged.
# rxn8837  model band did NOT converge (0.270 after 454 steps) -- the starting
#          path is itself questionable here.  Included as the case where the
#          idea should be weakest, not as a control.
#
# The model bands for rxn8885, rxn0894 and rxn7949 are not converged either
# (section 2.3b) and are left out.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn8827 rxn1320 rxn8837)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

W=$H/bs_uks_neb_modelpath/$RXN
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

# ------------------------------------------------------------------ 1
# The converged model band is the LAST set of images in neb.db, which holds
# every image of every iteration back to back.
module load Python/3.13.5-GCCcore-14.3.0 2>/dev/null
module load ASE 2>/dev/null

python3 - "$H/uma_m_neb_results/$RXN/neb.db" "$W" <<'EOF'
import sys
from ase.db import connect
from ase.io import write

db, out = sys.argv[1], sys.argv[2]
rows = list(connect(db).select())
if not rows:
    raise SystemExit('neb.db ist leer')
nat = rows[0].natoms
# the band width has to divide the row count, otherwise the tail is not a band
for n in (10, 8, 12):
    if len(rows) % n == 0:
        nimg = n
        break
else:
    raise SystemExit('Zeilenzahl %d passt zu keiner Bandbreite' % len(rows))
last = rows[-nimg:]
print('  %d Zeilen, Bandbreite %d, letzte %d als konvergiertes Band'
      % (len(rows), nimg, nimg))
for k, r in enumerate(last):
    write('%s/img_%d.xyz' % (out, k), r.toatoms(), format='xyz')
    try:
        print('    im%-2d  E = %.6f eV' % (k, r.energy))
    except Exception:
        print('    im%-2d' % k)
open('%s/nimg.txt' % out, 'w').write(str(nimg))
EOF
[ $? -eq 0 ] || { echo "ABBRUCH: Modellband nicht lesbar"; exit 2; }

NIMG=$(cat nimg.txt)
cp img_0.xyz reactant.xyz
cp img_$((NIMG-1)).xyz product.xyz

# ORCA reads a starting path as .allxyz: xyz blocks separated by a single ">"
: > modelpath.allxyz
for k in $(seq 0 $((NIMG-1))); do
  cat img_${k}.xyz >> modelpath.allxyz
  [ $k -lt $((NIMG-1)) ] && echo ">" >> modelpath.allxyz
done
echo "Startpfad: $NIMG Bilder in modelpath.allxyz"

# ------------------------------------------------------------------ 2
# Same level and settings as the cheap baseline.  Preopt is off because the
# endpoints come from the model band and moving them would break the
# correspondence with the rest of the path.
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
  NImages $((NIMG-2))
  MaxIter 500
  Preopt false
  PrintLevel 3
  NEB_Restart_XYZFile "$W/modelpath.allxyz"
end

* xyzfile 0 1 $W/reactant.xyz
EOF

$ORCA neb.inp > neb.out 2> neb.err
echo "rc=$?"

echo ""
echo "--- hat ORCA den Startpfad angenommen ---"
grep -iE "restart|allxyz|not implemented|images" neb.out | head -6
grep -E 'kill-11|Child terminated|finished by error' neb.out | tail -2
echo "  LBFGS-Zeilen: $(grep -cE '^ +LBFGS' neb.out)"
echo "  Bildorbitale: $(ls neb_im*.gbw 2>/dev/null | wc -l) von $NIMG"

echo ""
echo "--- Ausgang ---"
grep -E 'THE NEB OPTIMIZATION HAS CONVERGED|HURRAY|ORCA TERMINATED NORMALLY' \
     neb.out | tail -3
ls *NEB-CI_converged.xyz *NEB-TS_converged.xyz 2>/dev/null

echo ""
echo "  Die Bandphase steht nicht im Log -- Auswertung ueber neb_im*.gbw."
echo "Finished $(date)"
