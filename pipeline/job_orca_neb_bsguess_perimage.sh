#!/bin/bash
#SBATCH --job-name=nebbsimg
#SBATCH --partition=xeon24el8
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_neb_perimage/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_neb_perimage/slurm_%A_%a.err

# The actual repair attempt: every image starts on the broken sheet.
#
# What the earlier attempts established, in order:
#
#   MORead is rejected by the NEB module outright.
#   NEB_Restart_GBWName takes a BASENAME and reads one set of orbitals per
#     image, which is more than MORead would have given.
#   Handing it ten copies of one wavefunction from a foreign geometry
#     segfaults -- twice, and not for lack of memory (variant A, 4 procs and
#     48 GB, died at the same place as with 8 and 32).
#   Handing it orbitals that belong to the images runs fine -- variant B, 27
#     iterations, stable.
#
# So the condition is that each file belongs to the geometry it is read for.
# This job satisfies it the only way that also puts a BROKEN solution there:
# it computes one, per image, at that image's own coordinates.
#
#   1  take the converged path of the cheap baseline, image by image
#   2  per image a single point with STABPerform -> its own broken orbitals,
#      and the <S^2> of that solution recorded on the way
#   3  the NEB over the same path, no BrokenSym, those orbitals handed in
#
# Step 2 is worth having on its own: it says what the broken solution looks
# like at every point of the baseline path, which is the "before" picture
# against which step 3 is read.
#
# The comparison is clean because the geometries are the baseline's own:
#
#   bs_uks_neb_cheap/<rxn>        8 images, BrokenSym, same level
#   bs_uks_neb_perimage/<rxn>     same path, per-image broken orbitals
#
# rxn8827  baseline: broke at images 5-6, top at 7 restricted, result 0.019 A
#          from the RKS-TS.  Does a broken top change where it goes?
# rxn1320  baseline: never broke anywhere.  Does forcing the sheet at every
#          image find it at all?
# rxn8837  baseline: top already broken.  CONTROL -- must not get worse.
#
# As always the band's own SCFs never reach the main log; evaluation is the
# retroactive measurement over neb_im*.gbw.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn8827 rxn1320 rxn8837)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

W=$H/bs_uks_neb_perimage/$RXN
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

# ------------------------------------------------------------------ 1
BASE=$H/bs_uks_neb_cheap/$RXN
PATHF=$(ls $BASE/*_MEP.allxyz 2>/dev/null | head -1)
TRJ=$(ls $BASE/*_MEP_trj.xyz 2>/dev/null | head -1)
if [ -z "$TRJ" ]; then
  echo "ABBRUCH: keine Baseline-Trajektorie fuer $RXN"
  exit 2
fi
echo "Baseline-Pfad: ${PATHF:-$TRJ}"

NAT=$(head -1 $TRJ | tr -d ' \r')
awk -v nat="$NAT" 'BEGIN{i=-1}
    {if ((NR-1) % (nat+2) == 0) {i++; f=sprintf("img_%d.xyz", i)}
     print > f}' $TRJ
NIMG=$(ls img_*.xyz | wc -l)
echo "Bilder im Pfad: $NIMG"

# ------------------------------------------------------------------ 2
# One broken solution per image, at that image's own geometry.  STABPerform
# cannot run beside anything but a single point, which is why this is its own
# stage rather than part of the NEB.
echo ""
echo "--- Stufe 2: gebrochene Loesung je Bild ---"
: > s2_before.txt
NBROKEN=0
for k in $(seq 0 $((NIMG-1))); do
  cat > g$k.inp <<EOF
! UKS wB97X 6-31G(d) SP TightSCF SlowConv

%pal
  nprocs 8
end

%maxcore 3500

%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 500
end

* xyzfile 0 1 $W/img_${k}.xyz
EOF
  $ORCA g$k.inp > g$k.out 2> g$k.err
  S=$(grep 'Expectation value of <S\*\*2>' g$k.out | awk '{print $NF}' | tail -1)
  E=$(grep 'FINAL SINGLE POINT ENERGY' g$k.out | tail -1 \
      | sed -E 's/.*ENERGY[[:space:]]+(-?[0-9]+\.[0-9]+).*/\1/')
  [ -z "$S" ] && S=nan
  printf "%2d %10s %20s\n" "$k" "$S" "${E:-nan}" >> s2_before.txt
  printf "   im%-2d  <S^2> = %-10s  E = %s\n" "$k" "$S" "${E:-nan}"
  if [ -s g$k.gbw ]; then cp g$k.gbw guess_im${k}.gbw; fi
  awk -v s="$S" 'BEGIN{exit !(s > 0.3)}' && NBROKEN=$((NBROKEN+1))
  rm -f g$k.densities g$k*.tmp
done

echo ""
echo "  gebrochene Bilder auf dem Baseline-Pfad: $NBROKEN von $NIMG"
echo "  (das ist das Bild VOR dem Lauf -- die beste erreichbare Ausgangslage)"

if [ "$NBROKEN" -eq 0 ]; then
  echo "  HINWEIS: kein Bild dieses Pfades traegt eine gebrochene Loesung."
  echo "           Der Lauf geht weiter, kann aber nichts erzwingen, was es"
  echo "           nicht gibt."
fi
if [ "$(ls guess_im*.gbw 2>/dev/null | wc -l)" -ne "$NIMG" ]; then
  echo "ABBRUCH: nicht fuer jedes Bild Orbitale erzeugt"
  exit 3
fi

# ------------------------------------------------------------------ 3
# Same path, same level, no BrokenSym, the orbitals from stage 2.
# Preopt is off: the endpoints come from the baseline and are already relaxed,
# and relaxing them again would move the geometries away from the orbitals.
echo ""
echo "--- Stufe 3: NEB mit bildweise gebrochenen Startorbitalen ---"
cp img_0.xyz reactant.xyz
cp img_$((NIMG-1)).xyz product.xyz

cat > neb.inp <<EOF
! UKS wB97X 6-31G(d) NEB-TS TightSCF SlowConv

%pal
  nprocs 8
end

%maxcore 3500

%scf
  MaxIter 500
end

%neb
  Product "$W/product.xyz"
  NImages $((NIMG-2))
  MaxIter 500
  Preopt false
  PrintLevel 3
  NEB_Restart_GBWName "$W/guess"
end

* xyzfile 0 1 $W/reactant.xyz
EOF

$ORCA neb.inp > neb.out 2> neb.err
echo "rc=$?"

echo ""
echo "--- hat es ueberhaupt gerechnet ---"
grep -E 'kill-11|Child terminated|not implemented|THE NEB OPTIMIZATION HAS CONVERGED|ORCA TERMINATED NORMALLY' \
     neb.out | tail -4
echo "  LBFGS-Zeilen: $(grep -cE '^ +LBFGS' neb.out)"
echo "  Bildorbitale: $(ls neb_im*.gbw 2>/dev/null | wc -l) von $NIMG"

echo ""
echo "  Die Bandphase steht NICHT im Log. Auswertung ueber neb_im*.gbw,"
echo "  siehe job_orca_band_s2.sh. Das Vorher-Bild liegt in s2_before.txt."
echo "Finished $(date)"
