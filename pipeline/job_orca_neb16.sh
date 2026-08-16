#!/bin/bash
#SBATCH --job-name=neb16
#SBATCH --partition=xeon24el8
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_neb16/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_neb16/slurm_%A_%a.err

# Does a finer band put the top image on the broken sheet?
#
# With 8 images the broken region and the barrier top are, in the four
# interesting cases, one image apart.  The claim that this is a resolution
# problem does not depend on the energy-offset explanation, which was tested
# and refuted -- it only says that a denser band places its highest image
# closer to the true saddle, and if that saddle sits in the diradical region
# the top should fall inside it.
#
# Everything is identical to the cheap baseline except NImages, so the
# comparison is clean:
#
#   bs_uks_neb_cheap/<rxn>    8 images, same functional, basis, settings
#   bs_uks_neb16/<rxn>       16 images
#
# Three reactions, each asking something different:
#
#   rxn8827  8 images: broke at 5-6, top at 7, result 0.019 A from the RKS-TS.
#            The case the idea was built for.  Does the top move into the
#            broken region?
#   rxn1320  8 images: never broke anywhere, result 0.027 A from the RKS-TS.
#            Does a finer band find the broken sheet at all, or is this a
#            different failure?
#   rxn8837  8 images: top already broken, result 0.003 A from our structure.
#            CONTROL.  A finer band must not make this one worse; if it does,
#            the change itself is harmful and the other two prove nothing.
#
# As before, the band's own SCFs never reach the main log.  Evaluation is the
# retroactive measurement over neb_im*.gbw, see job_orca_band_s2.sh.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn8827 rxn1320 rxn8837)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

W=$H/bs_uks_neb16/$RXN
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
  NImages 16
  MaxIter 500
  Preopt true
  PrintLevel 3
end

* xyzfile 0 1 $W/reactant.xyz
EOF

$ORCA neb.inp > neb.out 2> neb.err
echo "rc=$?"

echo ""
echo "--- Ausgang ---"
grep -E 'THE NEB OPTIMIZATION HAS CONVERGED|HURRAY|ORCA TERMINATED NORMALLY|kill-11|finished by error' \
     neb.out | tail -4
echo "  LBFGS-Zeilen: $(grep -cE '^ +LBFGS' neb.out)"
echo "  Bildorbitale: $(ls neb_im*.gbw 2>/dev/null | wc -l) von 18"
ls *NEB-CI_converged.xyz *NEB-TS_converged.xyz 2>/dev/null

echo ""
echo "  ACHTUNG: das Hauptlog enthaelt die Band-SCFs nicht. Die Bandphase"
echo "  wird aus neb_im*.gbw nachgemessen. Diese Zeile hier sagt nichts"
echo "  ueber das Band aus:"
grep 'Expectation value of <S\*\*2>' neb.out | awk '{print $NF}' \
  | awk '$1<=1.8{n++; if($1>m)m=$1} END{printf "    aus PREOPT und TS-Opt: n=%d max=%.3f\n", n+0, m+0}'

echo "Finished $(date)"
