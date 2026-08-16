#!/bin/bash
#SBATCH --job-name=nebgbw2
#SBATCH --partition=xeon24el8
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/s242862/bs_uks_neb_gbw2/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_uks_neb_gbw2/slurm_%A_%a.err

# Why the first attempt at NEB_Restart_GBWName produced nothing, and which of
# two explanations is right.
#
# That run died with a segfault at the first band iteration -- zero LBFGS
# lines, only the two endpoint orbitals written, seven minutes against the
# baseline's ninety.  The crash is in numfreq_utils.cpp, the module that spawns
# the per-image child processes, not in the SCF.  Two candidates:
#
#   A  resources.  8 procs x 3500 MB maxcore against --mem=32G, and this run
#      additionally reads and projects a set of orbitals per image.
#
#   B  the feature.  NEB_Restart_GBWName exists to CONTINUE an interrupted
#      band, so the orbitals it expects come from those very images.  The first
#      attempt handed it ten copies of one wavefunction from a different
#      geometry, which may be outside what it supports.
#
# The four tasks separate them.  Variant A repeats the experiment with half the
# processes and half again the memory.  Variant B feeds the feature what it was
# built for: the converged per-image orbitals of the baseline run on the same
# reaction, at the cheap level, which exist and match image for image.
#
#   A crashes, B runs   -> the foreign guess is the problem
#   both crash          -> the feature does not work in 5.0.4 for our purpose
#   A runs              -> it was memory, and the experiment is back on
#
# Variant B is also a null test with a known answer: restarted from its own
# collapsed orbitals and without BrokenSym, the band should stay collapsed.  If
# it does not, something is wrong with the comparison rather than with the NEB.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
CFG=(A:rxn8827 A:rxn8837 B:rxn8827 B:rxn8837)
ENTRY=${CFG[$SLURM_ARRAY_TASK_ID]}
VAR=${ENTRY%%:*}
RXN=${ENTRY##*:}

W=$H/bs_uks_neb_gbw2/${VAR}_${RXN}
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: Variante $VAR, $RXN, node $SLURM_NODELIST  $(date)"

cp $H/orca_neb_results/$RXN/reactant.xyz reactant.xyz
cp $H/orca_neb_results/$RXN/product.xyz product.xyz

# ---------------------------------------------------------------- Orbitale
# ORCA consumes the guess files as it reads them -- after the first attempt
# guess_im1 through guess_im8 were gone.  They are therefore written fresh
# here rather than reused.
if [ "$VAR" = "A" ]; then
  SRC=$H/bs_uks_neb_moread/$RXN/bsguess.gbw
  if [ ! -s "$SRC" ]; then
    echo "ABBRUCH: gebrochener Startraten fehlt ($SRC)"
    exit 2
  fi
  S2=$(grep 'Expectation value of <S\*\*2>' \
       $H/bs_uks_neb_moread/$RXN/bsguess.out | awk '{print $NF}' | tail -1)
  echo "Variante A: ein gebrochener Guess (<S^2> = $S2) fuer alle Bilder"
  for i in $(seq 0 9); do cp $SRC guess_im${i}.gbw; done
else
  BASE=$(ls $H/bs_uks_neb_cheap/$RXN/neb_im0.gbw 2>/dev/null | sed 's/_im0\.gbw$//')
  if [ -z "$BASE" ]; then
    echo "ABBRUCH: die Baseline hat keine Bildorbitale"
    exit 2
  fi
  echo "Variante B: die eigenen Bildorbitale der Baseline, $BASE"
  n=0
  for i in $(seq 0 9); do
    [ -f ${BASE}_im${i}.gbw ] && cp ${BASE}_im${i}.gbw guess_im${i}.gbw && n=$((n+1))
  done
  echo "  $n von 10 uebernommen"
  if [ "$n" -lt 10 ]; then
    echo "ABBRUCH: unvollstaendige Bildorbitale -- Variante B braucht alle"
    exit 2
  fi
fi
echo "Startorbitale bereit: $(ls guess_im*.gbw | wc -l)"

# ---------------------------------------------------------------- NEB
# Same as the baseline except: no BrokenSym, orbitals handed in, and half the
# processes with more memory each.
cat > neb.inp <<EOF
! UKS wB97X 6-31G(d) NEB-TS TightSCF SlowConv

%pal
  nprocs 4
end

%maxcore 3000

%scf
  MaxIter 500
end

%neb
  Product "$W/product.xyz"
  NImages 8
  MaxIter 500
  Preopt true
  PrintLevel 3
  NEB_Restart_GBWName "$W/guess"
end

* xyzfile 0 1 $W/reactant.xyz
EOF

$ORCA neb.inp > neb.out 2> neb.err
RC=$?
echo "rc=$RC"

echo ""
echo "--- ist es wieder abgestuerzt ---"
grep -E "kill-11|Child terminated|finished by error|ORCA TERMINATED NORMALLY" \
     neb.out | tail -3
echo "  LBFGS-Zeilen (0 = das Band lief nicht): $(grep -cE '^ +LBFGS' neb.out)"
echo "  Bildorbitale erzeugt: $(ls neb_im*.gbw 2>/dev/null | wc -l) von 10"

echo ""
echo "--- <S^2> im Log, nur zur Info ---"
echo "  ACHTUNG: das Hauptlog enthaelt die Band-SCFs NICHT. Die Werte unten"
echo "  stammen aus PREOPT und der TS-Optimierung. Das Band wird aus"
echo "  neb_im*.gbw nachgemessen, siehe job_orca_band_s2.sh."
grep 'Expectation value of <S\*\*2>' neb.out | awk '{print $NF}' \
  | awk '$1<=1.8{n++; if($1>m)m=$1} END{printf "  n=%d  max=%.3f\n", n+0, m+0}'

echo "Finished $(date)"
