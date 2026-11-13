#!/bin/bash
#SBATCH --job-name=marksdft
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=180G
#SBATCH --output=/home/energy/bepab/logs/marksdft_%A_%a.out
#SBATCH --error=/home/energy/bepab/logs/marksdft_%A_%a.err

# DFT refinement of one model transition state per node (D32), ORCA 6.0.1:
#   1 uks_sp    UKS + STABPerform at the model structure
#   2 optts     OptTS from those orbitals, maximum 250 steps, starting Hessian a numerical
#               DFT Hessian (HESS=dft, default, as Marks et al. start from a DFT Hessian) or
#               the model Hessian model.hess (HESS=model, in the subfolder modelhess/)
#   3 stab      UKS + STABPerform at the DFT saddle; if the solution switched, back to 2 from
#               the stable orbitals (at most 3 rounds, D36); freq and IRC only after a
#               converged OptTS on a consistent solution
#   4 freq      NumFreq on the stable solution
#   5 irc       IRC from the DFT saddle with the frequency Hessian
# Finished steps are skipped on restart. Scratch is removed at the end.
#
#   N=$(wc -l < ~/marks_pilot/tasks.txt); sbatch --array=0-$(( N - 1 )) pipeline/job_marks_dft.sh
#   sbatch --array=0-4 --export=ALL,HESS=model,TASKS=$HOME/marks_pilot/tasks_model.txt pipeline/job_marks_dft.sh

source /etc/profile
source $HOME/opt/orca6.env
ORCA=$ORCADIR/orca
METHOD='wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3'
HEAD='%pal nprocs 24 end
%maxcore 3500'
SCF='  Thresh 1e-12
  TCut   1e-13
  MaxIter 300'
HESS=${HESS:-dft}
TASKS=${TASKS:-$HOME/marks_pilot/tasks.txt}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $TASKS)
RXN=$(echo "$LINE" | cut -d: -f1)
W=${ROOT:-$HOME/marks_pilot}/$RXN     # ROOT: another run folder (D44); STOP_AFTER=optts: no freq/IRC
if [ "$HESS" = model ]; then
  mkdir -p $W/modelhess && cp $W/ts_model.xyz $W/model.hess $W/modelhess/ && W=$W/modelhess
  GEOM='  InHess Read\n  InHessName "model.hess"'
else
  GEOM='  Calc_Hess true\n  NumHess true'
fi
cd $W || exit 2
ok () { grep -q "ORCA TERMINATED NORMALLY" $1.out 2>/dev/null; }
echo "$RXN start, starting Hessian $HESS, $(date)"

ok uks_sp || { printf '! UKS %s\n%s\n%%scf\n%s\n  STABPerform true\n  STABRestartUHFifUnstable true\nend\n* xyzfile 0 1 ts_model.xyz\n' "$METHOD" "$HEAD" "$SCF" > uks_sp.inp; $ORCA uks_sp.inp > uks_sp.out 2>&1; }
ok uks_sp || { echo "ABBRUCH uks_sp"; exit 7; }
echo "uks_sp done $(date)"

# OptTS, then a stability run at its last structure. If the solution switched (stability run
# unstable, a crossing between closed-shell and broken-symmetry at <S^2> 0.01 during OptTS, or
# <S^2> of OptTS and stability run more than 0.01 apart), OptTS is repeated from the stable
# orbitals at that structure, at most 3 times (D36). The count goes to stab_rounds; earlier
# rounds are kept in round<k>/. Only then is convergence required.
s2 () { grep "Expectation value of <S\*\*2>" $1 | sed -n "$2" | awk '{print $NF}'; }
START_GBW=uks_sp.gbw; START_XYZ=ts_model.xyz; R=0
while :; do
  ok optts || { cp $START_GBW optts_start.gbw; cp $START_XYZ optts_start.xyz
    printf '! UKS %s OptTS MORead\n%%moinp "optts_start.gbw"\n%s\n%%scf\n%s\nend\n%%geom\n'"$GEOM"'\n  MaxIter 250\nend\n* xyzfile 0 1 optts_start.xyz\n' "$METHOD" "$HEAD" "$SCF" > optts.inp
    $ORCA optts.inp > optts.out 2>&1; }
  ok optts && [ -f optts.xyz ] || { echo "ABBRUCH optts"; grep -i -m3 "error\|hess" optts.out; exit 8; }
  ok stab || { cp optts.gbw stab_start.gbw
    printf '! UKS %s MORead\n%%moinp "stab_start.gbw"\n%s\n%%scf\n%s\n  STABPerform true\n  STABRestartUHFifUnstable true\nend\n* xyzfile 0 1 optts.xyz\n' "$METHOD" "$HEAD" "$SCF" > stab.inp
    $ORCA stab.inp > stab.out 2>&1; }
  ok stab || { echo "ABBRUCH stab"; exit 9; }
  A=$(s2 optts.out 1p); B=$(s2 optts.out '$p'); C=$(s2 stab.out '$p')
  SWITCH=$(awk -v a=$A -v b=$B -v c=$C -v cut=0.01 'BEGIN { d = c - b; if (d < 0) d = -d
    print ((a > cut) != (b > cut) || d > cut) }')
  grep -q "wavefunction is unstable" stab.out && SWITCH=1
  echo "round $R: <S^2> OptTS $A -> $B, stability run $C, switch $SWITCH, $(date)"
  [ "$SWITCH" = 0 ] && break
  R=$(( R + 1 ))
  [ $R -gt 3 ] && { echo "ABBRUCH solution not resolved after 3 rounds"; echo $R > stab_rounds; exit 11; }
  mkdir -p round$R && mv optts* stab* round$R/
  START_GBW=round$R/stab.gbw; START_XYZ=round$R/optts.xyz
done
echo $R > stab_rounds
grep -q "THE OPTIMIZATION HAS CONVERGED" optts.out || { echo "ENDE optts not converged, no freq/IRC"; exit 0; }
echo "optts done $(date)"
if [ "$STOP_AFTER" = optts ]; then
  find . -maxdepth 2 -type f \( -name '*.tmp' -o -name '*.tmp.*' -o -name '*.densities*' -o -name '*_start.gbw' -o -name '*.ges' -o -name '*.bibtex' -o -name '*_property.txt' \) -delete
  echo "$RXN end after optts $(date)"; exit 0
fi

ok freq || { cp stab.gbw freq_start.gbw
  printf '! UKS %s NumFreq MORead\n%%moinp "freq_start.gbw"\n%s\n%%scf\n%s\nend\n* xyzfile 0 1 optts.xyz\n' "$METHOD" "$HEAD" "$SCF" > freq.inp
  $ORCA freq.inp > freq.out 2>&1; }
ok freq || { echo "ABBRUCH freq"; exit 10; }
echo "freq done $(date)"

ok irc || { cp stab.gbw irc_start.gbw
  printf '! UKS %s IRC MORead\n%%moinp "irc_start.gbw"\n%s\n%%scf\n%s\nend\n%%irc\n  InitHess Read\n  Hess_Filename "freq.hess"\n  MaxIter 80\nend\n* xyzfile 0 1 optts.xyz\n' "$METHOD" "$HEAD" "$SCF" > irc.inp
  $ORCA irc.inp > irc.out 2>&1; }
ok irc || echo "irc did not terminate normally"
find . -maxdepth 2 -type f \( -name '*.tmp' -o -name '*.tmp.*' -o -name '*.densities*' -o -name '*_start.gbw' -o -name '*.ges' -o -name '*.bibtex' -o -name '*_property.txt' \) -delete
echo "$RXN end $(date)"
