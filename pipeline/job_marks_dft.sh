#!/bin/bash
#SBATCH --job-name=marksdft
#SBATCH --partition=xeon24el9
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem=180G
#SBATCH --output=/home/energy/bepab/logs/marksdft_%A_%a.out
#SBATCH --error=/home/energy/bepab/logs/marksdft_%A_%a.err

# DFT refinement of one model transition state per node (D32), ORCA 6.0.1:
#   1 uks_sp    UKS + STABPerform at the model structure
#   2 optts     OptTS from those orbitals, starting Hessian from the model (model.hess)
#   3 stab      UKS + STABPerform at the DFT saddle
#   4 freq      NumFreq on the stable solution
#   5 irc       IRC from the DFT saddle with the frequency Hessian
# Finished steps are skipped on restart. Scratch is removed at the end.
#
#   N=$(wc -l < ~/marks_pilot/tasks.txt); sbatch --array=0-$(( N - 1 )) pipeline/job_marks_dft.sh

source /etc/profile
source $HOME/opt/orca6.env
ORCA=$ORCADIR/orca
METHOD='wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3'
HEAD='%pal nprocs 24 end
%maxcore 3500'
SCF='  Thresh 1e-12
  TCut   1e-13
  MaxIter 300'
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $HOME/marks_pilot/tasks.txt)
RXN=$(echo "$LINE" | cut -d: -f1)
W=$HOME/marks_pilot/$RXN
cd $W || exit 2
ok () { grep -q "ORCA TERMINATED NORMALLY" $1.out 2>/dev/null; }
echo "$RXN start $(date)"

ok uks_sp || { printf '! UKS %s\n%s\n%%scf\n%s\n  STABPerform true\n  STABRestartUHFifUnstable true\nend\n* xyzfile 0 1 ts_model.xyz\n' "$METHOD" "$HEAD" "$SCF" > uks_sp.inp; $ORCA uks_sp.inp > uks_sp.out 2>&1; }
ok uks_sp || { echo "ABBRUCH uks_sp"; exit 7; }
echo "uks_sp done $(date)"

ok optts || { cp uks_sp.gbw optts_start.gbw
  printf '! UKS %s OptTS MORead\n%%moinp "optts_start.gbw"\n%s\n%%scf\n%s\nend\n%%geom\n  InHess Read\n  InHessName "model.hess"\n  MaxIter 150\nend\n* xyzfile 0 1 ts_model.xyz\n' "$METHOD" "$HEAD" "$SCF" > optts.inp
  $ORCA optts.inp > optts.out 2>&1; }
ok optts && [ -f optts.xyz ] || { echo "ABBRUCH optts"; grep -i -m3 "error\|hess" optts.out; exit 8; }
echo "optts done $(date)"

ok stab || { cp optts.gbw stab_start.gbw
  printf '! UKS %s MORead\n%%moinp "stab_start.gbw"\n%s\n%%scf\n%s\n  STABPerform true\n  STABRestartUHFifUnstable true\nend\n* xyzfile 0 1 optts.xyz\n' "$METHOD" "$HEAD" "$SCF" > stab.inp
  $ORCA stab.inp > stab.out 2>&1; }
ok stab || { echo "ABBRUCH stab"; exit 9; }

ok freq || { cp stab.gbw freq_start.gbw
  printf '! UKS %s NumFreq MORead\n%%moinp "freq_start.gbw"\n%s\n%%scf\n%s\nend\n* xyzfile 0 1 optts.xyz\n' "$METHOD" "$HEAD" "$SCF" > freq.inp
  $ORCA freq.inp > freq.out 2>&1; }
ok freq || { echo "ABBRUCH freq"; exit 10; }
echo "freq done $(date)"

ok irc || { cp stab.gbw irc_start.gbw
  printf '! UKS %s IRC MORead\n%%moinp "irc_start.gbw"\n%s\n%%scf\n%s\nend\n%%irc\n  InitHess Read\n  Hess_Filename "freq.hess"\n  MaxIter 80\nend\n* xyzfile 0 1 optts.xyz\n' "$METHOD" "$HEAD" "$SCF" > irc.inp
  $ORCA irc.inp > irc.out 2>&1; }
ok irc || echo "irc did not terminate normally"
find . -maxdepth 1 -type f \( -name '*.tmp' -o -name '*.densities*' -o -name '*_start.gbw' -o -name '*.ges' -o -name '*.bibtex' -o -name '*_property.txt' \) -delete
echo "$RXN end $(date)"
