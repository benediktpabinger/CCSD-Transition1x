#!/bin/bash
#SBATCH --job-name=sp_same
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --output=/home/energy/s242862/sp_samelevel/slurm_%A_%a.out

# Same-level single points.  The point of this job is that the energies of the
# new NEB-CI saddle and of every previously known structure for the SAME
# reaction are computed with ONE method line, so they can be subtracted.
# Energies across levels of theory cannot; that is why nothing here reads a
# stored production energy.
#
# Two solutions are sought at every geometry: a plain UKS run that follows any
# instability downhill, and a broken-symmetry guess.  The lower of the two is
# the answer.  A geometry sitting on the closed-shell sheet will simply return
# the same number twice -- that is information, not a failure.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }
echo "ORCA: $ORCA"
read -r -a RXNS <<< "$RXN_LIST"
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
METHOD="wB97X 6-31G(d)"

W=$H/sp_samelevel/$RXN
mkdir -p $W && cd $W
echo "Reaktion: $RXN   Niveau: $METHOD"

first() { for p in "$@"; do for g in $p; do [ -f "$g" ] && { echo "$g"; return; }; done; done; }

declare -a LAB SRC
add() { s=$(first "$2"); [ -n "$s" ] && { LAB+=("$1"); SRC+=("$s"); }; }
add neu     "$H/bs_uks_nebci/$RXN/tsopt.xyz"
add unsere  "$H/orca_freq/ours_$RXN/start.xyz $H/orca_irc/${RXN}_ours/start.xyz"
add RKS-TS  "$H/orca_neb_results/$RXN/transition_state.xyz"
add UKS-NEB "$H/bs_uks_neb_results/$RXN/*NEB-TS_converged.xyz $H/bs_uks_neb_results/$RXN/*NEB-CI_converged.xyz"
add UMA-M   "$H/uma_m_neb_results/$RXN/transition_state.xyz"
add TSoptM  "$H/orca_freq/tsopt_${RXN}_UMA-M/start.xyz"
echo "Strukturen: ${#LAB[@]}   (${LAB[@]})"

# the atom count must agree across all of them, otherwise the subtraction is
# meaningless -- different molecules, not different saddles
N0=$(head -1 "${SRC[0]}" | tr -d ' \r')
for i in "${!SRC[@]}"; do
  n=$(head -1 "${SRC[$i]}" | tr -d ' \r')
  [ "$n" = "$N0" ] || { echo "ABBRUCH: ${LAB[$i]} hat $n Atome, erwartet $N0"; exit 3; }
done
echo "Atomzahl einheitlich: $N0"

runsp() {  # $1 tag  $2 xyz  $3 extra-simple-input  $4 extra-block
  d=$W/$1; mkdir -p $d; cp "$2" $d/geo.xyz
  cat > $d/sp.inp <<EOF
! UKS $METHOD SP TightSCF SlowConv $3
%pal
  nprocs 4
end
%maxcore 3500
$4
* xyzfile 0 1 geo.xyz
EOF
  (cd $d && $ORCA sp.inp > sp.out 2>&1)
  grep -q "ORCA TERMINATED NORMALLY" $d/sp.out || { echo "ABBRUCH: $1 nicht normal beendet"; tail -3 $d/sp.out; exit 6; }
  e=$(grep "FINAL SINGLE POINT ENERGY" $d/sp.out | tail -1 | awk '{print $NF}')
  case "$e" in ''|0.000000000000) echo "ABBRUCH: $1 ohne Energie ($e)"; exit 7;; esac
  echo "     $1  E=$e"
}

for i in "${!LAB[@]}"; do
  l=${LAB[$i]}; s=${SRC[$i]}
  echo "  -> $l  $s"
  runsp "${l}__plain" "$s" "" "%scf
  STABPerform true
  STABRestartUHFifUnstable true
end"
  runsp "${l}__bs" "$s" "" "%scf
  BrokenSym 1,1
end"
done

echo "FERTIG $RXN"
