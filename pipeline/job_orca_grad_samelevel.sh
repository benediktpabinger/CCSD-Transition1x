#!/bin/bash
#SBATCH --job-name=sp_grad
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --output=/home/energy/s242862/sp_grad/slurm_%A_%a.out

# An energy taken at a geometry that is not stationary at THIS level says
# nothing about saddle heights -- the point simply sits on a slope.  This job
# measures max|F| at every geometry the energy table compares, so each row is
# either usable or discardable.
#
# The orbitals come from the matching single point via MORead, so the gradient
# belongs to exactly the SCF solution whose energy was tabulated.  Stability
# analysis is deliberately absent: ORCA allows it only for RunTyp SinglePoint,
# and re-running it here would risk landing on a different solution anyway.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }

H=/home/energy/s242862
read -r -a RXNS <<< "$RXN_LIST"
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
METHOD="wB97X 6-31G(d)"
S=$H/sp_samelevel/$RXN
W=$H/sp_grad/$RXN
mkdir -p $W && cd $W
echo "Reaktion: $RXN   Niveau: $METHOD"

n_ok=0
for src in $S/*__*/; do
  tag=$(basename $src)
  [ -f "$src/sp.gbw" ] || { echo "  uebersprungen $tag (keine Orbitale)"; continue; }
  grep -q "ORCA TERMINATED NORMALLY" $src/sp.out || { echo "  uebersprungen $tag (SP unvollstaendig)"; continue; }
  d=$W/$tag; mkdir -p $d
  cp $src/geo.xyz $d/geo.xyz
  cp $src/sp.gbw  $d/start.gbw
  cat > $d/g.inp <<EOF
! UKS $METHOD EnGrad TightSCF SlowConv MORead
%moinp "start.gbw"
%pal
  nprocs 4
end
%maxcore 3500
* xyzfile 0 1 geo.xyz
EOF
  (cd $d && $ORCA g.inp > g.out 2>&1)
  grep -q "ORCA TERMINATED NORMALLY" $d/g.out || { echo "ABBRUCH: $tag nicht normal beendet"; tail -4 $d/g.out; exit 6; }
  [ -f $d/g.engrad ] || { echo "ABBRUCH: $tag ohne Gradientendatei"; exit 7; }
  n_ok=$((n_ok+1))
  echo "  ok $tag"
done

[ $n_ok -gt 0 ] || { echo "ABBRUCH: kein einziger Gradient"; exit 8; }
echo "FERTIG $RXN   $n_ok Gradienten"
