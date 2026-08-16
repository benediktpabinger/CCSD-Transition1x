#!/bin/bash
#SBATCH --job-name=tsbroken
#SBATCH --partition=xeon24el8
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/s242862/tsopt_broken/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/tsopt_broken/slurm_%A_%a.err

# Start the saddle search from the highest image that is actually on the
# broken sheet, instead of from the one the NEB called the top.
#
# The band measurement showed that what decides the outcome is the spin state
# of the highest image: where it is broken the NEB result sits 0.011 eV/A from
# stationary, where it is restricted it sits on the RKS-TS and 1.074 eV/A off.
# Four reactions found the broken sheet somewhere but not at the top:
#
#   rxn1283  broken 3-6, top 7     rxn8885  broken 6-7, top 5
#   rxn8827  broken 5-6, top 7     rxn5691  broken 4,   top 5
#
# For these the climbing image is one step beside the region that carries the
# chemistry.  This bypasses that choice: take the highest image with
# <S^2> > 0.3 and optimise from there.
#
# The index is read from the measurement rather than hard-coded, so the job
# cannot silently drift away from the data it is based on.
#
# Falsifiable both ways.  If the four converge onto saddles that other methods
# also found, the climbing-image choice was the whole problem.  If they do not,
# the cause lies deeper and "resolve the broken region better" is the wrong
# prescription.
#
# LEVEL: wB97X/6-31G(d), the cheap testbed, for turnaround.  The starting
# geometries come from the production bands, so this mixes levels -- fine for
# a starting point, and the comparison to our production structures is
# therefore indirect.  Note also that the cheap level breaks more deeply
# (rxn8827: -167 meV against -27.5), which makes a broken-symmetry
# optimisation easier here than it would be at the production level.  A
# success here is a reason to repeat it above, not a result about above.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn1283 rxn8827 rxn8885 rxn5691)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

W=$H/tsopt_broken/$RXN
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"

MEAS=$H/band_s2_v2/$RXN/band_s2.txt
if [ ! -s "$MEAS" ]; then
  echo "ABBRUCH: keine Bandmessung fuer $RXN"
  exit 2
fi

# highest image index with <S^2> > 0.3
K=$(awk '$3 != "nan" && $3+0 > 0.3 {k=$2} END {print (k=="" ? -1 : k)}' $MEAS)
S2=$(awk -v k="$K" '$2+0 == k {print $3}' $MEAS)
TOP=$(awk '$4 != "nan" {if (e=="" || $4+0 > e) {e=$4+0; t=$2}} END {print t}' $MEAS)
echo "hoechstes gebrochenes Bild: $K  (<S^2> = $S2)   Gipfel des Bandes: $TOP"

if [ "$K" -lt 0 ]; then
  echo "ABBRUCH: kein Bild dieses Bandes ist gebrochen -- nichts zu starten"
  exit 3
fi
if [ "$K" = "$TOP" ]; then
  echo "HINWEIS: hoechstes gebrochenes Bild IST der Gipfel; dieser Lauf"
  echo "         wiederholt dann nur, was der NEB ohnehin getan hat."
fi

# cut image K out of the trajectory
TRJ=$(ls $H/bs_uks_neb_results/$RXN/*_MEP_trj.xyz | head -1)
NAT=$(head -1 $TRJ | tr -d ' \r')
awk -v nat="$NAT" -v k="$K" 'BEGIN{i=-1}
    {if ((NR-1) % (nat+2) == 0) i++
     if (i == k) print}' $TRJ > start.xyz
echo "Startgeometrie: $(head -1 start.xyz) Atome aus Bild $K"

# ------------------------------------------------------------------ 1a
# The broken solution at THIS geometry.  STABPerform cannot run next to an
# optimisation, so it gets its own single point and the orbitals are handed on.
cat > bs.inp <<EOF
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

* xyzfile 0 1 $W/start.xyz
EOF

$ORCA bs.inp > bs.out 2> bs.err
BS2=$(grep 'Expectation value of <S\*\*2>' bs.out | awk '{print $NF}' | tail -1)
echo "1a: <S^2> am Startpunkt, billiges Niveau = $BS2"

if awk -v s="$BS2" 'BEGIN{exit !(s < 0.3)}'; then
  echo "HINWEIS: am billigen Niveau bricht dieser Punkt nicht (<S^2> = $BS2),"
  echo "         obwohl er es am Produktionsniveau tut. Lauf geht weiter,"
  echo "         das Ergebnis ist dann aber eine RKS-Optimierung."
fi

# ------------------------------------------------------------------ 2
# TS optimisation from those orbitals.  wB97X has no VV10, so the Hessian is
# analytic and Freq can run in the same job.
cat > tsopt.inp <<EOF
! UKS wB97X 6-31G(d) OptTS Freq TightSCF SlowConv MORead

%moinp "$W/bs.gbw"

%pal
  nprocs 8
end

%maxcore 3500

%geom
  Calc_Hess true
  MaxIter 200
end

%scf
  MaxIter 500
end

* xyzfile 0 1 $W/start.xyz
EOF

$ORCA tsopt.inp > tsopt.out 2> tsopt.err
echo "2: rc=$?"

echo ""
echo "--- Ergebnis ---"
grep -E "HURRAY|THE OPTIMIZATION HAS CONVERGED|ORCA TERMINATED NORMALLY" \
     tsopt.out | tail -3
echo "  <S^2> am Ende:  $(grep 'Expectation value of <S\*\*2>' tsopt.out \
    | awk '{print $NF}' | tail -1)"
echo "  imaginaere Frequenzen:"
grep -A 200 'VIBRATIONAL FREQUENCIES' tsopt.out | tail -n +5 \
  | awk '/cm\*\*-1/ && $2+0 < -1 {printf "    %s cm-1\n", $2}' | head -5
echo "  Endenergie: $(grep 'FINAL SINGLE POINT ENERGY' tsopt.out \
    | awk '{print $5}' | tail -1)"

echo ""
echo "--- wie weit vom Startbild und vom RKS-TS ---"
ls *.xyz | head
echo "Finished $(date)"
