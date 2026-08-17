#!/bin/bash
#SBATCH --job-name=cfreq
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --output=/home/energy/s242862/freq_central/slurm_%A_%a.out

# The frequency step embedded in OptTS came out with FORWARD differences
# (3N displacements against one reference).  Every standalone NumFreq in this
# project used CENTRAL differences (6N, errors cancel between +h and -h), so
# this job repeats the frequency alone with the recipe that produced those
# 108 Hessians -- copied verbatim from orca_freq/nebts_rxn0346/numfreq.inp
# rather than reconstructed from the manual.
#
# The geometry is not re-optimised.  Same structure, same orbitals, only the
# difference scheme changes, so the comparison is clean.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a
ORCA=$(which orca)
[ -x "$ORCA" ] || { echo "ABBRUCH: orca nicht ausfuehrbar ($ORCA)"; exit 5; }

H=/home/energy/s242862
read -r -a RXNS <<< "$RXN_LIST"
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
SRC=$H/bs_uks_nebci_prod/$RXN
W=$H/freq_central/$RXN
mkdir -p $W && cd $W
echo "Reaktion: $RXN"

for f in tsopt2.xyz tsopt2.gbw tsopt2.out; do
  [ -f $SRC/$f ] || { echo "ABBRUCH: $SRC/$f fehlt"; exit 3; }
done
cp $SRC/tsopt2.xyz start.xyz
cp $SRC/tsopt2.gbw  start.gbw

cat > numfreq.inp <<EOF
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF NumFreq MORead
%moinp "start.gbw"
%pal nprocs 12 end
%maxcore 4500
%scf
  MaxIter 300
end
%freq
  CentralDiff true
  Increment 0.005
end
* xyzfile 0 1 start.xyz
EOF

$ORCA numfreq.inp > numfreq.out 2>&1

grep -q "ORCA TERMINATED NORMALLY" numfreq.out || { echo "ABBRUCH: nicht normal beendet"; tail -5 numfreq.out; exit 6; }
grep -q "Central differences            ... used" numfreq.out || { echo "ABBRUCH: doch keine zentralen Differenzen"; grep -m1 "Central differences" numfreq.out; exit 7; }
grep -q "VIBRATIONAL FREQUENCIES" numfreq.out || { echo "ABBRUCH: keine Frequenzen"; exit 8; }

echo "--- Ergebnis ---"
grep -m1 "Number of displacements" numfreq.out
echo "  vorwaerts (alt):"
awk '/VIBRATIONAL FREQUENCIES/{f=NR} {a[NR]=$0} END{for(i=f;i<=f+30;i++) print a[i]}' $SRC/tsopt2.out | grep "cm\*\*-1" | awk '$2+0 < -1 {print "     " $0}'
echo "  zentral (neu):"
awk '/VIBRATIONAL FREQUENCIES/{f=NR} {a[NR]=$0} END{for(i=f;i<=f+30;i++) print a[i]}' numfreq.out | grep "cm\*\*-1" | awk '$2+0 < -1 {print "     " $0}'
echo "FERTIG $RXN"
