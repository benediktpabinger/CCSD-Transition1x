#!/bin/bash
#SBATCH --job-name=verify_bs
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=/home/energy/s242862/verify_bs/slurm_%j.out
#SBATCH --error=/home/energy/s242862/verify_bs/slurm_%j.err

# Reproduce the single point that the NEB performed at its final TS geometry,
# using the IDENTICAL ORCA input from pipeline/orca_neb_omol25.py, and record
# the SCF formalism and <S**2>. The NEB's own ORCA outputs were deleted with
# the scratch directory, so they have to be regenerated.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
W=$H/verify_bs
mkdir -p $W; cd $W
ORCA=$(which orca)

: > $W/results.txt

for d in $H/orca_neb_omol25/*/; do
  r=$(basename "$d")
  [ -f "$d/converged" ] || continue
  [ -f "$d/transition_state.xyz" ] || continue

  cp "$d/transition_state.xyz" $W/$r.xyz
  # exactly the input the NEB used (SIMPLEINPUT + blocks + charge 0 mult 1)
  cat > $W/$r.inp <<EOF
! wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3 EnGrad
%pal nprocs 12 end
%maxcore 3000
%scf
  maxiter 200
  Thresh 1e-12
  TCut   1e-13
end
* xyzfile 0 1 $r.xyz
EOF

  $ORCA $W/$r.inp > $W/$r.out 2>/dev/null

  E=$(grep 'FINAL SINGLE POINT ENERGY' $W/$r.out | tail -1 | awk '{print $NF}')
  S2=$(grep 'Expectation value of <S\*\*2>' $W/$r.out | tail -1 | awk '{print $NF}')
  # which formalism did ORCA actually run?
  FORM=$(grep -m1 -E "Hartree-Fock type +HFTyp" $W/$r.out | awk '{print $NF}')
  # any spin-symmetry-broken guess logged?
  BRK=$(grep -cE "Rotate|BrokenSym|FlipSpin|spin.*symmetry.*brok|UHF STABILITY" $W/$r.out)

  echo "$r E=${E:-NA} S2=${S2:-NOTPRINTED} HFTyp=${FORM:-NA} breakmsg=$BRK" >> $W/results.txt
  echo "$r  E=${E:-NA}  S2=${S2:-NOTPRINTED}  HFTyp=${FORM:-NA}"
  rm -f $W/$r.gbw $W/$r.densities $W/$r*.tmp 2>/dev/null
done

echo ""
echo "=== ZUSAMMENFASSUNG ==="
cat $W/results.txt
echo ""
echo "HFTyp-Verteilung:"; grep -oE 'HFTyp=[A-Za-z]+' $W/results.txt | sort | uniq -c
echo "S2 gedruckt?:";     grep -oE 'S2=[A-Za-z0-9.-]+' $W/results.txt | sort | uniq -c | head
