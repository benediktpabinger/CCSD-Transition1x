#!/bin/bash
#SBATCH --job-name=freqsweep
#SBATCH --partition=xeon24el8
#SBATCH --nodes=1
# ORCA parallelises over MPI processes, not threads -> ntasks, not cpus-per-task
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/orca_freq/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/orca_freq/slurm_%A_%a.err

# A frequency at every structure that has one to give.
#
# The reason to stop selecting: the 15 model frequencies that exist were chosen
# on two nights by two different criteria, and the reactions left out include
# rxn1320 and rxn4518, where our own structure is known to be wrong and the
# model is therefore the only remaining candidate. A gap produced that way is
# indistinguishable in the tables from a test that was run and failed.
#
# The task list is generated from the data by make_freq_list.py, not written
# by hand, so the selection cannot drift again.
#
# Three stages per structure, the chain validated earlier today against PySCF
# (mode overlap >= 0.9994 on four structures):
#   1a  broken-symmetry single point with STABPerform. Has to stand alone --
#       ORCA allows no RunTyp but SinglePoint beside a stability analysis.
#   1b  gradient, reading those orbitals, so the surface is not re-derived
#   2   NumFreq, same orbitals. VV10 has no analytic second derivatives in ORCA
#       5.0.4 either, so numerical is not a choice.
#
# Structures whose gradient is large are included on purpose. n_imag there says
# nothing about transition states, but the Hessian is exactly what OptTS needs
# as a starting point, so the work carries over to the next step instead of
# being repeated.
#
# Throttled to 16 concurrent: the account allows 20 and four IRC jobs are running.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
LIST=${FREQ_LIST:-$H/freq_tasks.txt}

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
LABEL=$(echo "$LINE" | awk '{print $1}')
GEOM=$(echo "$LINE" | awk '{print $2}')

if [ -z "$LABEL" ] || [ ! -f "$GEOM" ]; then
  echo "bad task $SLURM_ARRAY_TASK_ID: '$LINE'"; exit 1
fi

W=$H/orca_freq/$LABEL
mkdir -p $W
cd $W

ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $LABEL  node $SLURM_NODELIST  $(date)"
echo "geometry: $GEOM"
cp $GEOM start.xyz
head -1 start.xyz

cat > bs_sp.inp <<'EOF'
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF
%pal nprocs 12 end
%maxcore 3000
%scf
  STABPerform true
  STABRestartUHFifUnstable true
  MaxIter 300
end
* xyzfile 0 1 start.xyz
EOF

echo ""
echo "=== 1a  BS single point ==="
$ORCA bs_sp.inp > bs_sp.out 2> bs_sp.err
grep -E 'is unstable|is stable|FINAL SINGLE POINT ENERGY|Expectation value of <S\*\*2>|finished by error' bs_sp.out | tail -6
[ -f bs_sp.gbw ] || { echo "NO GBW - stopping"; exit 1; }
cp bs_sp.gbw bs_start.gbw

cat > engrad.inp <<'EOF'
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad MORead
%moinp "bs_start.gbw"
%pal nprocs 12 end
%maxcore 3000
%scf
  MaxIter 300
end
* xyzfile 0 1 start.xyz
EOF

echo ""
echo "=== 1b  gradient ==="
$ORCA engrad.inp > engrad.out 2> engrad.err
grep 'Expectation value of <S\*\*2>' engrad.out | tail -1
awk '/CARTESIAN GRADIENT/{f=1;next} f&&NF>=6{
      for(i=4;i<=6;i++){v=$i<0?-$i:$i; if(v>m)m=v}}
     f&&/Difference to translation/{printf "  max |dE/dx| = %.6f Eh/Bohr = %.4f eV/A\n", m, m*51.42208; exit}' \
    engrad.out

cat > numfreq.inp <<'EOF'
! UKS wB97M-V def2-TZVP def2/J RIJCOSX TightSCF NumFreq MORead
%moinp "bs_start.gbw"
%pal nprocs 12 end
%maxcore 3000
%scf
  MaxIter 300
end
%freq
  CentralDiff true
  Increment 0.005
end
* xyzfile 0 1 start.xyz
EOF

echo ""
echo "=== 2  NumFreq ==="
$ORCA numfreq.inp > numfreq.out 2> numfreq.err
echo ""
awk '/VIBRATIONAL FREQUENCIES/{f=1} f{print} /NORMAL MODES/{if(f)exit}' \
    numfreq.out | head -22
echo ""
grep 'Expectation value of <S\*\*2>' numfreq.lastscf 2>/dev/null | tail -1
grep -E 'ORCA TERMINATED NORMALLY|finished by error' numfreq.out | tail -1
ls -la numfreq.hess 2>/dev/null
echo "Finished $(date)"
