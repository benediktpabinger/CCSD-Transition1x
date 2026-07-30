#!/bin/bash
#SBATCH --job-name=omol26
#SBATCH --partition=xeon24el8
#SBATCH --array=0-25
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home/energy/s242862/omol25_settings/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/omol25_settings/slurm_%A_%a.err

# Reproduce the OMol25 protocol (arXiv 2505.08762, Sec. 2.7 + App. A) on the
# top-26 N_FOD reactions, at the ORCA NEB TS geometries:
#   wB97M-V / def2-TZVPD, RIJ+COSX, TightSCF, DEFGRID3, thresh 1e-12, tcut 1e-13
#   spin symmetry broken by a 20 deg HOMO-LUMO rotation in the beta space
# ORCA 6.0.0 is not installed here; 5.0.4 is used with the thresholds set by hand.
#
# Three calculations per reaction:
#   A  RKS                      -> closed-shell reference
#   B  UKS + 20 deg beta rotation -> the OMol25 protocol
#   C  UKS + stability restart    -> our protocol, SAME settings
# B vs C isolates the protocol; C vs our earlier def2-TZVP runs isolates settings.

source /etc/profile
module load gompi/2023a
module load ORCA/5.0.4-gompi-2023a

H=/home/energy/s242862
RXNS=(rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn7945 rxn7937 rxn6196 rxn0346 rxn1150 rxn0896 rxn4518 rxn3107 rxn8837 rxn7060 rxn5691 rxn1283 rxn8827 rxn4522 rxn7936 rxn1147 rxn0894 rxn0101 rxn10005 rxn10054 rxn7957)
HOMOS=(24 24 22 22 24 24 24 24 22 22 22 22 22 24 24 24 22 24 22 24 22 22 22 25 24 24)
LUMOS=(25 25 23 23 25 25 25 25 23 23 23 23 23 25 25 25 23 25 23 25 23 23 23 26 25 25)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
HOMO=${HOMOS[$SLURM_ARRAY_TASK_ID]}
LUMO=${LUMOS[$SLURM_ARRAY_TASK_ID]}

W=$H/omol25_settings/$RXN
mkdir -p $W; cd $W
ORCA=$(which orca)
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  HOMO=$HOMO LUMO=$LUMO  node $SLURM_NODELIST  $(date)"

cp $H/orca_neb_results/$RXN/transition_state.xyz start.xyz

HEAD='%pal nprocs 12 end
%maxcore 3000'
SCFCOMMON='  Thresh 1e-12
  TCut   1e-13
  MaxIter 300'

cat > A_rks.inp <<EOF
! RKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
$HEAD
%scf
$SCFCOMMON
end
* xyzfile 0 1 start.xyz
EOF

cat > B_rot20.inp <<EOF
! UKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
$HEAD
%scf
$SCFCOMMON
  Rotate {$HOMO, $LUMO, 20, 1, 1} end
end
* xyzfile 0 1 start.xyz
EOF

cat > C_stab.inp <<EOF
! UKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
$HEAD
%scf
$SCFCOMMON
  STABPerform true
  STABRestartUHFifUnstable true
end
* xyzfile 0 1 start.xyz
EOF

for v in A_rks B_rot20 C_stab; do
  echo ""
  echo "======== $v ========"
  $ORCA $v.inp > $v.out 2> $v.err
  echo "rc=$?"
  grep -E 'ORCA finished by error|aborting' $v.out | head -3
  E=$(grep 'FINAL SINGLE POINT ENERGY' $v.out | tail -1 | awk '{print $NF}')
  S2=$(grep 'Expectation value of <S\*\*2>' $v.out | tail -1 | awk '{print $NF}')
  echo "E=$E  S2=$S2"
  echo "$v E=$E S2=$S2" >> summary.txt
done

echo "Finished $(date)"
