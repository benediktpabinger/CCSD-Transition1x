#!/bin/bash
#SBATCH --job-name=irc2
#SBATCH --partition=xeon24el8
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_irc2/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_irc2/slurm_%A_%a.err

# A real IRC on the two contested reactions, launched from both rival saddles.
#
# rxn1147 and rxn7957 are the two cases the three-stage rule reversed, and both
# were decided by a judgement call: "this reactive bond is already at its normal
# length, so the reaction is finished at that point". That judgement is the last
# unautomated step in the whole argument. An IRC settles it without judgement --
# if a structure lies past the transition state, the path from the other saddle
# runs straight through it on the way down.
#
# Each reaction is launched from our broken-symmetry saddle and from the model
# saddle that beat it on energy. All four Hessians already exist.
#
#   0  rxn1147  ours    our saddle, C1-O5 at 1.864 A
#   1  rxn1147  UMA-S   model saddle, C1-O5 at 1.497 A, 234 meV lower
#   2  rxn7957  ours    our saddle, C5-H7 at 1.120 A
#   3  rxn7957  UMA-M   model saddle, C5-H7 at 1.190 A, 890 meV lower
#
# The deliverable is the bond trace, not the endpoint: path_forward.json and
# path_backward.json carry both reactive bond lengths at every step.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000

RXNS=(rxn1147 rxn1147 rxn7957 rxn7957)
SRCS=(ours    UMA-S   ours    UMA-M)

export IRC_RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}
export IRC_SRC=${SRCS[$SLURM_ARRAY_TASK_ID]}
export IRC_STEP=0.15
export IRC_MAX=60
export IRC_OUT=/home/energy/s242862/bs_irc2

mkdir -p $IRC_OUT
echo "Task $SLURM_ARRAY_TASK_ID: $IRC_RXN $IRC_SRC  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_irc2.py
RC=$?
echo "rc=$RC $(date)"
exit $RC
