#!/bin/bash
#SBATCH --job-name=tsumam
#SBATCH --partition=xeon24el8
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/home/energy/s242862/bs_tsopt_umam/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_tsopt_umam/slurm_%A_%a.err

# The from-model saddle searches that were never run.
#
# Ten of the nineteen multireference reactions have a transition-state
# optimisation started from a model geometry; nine do not, and rxn0894 has one
# only from UMA-S.  That gap is listed under the chapter's own limitations,
# and it matters because every other optimisation in this project started from
# the RKS reference -- the point known not to be on the ground-state surface.
# A search that only ever starts there cannot find a saddle that lies
# elsewhere, and the limitation section says so.
#
# UMA-M is forced through TSOPT_MODEL rather than left to the driver's own
# choice.  bs_tsopt_retry.py picks the model whose symmetry breaking is
# deepest, and that criterion sent the July sweep to the floor of the
# diradical valley instead of to a saddle: the most strongly broken point is
# not the one nearest to stationarity.  Fixing the model also makes these
# comparable to the nine that exist, eight of which started from UMA-M.
#
# THE PREDICTION, WRITTEN DOWN BEFORE THE RUN.  The triage criterion says a
# refinement succeeds when the DFT gradient at the model geometry is below
# about 0.25 eV/A, and fails above.  Current record: 6 of 7 below, 0 of 3
# above.  The gradients here are
#
#   rxn1320  0.044     rxn5690  0.112     rxn4113  0.185
#   rxn4518  0.055     rxn1283  0.125     rxn8885  0.190
#   rxn4522  0.083     rxn6196  0.138
#   rxn5691  0.085
#
# all below the threshold, so the criterion predicts that all nine reach a
# valid saddle.  rxn0894 is the negative control: its UMA-M geometry carries
# about 1.32 eV/A, far above, and the UMA-S attempt from 0.776 ended on a
# minimum.  A failure there is expected; a failure among the other nine
# falsifies the criterion.
#
# This can also cost us.  Several of the nine currently have no rival to our
# structure, and a saddle found here can take that away -- which is the reason
# to run it.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000
export TSOPT_OUT=/home/energy/s242862/bs_tsopt_umam
export TSOPT_MODEL=UMA-M

# ordered by predicted gradient, cheapest bet first
RXNS=(rxn1320 rxn4518 rxn4522 rxn5691 rxn5690 rxn1283 rxn6196 rxn4113 rxn8885 rxn0894)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p $TSOPT_OUT
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  from UMA-M  node $SLURM_NODELIST  $(date)"

G=/home/energy/s242862/uma_m_neb_results/$RXN/transition_state.xyz
if [ ! -f "$G" ]; then
  echo "ABBRUCH: keine UMA-M-Geometrie fuer $RXN"
  exit 2
fi
echo "Start: $G"

python3 /home/energy/s242862/bs_tsopt_retry.py $RXN frommodel
RC=$?
echo "rc=$RC $(date)"

echo ""
echo "--- Ergebnis ---"
R=$TSOPT_OUT/$RXN/result.json
if [ -f "$R" ]; then
  python3 - "$R" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ('status', 'start', 'start_model', 'e_uks_final', 's2_final',
          'grad_final', 'rmsd_to_start', 'steps'):
    if k in d:
        print(f'  {k:<16} {d[k]}')
EOF
else
  echo "  keine result.json"
fi
exit $RC
