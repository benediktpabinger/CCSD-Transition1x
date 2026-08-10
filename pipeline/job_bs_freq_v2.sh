#!/bin/bash
#SBATCH --job-name=bsfreq2
#SBATCH --partition=xeon24el8
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/home/energy/s242862/bs_freq_v2/slurm_%A_%a.out
#SBATCH --error=/home/energy/s242862/bs_freq_v2/slurm_%A_%a.err

# Numerical BS-UKS frequencies at the v2 TS geometries.
#
# Two of these converged with strong broken symmetry:
#   rxn8832  <S^2> 0.870 -> 1.001
#   rxn7957  <S^2> 0.513 -> 0.709
#
# Three were flagged BS_LOST by the S2_MIN = 0.3 gate but converged
# geometrically, and the external instability is still present at the final
# geometry (rxn3107: lambda_min_ext = -0.0052 at step 78, down from -0.0125 at
# the start). The question is whether these are genuine weakly diradical saddle
# points that the fixed 0.3 threshold misclassified as failures:
#   rxn3107  <S^2> 0.409 -> 0.179
#   rxn8885  <S^2> 0.507 -> 0.153
#   rxn7060  <S^2> 0.374 -> 0.047   (weakest; BS and RKS nearly degenerate here)

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
module load ASE

export OMP_NUM_THREADS=12
export PYSCF_MAX_MEMORY=50000
export BSFREQ_SRC=bs_tsopt_v2
export BSFREQ_OUT=bs_freq_v2

RXNS=(rxn8832 rxn7957 rxn3107 rxn8885 rxn7060)
RXN=${RXNS[$SLURM_ARRAY_TASK_ID]}

mkdir -p /home/energy/s242862/bs_freq_v2
echo "Task $SLURM_ARRAY_TASK_ID: $RXN  node $SLURM_NODELIST  $(date)"
python3 /home/energy/s242862/bs_freq2.py $RXN
RC=$?
echo "rc=$RC $(date)"
exit $RC
