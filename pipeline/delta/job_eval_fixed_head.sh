#!/bin/bash
#SBATCH --job-name=eval_fixed_head
#SBATCH --partition=h200
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/energy/s242862/logs/eval_fixed_head_%j.log

# Evaluate the fixed delta head (corrected irreps) with the same protocols as v2,
# plus the rotation-invariance test old vs. fixed.
#   1. rotation invariance            -> ~/delta_head/rotation_invariance.json
#   2. 30-reaction fixed-geometry SP  -> ~/delta_head/eval_benchmark_sp_fixed_full.json
#   3. RKS-stable subset + OMol25     -> ~/delta_head/eval_sp_rks_stable_fixed.json

module load Python/3.13.5-GCCcore-14.3.0

P=/home/energy/s242862/pipeline/delta

echo "=== 1/3 rotation invariance ==="
python3 -u ${P}/rot_invariance_head.py

echo "=== 2/3 benchmark SP (fixed head) ==="
python3 -u ${P}/eval_benchmark_sp_fixed.py

echo "=== 3/3 RKS-stable SP (fixed head) ==="
python3 -u ${P}/eval_sp_rks_stable_fixed.py

echo "All done."
