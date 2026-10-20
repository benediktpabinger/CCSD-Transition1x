#!/bin/bash
#SBATCH --job-name=eval_fixed_nod3_cpu
#SBATCH --partition=xeon24el9
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH --output=/home/energy/s242862/logs/eval_fixed_nod3_cpu_%j.log
# Same as job_eval_fixed_head.sh, but the cheap-level single points come from
# mr_benchmark/orca_engrad_nod3 (wB97X/6-31G(d) without D3). Outputs get the
# suffix _nod3; the original JSONs are not touched.
module load Python/3.13.5-GCCcore-14.3.0
P=/home/energy/s242862/pipeline/delta
echo "=== benchmark SP (fixed head, no D3) ==="
python3 -u ${P}/eval_benchmark_sp_fixed_nod3.py
echo "=== RKS-stable SP (fixed head, no D3) ==="
python3 -u ${P}/eval_sp_rks_stable_fixed_nod3.py
echo "All done."
