#!/bin/bash
#SBATCH --job-name=fod_screen
#SBATCH --partition=xeon24el8
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120GB
#SBATCH --array=0-11
#SBATCH --output=/home/energy/s242862/logs/fod_screen_%A_%a.log

# FOD screening: 12 nodes, each processes ~24 reactions in parallel.
# 6 parallel PySCF workers per node, each using 4 OMP threads = 24 CPUs used.
#
# Generate reaction list first (done once):
#   python3 -c "
#     import os; test=set(l.strip() for l in open('/home/energy/s242862/ccsd_dataset/test_reactions.txt'))
#     neb='/home/energy/s242862/orca_neb_results'
#     rxns=sorted(r for r in os.listdir(neb) if r in test
#                 and os.path.exists(f'{neb}/{r}/converged')
#                 and os.path.exists(f'{neb}/{r}/transition_state.xyz'))
#     open('/home/energy/s242862/fod_rxn_list.txt','w').write('\n'.join(rxns))
#   "
# Then: sbatch job_fod_screen.sh  (array=0-11 covers 279 reactions in 12 chunks)

set -e

module load Python/3.13.5-GCCcore-14.3.0
mkdir -p /home/energy/s242862/logs

RXN_LIST=/home/energy/s242862/fod_rxn_list.txt
N_WORKERS=6
OMP_PER_WORKER=4
CHUNK_SIZE=24    # reactions per node (slightly over N_WORKERS, some workers do 2)

export OMP_NUM_THREADS=${OMP_PER_WORKER}
export PYSCF_MAX_MEMORY=9000   # MB per worker
export N_WORKERS CHUNK_SIZE

TASK=${SLURM_ARRAY_TASK_ID}
export START=$(( TASK * CHUNK_SIZE ))

python3 - <<'PYEOF'
import os, sys, json, subprocess, multiprocessing as mp

rxn_list  = '/home/energy/s242862/fod_rxn_list.txt'
script    = '/home/energy/s242862/pipeline/screen_fod.py'
n_workers = int(os.environ['N_WORKERS'])
start     = int(os.environ['START'])
chunk     = int(os.environ['CHUNK_SIZE'])

rxns = [l.strip() for l in open(rxn_list) if l.strip()]
batch = rxns[start : start + chunk]

if not batch:
    print(f"Task {os.environ['SLURM_ARRAY_TASK_ID']}: no reactions — done")
    sys.exit(0)

print(f"Task {os.environ['SLURM_ARRAY_TASK_ID']}: {len(batch)} reactions, workers={n_workers}", flush=True)

def run_one(rxn):
    result = subprocess.run(
        ['python3', script, rxn],
        capture_output=True, text=True
    )
    print(result.stdout, end='', flush=True)
    if result.returncode != 0:
        print(f'ERROR {rxn}:\n{result.stderr[-500:]}', flush=True)

with mp.Pool(n_workers) as pool:
    pool.map(run_one, batch)

print("Batch done.", flush=True)
PYEOF
