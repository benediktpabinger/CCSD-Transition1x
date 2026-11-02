#!/bin/bash
#SBATCH --job-name=fsmstart
#SBATCH --partition=xeon24el9_test
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/energy/bepab/logs/fsmstart_%j.out
#SBATCH --error=/home/energy/bepab/logs/fsmstart_%j.err

# Submit with --dependency on the FSM job: lists the converged structures, then runs
# the Hessian and downhill check of pipeline/pilot_check.py on them (D27).

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
cd $HOME/Transition1x
mkdir -p $HOME/neb_fsm/check
python3 - <<'PY'
import csv, json, os
H = os.path.expanduser('~')
n = 0
with open(f'{H}/neb_fsm/check/cases.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'group', 'bin', 'model', 'start'])
    for r in csv.DictReader(open('results/d24_set.csv')):
        p = f"{H}/neb_fsm/fsm/esen/{r['rxn']}/result.json"
        if os.path.exists(p) and json.load(open(p))['criterion_met']:
            w.writerow([r['rxn'], r['group'], r['bin'], 'esen', 'fsm'])
            n += 1
print(n, 'structures to check')
PY
sbatch --export=ALL,MODEL=esen,OUTDIR=$HOME/neb_fsm/check,ROOT=$HOME/neb_fsm pipeline/job_pilot_check.sh
