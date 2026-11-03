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
export MODEL=${MODEL:-esen}
export CK=$HOME/neb_fsm/check
[ "$MODEL" != esen ] && export CK=$HOME/neb_fsm/check_$MODEL
mkdir -p $CK
python3 - <<'PY'
import csv, json, os
H = os.path.expanduser('~')
M, CK = os.environ['MODEL'], os.environ['CK']
n = 0
with open(f'{CK}/cases.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'group', 'bin', 'model', 'start'])
    for r in [r for name in ('d24_set', 'd29_set') if os.path.exists(f'results/{name}.csv') for r in csv.DictReader(open(f'results/{name}.csv'))]:
        p = f"{H}/neb_fsm/fsm/{M}/{r['rxn']}/result.json"
        if os.path.exists(p) and json.load(open(p))['criterion_met']:
            w.writerow([r['rxn'], r['group'], r['bin'], M, 'fsm'])
            n += 1
print(n, 'structures to check')
PY
sbatch --export=ALL,MODEL=$MODEL,OUTDIR=$CK,ROOT=$HOME/neb_fsm pipeline/job_pilot_check.sh
