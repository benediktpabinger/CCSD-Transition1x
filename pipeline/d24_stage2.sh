#!/bin/bash
# Second stage of D24, runs detached on the login node: waits for the new NEBs, then
# DFT at their structures, then the Hessian and downhill check of all eSEN structures.
source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
cd $HOME/Transition1x
while [ -n "$(squeue -u $USER -h -n d24neb)" ]; do sleep 120; done
python3 pipeline/mk_d24_tasks.py neb_d24 $HOME/d24_tasks_new.txt
mkdir -p $HOME/neb_d24_all/t1x/esen $HOME/neb_d24/check
python3 - <<'PY'
import csv, json, os
H = os.path.expanduser('~')
with open(f'{H}/neb_d24/check/cases.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'group', 'bin', 'model', 'start'])
    for r in csv.DictReader(open('results/d24_set.csv')):
        d = f"{H}/{r['root']}/t1x/esen/{r['rxn']}"
        l = f"{H}/neb_d24_all/t1x/esen/{r['rxn']}"
        if not os.path.lexists(l):
            os.symlink(d, l)
        if os.path.exists(f'{d}/result.json') and json.load(open(f'{d}/result.json'))['criterion_met']:
            w.writerow([r['rxn'], r['group'], r['bin'], 'esen', 't1x'])
PY
sbatch --export=ALL,MODEL=esen,OUTDIR=$HOME/neb_d24/check,ROOT=$HOME/neb_d24_all pipeline/job_pilot_check.sh
N=$(wc -l < $HOME/d24_tasks_new.txt)
pipeline/chunk_submit.sh $HOME/logs/d24_new.state $(( (N + 1) / 2 )) 46 "--job-name=d24dft --export=ALL,TASKFILE=$HOME/d24_tasks_new.txt,OUTROOT=$HOME/orca6_d24 --output=$HOME/logs/d24dft_%A_%a.out --error=$HOME/logs/d24dft_%A_%a.err" pipeline/job_border_zone.sh
