#!/bin/bash
#SBATCH --job-name=idppstart
#SBATCH --partition=xeon24el9_test
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/energy/bepab/logs/idppstart_%j.out
#SBATCH --error=/home/energy/bepab/logs/idppstart_%j.err

# Submit with --dependency on the interpolated-start NEB arrays (D33): lists every
# converged run of the 510 reactions, then one check job per model.

source /etc/profile
module load Python/3.13.5-GCCcore-14.3.0
cd $HOME/Transition1x
python3 - <<'PY'
import csv, json, os
H = os.path.expanduser('~')
n = 0
with open(f'{H}/neb_pilot/check_idpp/cases.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'group', 'bin', 'model', 'start'])
    for m in ('uma-s', 'uma-m', 'esen', 'mace-omol'):
        for name in ('d24_set', 'd29_set'):
            for r in csv.DictReader(open(f'results/{name}.csv')):
                p = f"{H}/neb_pilot/idpp/{m}/{r['rxn']}/result.json"
                if os.path.exists(p) and json.load(open(p))['criterion_met']:
                    w.writerow([r['rxn'], r['group'], r['bin'], m, 'idpp'])
                    n += 1
print(n, 'converged runs to check')
PY
for M in uma-s uma-m esen mace-omol; do
  sbatch --job-name=chk-idpp-$M --export=ALL,MODEL=$M,OUTDIR=$HOME/neb_pilot/check_idpp,ROOT=$HOME/neb_pilot pipeline/job_pilot_check.sh
done
