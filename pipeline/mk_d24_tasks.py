"""ORCA task list of the DFT test at found structures (D24).

For each eSEN run from the Transition1x start in results/d24_set.csv: the top image,
and for runs that met the criterion also the reactant, as plain xyz in ~/d24_xyz.
Task lines rxn_ts:xyz:nel and rxn_r:xyz:nel, for pipeline/job_border_zone.sh.

    python pipeline/mk_d24_tasks.py neb_pilot ~/d24_tasks_pilot.txt
"""
import csv
import json
import os
import sys

from ase.io import read, write

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
root, taskfile = sys.argv[1], sys.argv[2]
os.makedirs(f'{H}/d24_xyz', exist_ok=True)
lines = []
for r in csv.DictReader(open(f'{REPO}/results/d24_set.csv')):
    if r['root'] != root:
        continue
    d = f"{H}/{root}/t1x/esen/{r['rxn']}"
    if not os.path.exists(f'{d}/result.json'):
        print('no result', r['rxn'])
        continue
    res = json.load(open(f'{d}/result.json'))
    if not os.path.exists(f'{d}/transition_state.xyz'):
        print('no structure', r['rxn'], res['status'])
        continue
    todo = [('ts', 'transition_state.xyz')] + ([('r', 'reactant.xyz')] if res['criterion_met'] else [])
    for tag, name in todo:
        a = read(f'{d}/{name}')
        p = f"{H}/d24_xyz/{r['rxn']}_{tag}.xyz"
        write(p, a, format='xyz')
        lines.append(f"{r['rxn']}_{tag}:{p}:{int(a.numbers.sum())}")
open(taskfile, 'w').write('\n'.join(lines) + '\n')
print(f'{len(lines)} structures, {len(lines) // 2 + len(lines) % 2} tasks, {taskfile}')
