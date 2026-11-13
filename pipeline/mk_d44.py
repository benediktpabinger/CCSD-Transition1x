"""D44: selection for the DFT saddle optimisations of the geometry-effect test.

MACE-OMol freezing-string saddles with a D37 transition-state gradient, three groups, 30 each,
drawn at random (seed 0, from the sorted list): low-MR baseline; closed-shell reactions with
N_FOD >= 0.5 (twins); broken-symmetry reactions (bs and N_FOD >= 1). A drawn reaction whose
D32 pilot run already holds a converged OptTS on a consistent solution is taken from the pilot
and not recomputed. Writes ~/geomfx/<rxn>/ts_model.xyz and the task list for
pipeline/job_marks_dft.sh (ROOT=~/geomfx, STOP_AFTER=optts).

    python pipeline/mk_d44.py -> results/d44_selection.csv, ~/geomfx/tasks.txt
"""
import csv
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d37_table import dft_forces, runs  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
ROOT = f'{H}/geomfx'
pilot = {r['run']: r for r in csv.DictReader(open(f'{RES}/marks_pilot.csv'))}
usable = {k for k, r in pilot.items() if '/' not in k and r.get('optts_conv') == '1'
          and r['outcome'] not in ('solution_switch', 'technical', 'pending')}

pool = {'baseline': [], 'cs_mr': [], 'bs': []}
for rx, r, d, dft in runs('mace-omol'):
    if dft_forces(f'{dft}/uks_engrad.out') is None:
        continue
    g = r['group']
    if g == 'baseline':
        pool['baseline'].append((rx, d, r))
    elif g == 'closed' and float(r['nfod']) >= 0.5:
        pool['cs_mr'].append((rx, d, r))
    elif g in ('bs', 'high'):
        pool['bs'].append((rx, d, r))
rng = random.Random(0)
rows, tasks = [], []
for g, P in pool.items():
    for rx, d, r in rng.sample(sorted(P, key=lambda t: t[0]), 30):
        src = 'pilot' if rx in usable else 'new'
        rows.append(dict(rxn=rx, group=g, nfod=r['nfod'], source=src))
        if src == 'new':
            os.makedirs(f'{ROOT}/{rx}', exist_ok=True)
            shutil.copy(f'{d}/transition_state.xyz', f'{ROOT}/{rx}/ts_model.xyz')
            tasks.append(rx)
    print(g, len(P), 'in the pool, 30 drawn,', sum(1 for x in rows if x['group'] == g and x['source'] == 'pilot'), 'from the pilot')
with open(f'{RES}/d44_selection.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
open(f'{ROOT}/tasks.txt', 'w').write('\n'.join(tasks) + '\n')
print(len(tasks), 'new DFT saddle optimisations')
