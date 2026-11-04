"""ORCA task lists of the DFT test at the freezing-string structures of all four models (D37).

Every converged freezing-string run (Sella criterion met) of the 510 reactions of D24 and
D29, as D28 did for eSEN on the 310 of D24: the refined structure goes through the D24
protocol of pipeline/job_border_zone.sh. eSEN structures already computed for D28 are
skipped, so eSEN adds only the D29 reactions. Task lines name:xyz:nel.

    python pipeline/mk_d37_tasks.py
      -> ~/d37_tasks_esen.txt  (output ~/orca6_fsm, names rxn_ts as in D28)
         ~/d37_tasks_models.txt (output ~/orca6_fsm_models, names rxn_<model>_ts)
"""
import csv
import json
import os

from ase.io import read

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SETS = [r for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{REPO}/results/{n}.csv'))]
out = {'esen': [], 'models': []}
for m in ('esen', 'uma-s', 'uma-m', 'mace-omol'):
    for r in SETS:
        d = f"{H}/neb_fsm/fsm/{m}/{r['rxn']}"
        if not (os.path.exists(f'{d}/result.json') and json.load(open(f'{d}/result.json'))['criterion_met']):
            continue
        nel = int(read(f'{d}/transition_state.xyz').numbers.sum())
        if m == 'esen':
            if os.path.exists(f"{H}/orca6_fsm/{r['rxn']}_ts"):
                continue
            out['esen'].append(f"{r['rxn']}_ts:{d}/transition_state.xyz:{nel}")
        else:
            out['models'].append(f"{r['rxn']}_{m}_ts:{d}/transition_state.xyz:{nel}")
for k, L in out.items():
    open(f'{H}/d37_tasks_{k}.txt', 'w').write('\n'.join(L) + '\n')
    print(k, len(L), 'structures')
