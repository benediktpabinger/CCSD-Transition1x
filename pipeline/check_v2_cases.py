"""Case lists for the rerun of the D23 check with the D35 fixes, one per check set.

Freezing string and interpolated-start NEB are listed from the run results (every
converged run of the 510 reactions of D24 and D29), so a later call adds runs that
finished since. The pilot lower-barrier cases and the eSEN Transition1x-start NEBs
are finished and copied from the first check. Prints the count of each set next to
the first check.

    python pipeline/check_v2_cases.py
"""
import csv
import json
import os
import shutil

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
M = ('uma-s', 'uma-m', 'esen', 'mace-omol')
SETS = [r for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{RES}/{n}.csv'))]

# name: (new outdir, root, old outdirs)
CHECKS = {
    'pilot': (f'{H}/neb_pilot/check_v2', f'{H}/neb_pilot', [f'{H}/neb_pilot/check']),
    'idpp': (f'{H}/neb_pilot/check_idpp_v2', f'{H}/neb_pilot', [f'{H}/neb_pilot/check_idpp']),
    'fsm': (f'{H}/neb_fsm/check_v2', f'{H}/neb_fsm',
            [f'{H}/neb_fsm/check'] + [f'{H}/neb_fsm/check_{m}' for m in M if m != 'esen']),
    'd24': (f'{H}/neb_d24/check_v2', f'{H}/neb_d24_all', [f'{H}/neb_d24/check']),
    'd29': (f'{H}/neb_d29/check_v2', f'{H}/neb_d29', [f'{H}/neb_d29/check']),
}


def converged(root, start):
    out = []
    for m in M:
        for r in SETS:
            p = f"{root}/{start}/{m}/{r['rxn']}/result.json"
            if os.path.exists(p) and json.load(open(p))['criterion_met']:
                out.append([r['rxn'], r['group'], r['bin'], m, start])
    return out


for name, (new, root, old) in CHECKS.items():
    os.makedirs(new, exist_ok=True)
    n_old = sum(sum(1 for _ in csv.DictReader(open(f'{d}/cases.csv'))) for d in old)
    if name in ('fsm', 'idpp'):
        rows = converged(root, name)
        with open(f'{new}/cases.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['rxn', 'group', 'bin', 'model', 'start'])
            w.writerows(rows)
        n = len(rows)
    else:
        shutil.copy(f'{old[0]}/cases.csv', f'{new}/cases.csv')
        n = n_old
    print(f'{name:6s} {n:5d} structures (first check {n_old})  {new}')
