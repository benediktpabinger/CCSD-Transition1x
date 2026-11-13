"""D40: model barrier against DFT at the freezing-string structures of all four models.

DFT barrier = E(TS) - E(R), both the EnGrad on the UKS orbitals (gradient-type, D36), at the
saddle (D37) and the reactant each model relaxed (D40); model barrier = barrier_ev of the run,
from the same two structures. Reading fixed in D40: "barriers stay mostly within chemical
accuracy" holds if the median absolute error is below 43 meV in every group on at least three
of four models.

Output results/d40_barriers.csv, results/d40_barriers_missing.csv

    python pipeline/d40_table.py
"""
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d37_table import M, runs  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
HA = 27211.386245988
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')


def energy(p):
    t = open(p, errors='replace').read() if os.path.exists(p) else ''
    if 'CARTESIAN GRADIENT' not in t:
        return None
    return float(E_RE.findall(t)[-1])


rows, missing = [], []
for m in M:
    for rx, r, d, dft in runs(m):
        et = energy(f'{dft}/uks_engrad.out')
        if et is None:
            continue                                   # no TS gradient: already listed by D37
        er = energy(f'{H}/orca6_fsm_models/{rx}_{m}_r/uks_engrad.out')
        if er is None:
            missing.append(dict(model=m, rxn=rx, group=r['group'], reason='no reactant gradient'))
            continue
        bm = 1000 * json.load(open(f'{d}/result.json'))['barrier_ev']
        bd = (et - er) * HA
        rows.append(dict(model=m, rxn=rx, group=r['group'], nfod=float(r['nfod']), barrier_model_mev=bm,
                         barrier_dft_mev=bd, error_mev=bm - bd))
for name, R in (('d40_barriers', rows), ('d40_barriers_missing', missing)):
    with open(f'{RES}/{name}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(R[0].keys()) if R else ['model', 'rxn', 'group', 'reason'])
        w.writeheader()
        for x in R:
            w.writerow({k: ('%.2f' % v if isinstance(v, float) else v) for k, v in x.items()})
print(len(rows), 'barriers; missing', len(missing), [(x['model'], x['rxn']) for x in missing])
print('%-10s %-9s %4s %10s %10s %8s' % ('model', 'group', 'n', '|err| med', 'err med', '<=43meV'))
ok_models = 0
for m in M:
    good = True
    for g in ('baseline', 'closed', 'bs', 'high'):
        e = np.array([x['error_mev'] for x in rows if x['model'] == m and x['group'] == g])
        med = np.median(np.abs(e))
        good &= med < 43
        print('%-10s %-9s %4d %10.1f %10.1f %7.0f%%' % (m, g, len(e), med, np.median(e), 100 * np.mean(np.abs(e) <= 43)))
    ok_models += good
print(f'median |error| below 43 meV in every group on {ok_models} of 4 models -> D40 statement {"holds" if ok_models >= 3 else "does not hold"}')
