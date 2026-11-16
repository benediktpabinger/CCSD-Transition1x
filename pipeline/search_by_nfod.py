"""Search success by N_FOD, and the success expected for a test set with the N_FOD mix of
Transition1x (for comparison with Zhao et al., Adv. Sci. 2025, 69 % intended on Transition1x).

Freezing string + Sella on the four models (D27, D30), D35 check: a converged search passes if
it is class route. N_FOD at the stored TS (results/fod_ts_all.csv). Bins of N_FOD; the
Transition1x share of each bin over the 10 009 reactions found in OMol25; weighted success =
sum over bins of (share x success in the bin).

Output results/search_by_nfod.csv (per model and bin), printed weighted success per model

    python pipeline/search_by_nfod.py
"""
import csv
import glob
import json
import os

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
M = ('esen', 'uma-s', 'uma-m', 'mace-omol')
EDGES = [0, .1, .2, .3, .5, .8, 9]

fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
found = [k for k in (r['rxn'] for r in csv.DictReader(open(f'{RES}/omol25_ts_all.csv')) if r['found'] == '1') if k in fod]
allf = np.array([fod[k] for k in found])
share = [float(((allf >= a) & (allf < b)).mean()) for a, b in zip(EDGES[:-1], EDGES[1:])]

rows = []
for m in M:
    J = [json.load(open(f)) for f in glob.glob(f'{H}/neb_fsm/check_v2/{m}/*_fsm.json')]
    n = np.array([fod[j['rxn']] for j in J])
    ok = np.array([j['cls'] == 'route' for j in J])
    w = 0.0
    for (a, b), s in zip(zip(EDGES[:-1], EDGES[1:]), share):
        sel = (n >= a) & (n < b)
        rate = float(ok[sel].mean())
        w += s * rate
        rows.append(dict(model=m, nfod_lo=a, nfod_hi=b if b < 9 else '', n_checked=int(sel.sum()),
                         success=rate, t1x_share=s))
    print(f'{m:10s} Transition1x-weighted success {100 * w:.1f} %')
    rows.append(dict(model=m, nfod_lo='weighted', nfod_hi='', n_checked=len(J), success=w, t1x_share=1.0))
with open(f'{RES}/search_by_nfod.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    for r in rows:
        wr.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
