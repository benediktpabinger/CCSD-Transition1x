"""Is the rise of force error with N_FOD only a marker for stretched bonds?

At every stored transition state: stretch = largest d / (r_i + r_j) over the atom pairs
whose bond changes between reactant and product (bond graph as in pilot_check.py), the
Transition1x endpoints. Force error from results/model_sp_ts_all.csv, closed-shell labels
(<S^2> <= 0.001) only, so broken symmetry is not in the comparison.

Reading fixed before the run: the N_FOD effect is real if, within quartiles of stretch and
at fixed atom count, the median force error of N_FOD >= 0.5 stays above that of N_FOD < 0.2
for all four models. If the ratio falls to about 1, N_FOD was a marker for stretch.

Output results/mr_vs_stretch.csv

    python pipeline/mr_vs_stretch.py
"""
import csv
import json
import os

import h5py
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, natural_cutoffs
from scipy.stats import rankdata

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
M = ('uma-s', 'uma-m', 'esen', 'mace-omol')


def bonds(Z, x):
    a = Atoms(numbers=Z, positions=x)
    nl = NeighborList([1.15 * c for c in natural_cutoffs(a)], self_interaction=False, bothways=False, skin=0)
    nl.update(a)
    return {(min(i, int(j)), max(i, int(j))) for i in range(len(a)) for j in nl.get_neighbors(i)[0]}


ts = json.load(open(f'{H}/t1x_ts_all.json'))
fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
s2 = {r['rxn']: float(r['s2']) for r in csv.DictReader(open(f'{RES}/omol25_ts_all.csv')) if r['found'] == '1'}
geo = {}
with h5py.File(f'{H}/data/Transition1x.h5', 'r') as f:
    for split in f:
        for formula in f[split]:
            for rx in f[split][formula]:
                if rx not in s2 or rx not in ts:
                    continue
                g = f[split][formula][rx]
                Z = g['atomic_numbers'][:]
                bR, bP = bonds(Z, g['positions'][0]), bonds(Z, g['positions'][9])
                x = np.array(ts[rx]['positions'])
                pairs = bR ^ bP
                ratio = [np.linalg.norm(x[i] - x[j]) / (covalent_radii[Z[i]] + covalent_radii[Z[j]]) for i, j in pairs]
                geo[rx] = dict(stretch=max(ratio) if ratio else np.nan, n_atoms=len(Z), n_changed=len(pairs))
print(len(geo), 'reactions with geometry')
rows = []
for r in csv.DictReader(open(f'{RES}/model_sp_ts_all.csv')):
    rx = r['rxn']
    if rx in geo and s2[rx] <= 1e-3 and np.isfinite(geo[rx]['stretch']):
        rows.append(dict(rxn=rx, model=r['model'], nfod=fod[rx], f_mae=float(r['f_mae']), f_max=float(r['f_maxdiff']), **geo[rx]))
print(len(rows), 'rows, closed-shell labels')


def partial(y, x, z):
    """Spearman of y and x with the columns of z removed."""
    ry, rx = rankdata(y), rankdata(x)
    Z = np.column_stack([rankdata(c) for c in z] + [np.ones(len(y))])
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    return float(np.corrcoef(ey, ex)[0, 1])


out = []
for m in M:
    S = [r for r in rows if r['model'] == m]
    y = np.array([r['f_mae'] for r in S])
    n, st, na = (np.array([r[k] for r in S]) for k in ('nfod', 'stretch', 'n_atoms'))
    print(f'\n{m}: n={len(S)}  Spearman force error with N_FOD {partial(y, n, []):.2f}, with stretch {partial(y, st, []):.2f}, with atoms {partial(y, na, []):.2f}')
    print(f'   partial: N_FOD given stretch+atoms {partial(y, n, [st, na]):.2f}; stretch given N_FOD+atoms {partial(y, st, [n, na]):.2f}; N_FOD vs stretch {partial(n, st, []):.2f}')
    q = np.percentile(st, [25, 50, 75])
    for lo, hi, lab in ((0, q[0], 'Q1'), (q[0], q[1], 'Q2'), (q[1], q[2], 'Q3'), (q[2], 9, 'Q4')):
        rat = []
        for a in range(6, 24):
            A = [i for i in range(len(S)) if lo <= st[i] < hi and na[i] == a and n[i] < 0.2]
            B = [i for i in range(len(S)) if lo <= st[i] < hi and na[i] == a and n[i] >= 0.5]
            if len(A) >= 5 and len(B) >= 5:
                rat.append((np.median(y[B]) / np.median(y[A]), min(len(A), len(B))))
        if rat:
            w = np.array([v[1] for v in rat], float)
            v = float(np.average([x[0] for x in rat], weights=w))
            print('   stretch %s %.2f-%.2f: high/low N_FOD force error %.2f over %d atom counts, n low %d high %d'
                  % (lab, lo, min(hi, st.max()), v, len(rat), ((st >= lo) & (st < hi) & (n < .2)).sum(), ((st >= lo) & (st < hi) & (n >= .5)).sum()))
            out.append(dict(model=m, quartile=lab, ratio=v, atom_counts=len(rat)))
with open(f'{RES}/mr_vs_stretch.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    for r in out:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
