"""Does the force error grow when a search leaves the training path?

Same reaction, same model, two geometries:
    stored   the Transition1x transition state, a training point of the model
             (results/model_sp_ts_all.csv, against the OMol25 label)
    found    the transition state the model delivered in the thesis
             (results/omol25_model_geoms.csv, against our UKS DFT at that geometry)

Compared: mean component error (f_mae / f_err_mae) and largest component error
(f_maxdiff / f_err_max). Pairs are grouped by the class of the found structure,
<S^2> at that geometry, as in the thesis. Intervals resample reactions, because
the three models of one reaction are not independent.

Also reported: the Spearman correlation of the growth with the distance between
stored and found structure, and a control where the groups are compared at equal
N_FOD (the 45 reactions overlap only in the high-MR range).

Output results/force_error_off_path.csv, one row per reaction and model.

    python pipeline/force_error_off_path.py
"""
import csv
import json
import os

import numpy as np
from ase.io import read
from scipy.stats import spearmanr

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
DIRS = {'uma-s': 'uma_neb_results', 'uma-m': 'uma_m_neb_results', 'esen': 'esen_neb_results'}


def rmsd(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((A @ R - B) ** 2).sum(1).mean()))


stored = {(r['rxn'], r['model']): r for r in csv.DictReader(open(f'{RES}/model_sp_ts_all.csv'))}
fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
ts = json.load(open(f'{H}/t1x_ts_all.json'))
rows = []
for r in csv.DictReader(open(f'{RES}/omol25_model_geoms.csv')):
    s = stored.get((r['rxn'], r['model']))
    if s is None:
        continue
    found = read(f"{H}/{DIRS[r['model']]}/{r['rxn']}/transition_state.xyz").positions
    rows.append(dict(rxn=r['rxn'], model=r['model'], group='bs' if r['unstable_ts'] == '1' else 'closed',
                     nfod=fod[r['rxn']], dist_A=rmsd(found, np.array(ts[r['rxn']]['positions'])),
                     mae_stored=float(s['f_mae']), mae_found=float(r['f_err_mae']),
                     max_stored=float(s['f_maxdiff']), max_found=float(r['f_err_max'])))

rng = np.random.default_rng(0)


def med_ratio(S, a, b, n=5000):
    """ratio of medians with a 95 % interval over reactions."""
    rx = sorted({r['rxn'] for r in S})
    by = {x: [r for r in S if r['rxn'] == x] for x in rx}
    val = np.median([r[a] for r in S]) / np.median([r[b] for r in S])
    out = []
    for _ in range(n):
        T = [r for x in rng.choice(rx, len(rx)) for r in by[x]]
        out.append(np.median([r[a] for r in T]) / np.median([r[b] for r in T]))
    return val, *np.percentile(out, [2.5, 97.5])


print('%d pairs (reaction, model), %d reactions' % (len(rows), len({r['rxn'] for r in rows})))
for lab, S0 in (('all N_FOD', rows), ('N_FOD 0.70 to 0.90, equal MR', [r for r in rows if 0.70 <= r['nfod'] <= 0.90])):
    print('\n' + lab)
    print('%-7s %-5s %4s %5s   %-9s %-9s %s' % ('group', 'kind', 'n', 'rxns', 'stored', 'found', 'found/stored [95 %]'))
    for g in ('closed', 'bs'):
        S = [r for r in S0 if r['group'] == g]
        for kind, a, b in (('mae', 'mae_found', 'mae_stored'), ('max', 'max_found', 'max_stored')):
            v, lo, hi = med_ratio(S, a, b)
            print('%-7s %-5s %4d %5d   %-9.4f %-9.4f %.2f [%.2f, %.2f]'
                  % (g, kind, len(S), len({r['rxn'] for r in S}), np.median([r[b] for r in S]),
                     np.median([r[a] for r in S]), v, lo, hi))
print('\nby model, largest error, median found/stored')
for m in DIRS:
    print('  %-6s' % m, '  '.join('%s %.2f' % (g, np.median([r['max_found'] for r in rows if r['model'] == m and r['group'] == g])
                                               / np.median([r['max_stored'] for r in rows if r['model'] == m and r['group'] == g]))
                                  for g in ('closed', 'bs')))
print('\ngrowth against distance from the stored structure')
for g in ('closed', 'bs'):
    S = [r for r in rows if r['group'] == g]
    x, y = [r['dist_A'] for r in S], [r['mae_found'] / r['mae_stored'] for r in S]
    print('  %-7s median distance %.3f A, median growth %.2f, Spearman rho %.2f' % (g, np.median(x), np.median(y), spearmanr(x, y).correlation))
with open(f'{RES}/force_error_off_path.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
print('results/force_error_off_path.csv')
