"""Numbers for the paper Results, classified at the located transition state (D45, v6 figures).

Per model, at every located transition state with a DFT force (results/d37_forces.csv):
N_FOD from results/fod_model_ts.csv, spin class from the DFT <S^2> there (broken-symmetry if > 0.05).

  low   closed-shell, N_FOD < 0.2          high   N_FOD >= 1 (either class)
  force error = f_err_mae (meV/A), barrier error = |error_mev| (results/d40_barriers.csv)
  ratio = median(high) / median(low), 95 % bootstrap CI (5000 resamples, seed 0)
  share_43 = share of |barrier error| > 43 meV
  bs_cs: broken-symmetry / closed-shell, median ratio at 0.4 <= N_FOD < 0.9, same bootstrap

Output results/model_ts_summary.csv

    python pipeline/model_ts_summary.py
"""
import csv
import os

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
M = ('esen', 'uma-s', 'uma-m', 'mace-omol')

nf = {(r['model'], r['rxn']): float(r['nfod_model_ts']) for r in csv.DictReader(open(f'{RES}/fod_model_ts.csv'))}
F = {(r['model'], r['rxn']): r for r in csv.DictReader(open(f'{RES}/d37_forces.csv'))}
B = {(r['model'], r['rxn']): float(r['error_mev']) for r in csv.DictReader(open(f'{RES}/d40_barriers.csv'))}
rng = np.random.default_rng(0)


def bs(k):
    return abs(float(F[k]['s2'])) > 0.05


def force(S):
    return [1000 * float(F[k]['f_err_mae']) for k in S]


def barrier(S):
    return [abs(B[k]) for k in S if k in B]


def ratio(a, b, n=5000):
    a, b = np.array(a), np.array(b)
    r = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b))) for _ in range(n)]
    return np.median(a) / np.median(b), np.percentile(r, 2.5), np.percentile(r, 97.5)


rows = []
for m in M:
    K = [k for k in F if k[0] == m]
    lo = [k for k in K if nf[k] < 0.2 and not bs(k)]
    hi = [k for k in K if nf[k] >= 1]
    mid = [k for k in K if 0.4 <= nf[k] < 0.9]
    s1, s0 = [k for k in mid if bs(k)], [k for k in mid if not bs(k)]
    for q, fn in (('force', force), ('barrier', barrier)):
        r, a, b = ratio(fn(hi), fn(lo))
        c, ca, cb = ratio(fn(s1), fn(s0))
        rows.append(dict(model=m, quantity=q, n_low=len(fn(lo)), n_high=len(fn(hi)),
                         median_low=round(float(np.median(fn(lo))), 2), median_high=round(float(np.median(fn(hi))), 2),
                         ratio=round(r, 2), ratio_lo=round(a, 2), ratio_hi=round(b, 2),
                         share_43_low=round(100 * np.mean(np.array(fn(lo)) > 43), 1) if q == 'barrier' else '',
                         share_43_high=round(100 * np.mean(np.array(fn(hi)) > 43), 1) if q == 'barrier' else '',
                         n_bs=len(fn(s1)), n_cs=len(fn(s0)),
                         bs_cs=round(c, 2), bs_cs_lo=round(ca, 2), bs_cs_hi=round(cb, 2)))
with open(f'{RES}/model_ts_summary.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
