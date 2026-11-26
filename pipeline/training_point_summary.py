"""Paper 3.4: force error at the Transition1x transition states that are in the OMol25 training split.

Per model, from results/model_sp_ts_all.csv (model vs OMol25 label at the stored TS), split == train only:
  low   OMol25 closed-shell, N_FOD < 0.2       high   N_FOD >= 1 (either class)
  force error = f_mae (meV/A); ratio = median(high) / median(low), 95 % bootstrap CI (5000, seed 0)
Spin class is OMol25's (group_omol) until the D46 label check gives ours at all 10,009 TS.

Output results/training_point_summary.csv

    python pipeline/training_point_summary.py
"""
import csv
import os

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
rows = [r for r in csv.DictReader(open(f'{RES}/model_sp_ts_all.csv')) if r['split'] == 'train' and r['f_mae']]
rng = np.random.default_rng(0)


def ratio(a, b, n=5000):
    a, b = np.array(a), np.array(b)
    r = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b))) for _ in range(n)]
    return np.median(a) / np.median(b), np.percentile(r, 2.5), np.percentile(r, 97.5)


out = []
for m in ('esen', 'uma-s', 'uma-m', 'mace-omol'):
    R = [r for r in rows if r['model'] == m]
    lo = [1000 * float(r['f_mae']) for r in R if float(r['nfod']) < 0.2 and r['group_omol'] == 'closed']
    hi = [1000 * float(r['f_mae']) for r in R if float(r['nfod']) >= 1]
    q, a, b = ratio(hi, lo)
    out.append(dict(model=m, n_low=len(lo), n_high=len(hi), median_low=round(float(np.median(lo)), 2),
                    median_high=round(float(np.median(hi)), 2), ratio=round(q, 2), ratio_lo=round(a, 2), ratio_hi=round(b, 2)))
with open(f'{RES}/training_point_summary.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for r in out:
    print(r)
