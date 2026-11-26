"""Paper Discussion and SI S6: what the rise of the force error with N_FOD is not caused by.

All at the Transition1x transition states (no new compute):
  mace-t1x   thesis MACE (Transition1x train split, wB97X/6-31G(d), restricted labels only) against its own labels,
             Transition1x train split only (results/mace_t1x_ts_errors.csv); N_FOD and OMol25 spin class from
             results/model_sp_ts_all.csv. Ratio high/low as in pipeline/training_point_summary.py, and BS/CS
             medians at 0.4 <= N_FOD < 0.9 (OMol25 class). Bootstrap 5000, seed 0.
  gap        Spearman of the force error (OMol25 models and mace-t1x) with the OMol25 HOMO-LUMO gap (smaller of
             alpha/beta, ~/omol25_ts_all.jsonl) and with N_FOD, and partial rank correlations of each given the other.
  roughness  D39 pilot (results/label_roughness.csv, 275 reactions): Spearman of roughness and jump rate with N_FOD,
             of the OMol25 model errors with roughness and jump rate, and partial error~jump given N_FOD.

Output results/cause_checks.csv (check, model, quantity, value, lo, hi, n)

    python pipeline/cause_checks.py
"""
import csv
import json
import os
import re

import numpy as np
from scipy.stats import rankdata, spearmanr

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
M = ('esen', 'uma-s', 'uma-m', 'mace-omol')
rng = np.random.default_rng(0)
out = []


def add(check, model, quantity, value, lo='', hi='', n=''):
    out.append(dict(check=check, model=model, quantity=quantity, value=round(float(value), 3),
                    lo='' if lo == '' else round(float(lo), 3), hi='' if hi == '' else round(float(hi), 3), n=n))


def ratio(a, b, n=5000):
    a, b = np.array(a), np.array(b)
    r = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b))) for _ in range(n)]
    return np.median(a) / np.median(b), np.percentile(r, 2.5), np.percentile(r, 97.5)


def partial(a, b, c):
    """Rank correlation of a and b after removing the linear rank dependence of each on c."""
    def res(x):
        x, z = rankdata(x), rankdata(c)
        return x - np.polyval(np.polyfit(z, x, 1), z)
    return np.corrcoef(res(a), res(b))[0, 1]


sp = list(csv.DictReader(open(f'{RES}/model_sp_ts_all.csv')))
nf = {r['rxn']: float(r['nfod']) for r in sp if r['model'] == 'esen'}
cls = {r['rxn']: r['group_omol'] for r in sp if r['model'] == 'esen'}
err = {m: {r['rxn']: float(r['f_mae']) for r in sp if r['model'] == m and r['f_mae']} for m in M}

t1 = [r for r in csv.DictReader(open(f'{RES}/mace_t1x_ts_errors.csv')) if r['split'] == 'train' and r['rxn'] in nf]
err['mace-t1x'] = {r['rxn']: float(r['f_mae']) for r in t1}
E = err['mace-t1x']
lo = [1000 * E[k] for k in E if nf[k] < 0.2 and cls[k] == 'closed']
hi = [1000 * E[k] for k in E if nf[k] >= 1]
add('mace-t1x', 'mace-t1x', 'median_low', np.median(lo), n=len(lo))
add('mace-t1x', 'mace-t1x', 'median_high', np.median(hi), n=len(hi))
add('mace-t1x', 'mace-t1x', 'ratio_high_low', *ratio(hi, lo))
s1 = [1000 * E[k] for k in E if 0.4 <= nf[k] < 0.9 and cls[k] != 'closed']
s0 = [1000 * E[k] for k in E if 0.4 <= nf[k] < 0.9 and cls[k] == 'closed']
add('mace-t1x', 'mace-t1x', 'bs_cs_mid', *ratio(s1, s0), n=f'{len(s1)}/{len(s0)}')

gap = {}
for line in open(f'{H}/omol25_ts_all.jsonl'):
    d = json.loads(line)
    m = re.search(r't1x_(rxn\d+)_', d['source'])
    if m:
        gap[m.group(1)] = min(d['homo_lumo_gap']['__ndarray__'][2])
for m in M + ('mace-t1x',):
    K = [k for k in err[m] if k in gap]
    e, g, n = (np.array([d[k] for k in K]) for d in (err[m], gap, nf))
    add('gap', m, 'rho_err_gap', spearmanr(e, g)[0], n=len(K))
    add('gap', m, 'rho_err_nfod', spearmanr(e, n)[0], n=len(K))
    add('gap', m, 'rho_gap_nfod', spearmanr(g, n)[0], n=len(K))
    add('gap', m, 'partial_err_gap_given_nfod', partial(e, g, n), n=len(K))
    add('gap', m, 'partial_err_nfod_given_gap', partial(e, n, g), n=len(K))

rough = list(csv.DictReader(open(f'{RES}/label_roughness.csv')))
col = lambda S, k: np.array([float(r[k]) for r in S])
add('roughness', 'labels', 'rho_roughness_nfod', spearmanr(col(rough, 'roughness'), col(rough, 'nfod'))[0], n=len(rough))
add('roughness', 'labels', 'rho_jump_nfod', spearmanr(col(rough, 'jump_rate'), col(rough, 'nfod'))[0], n=len(rough))
add('roughness', 'labels', 'rho_jump_fracbs', spearmanr(col(rough, 'jump_rate'), col(rough, 'frac_bs_frames'))[0], n=len(rough))
add('roughness', 'labels', 'max_nfod', col(rough, 'nfod').max(), n=int((col(rough, 'nfod') >= 1).sum()))
for m in M:
    S = [r for r in rough if r['err_' + m] not in ('', 'nan')]
    e, ro, j, n = col(S, 'err_' + m), col(S, 'roughness'), col(S, 'jump_rate'), col(S, 'nfod')
    add('roughness', m, 'rho_err_roughness', spearmanr(e, ro)[0], n=len(S))
    add('roughness', m, 'rho_err_jump', spearmanr(e, j)[0], n=len(S))
    add('roughness', m, 'partial_err_jump_given_nfod', partial(e, j, n), n=len(S))
    add('roughness', m, 'partial_err_nfod_given_jump', partial(e, n, j), n=len(S))

with open(f'{RES}/cause_checks.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for r in out:
    print(r)
