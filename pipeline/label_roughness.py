"""D39: roughness and consistency of the OMol25 labels along the Transition1x training paths.

Frames: every trans1x record in ~/omol25_hits_train.jsonl (OMol25 energy, forces, <S^2>),
grouped by reaction; reactions with at least 200 frames. For every frame, its nearest other
frame of the same reaction by RMSD (same atom order, no alignment: OMol25 kept the Transition1x
coordinates); pairs with 0.005 <= RMSD <= 0.05 A are used.

  roughness  RMS force difference / RMSD of the pair (eV/A^2), median per reaction
  jump       |dE - W| > 20 meV with W = -0.5 (F1 + F2) . (x2 - x1), the work along the straight
             step, or a class flip across <S^2> = 0.05; jump rate per reaction

Then, per reaction, Spearman correlations with N_FOD at the stored TS (fod_ts_all.csv) and with
each model's force error at the stored TS (model_sp_ts_all.csv, f_mae). Reading fixed in D39.

Output results/label_roughness.csv (per reaction), printed correlations

    python pipeline/label_roughness.py      (run as a short job: 700 MB of JSON)
"""
import collections
import csv
import json
import os
import re

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
RXN_RE = re.compile(r'trans1x/t1x_(rxn\d+)_')
M = ('uma-s', 'uma-m', 'esen', 'mace-omol')

frames = collections.defaultdict(list)
with open(f'{H}/omol25_hits_train.jsonl') as fh:
    for line in fh:
        if '"source": "trans1x' not in line:
            continue
        j = json.loads(line)
        rx = RXN_RE.search(j['source']).group(1)
        frames[rx].append((np.array(j['positions']), np.array(j['forces']), j['energy'], j['s_squared']))
print(len(frames), 'reactions', sum(len(v) for v in frames.values()), 'frames', flush=True)

fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
err = collections.defaultdict(dict)
for r in csv.DictReader(open(f'{RES}/model_sp_ts_all.csv')):
    if r['f_mae']:
        err[r['model']][r['rxn']] = float(r['f_mae'])

rows = []
for rx, F in frames.items():
    if len(F) < 200 or rx not in fod:
        continue
    X = np.array([f[0] for f in F])
    n_at = X.shape[1] // 3
    tree = cKDTree(X)
    d, idx = tree.query(X, k=2)
    rough, jumps, npair, done = [], 0, 0, set()
    for i in range(len(F)):
        j = int(idx[i, 1])
        rmsd = d[i, 1] / np.sqrt(n_at)
        if not (0.005 <= rmsd <= 0.05) or (min(i, j), max(i, j)) in done:
            continue
        done.add((min(i, j), max(i, j)))
        npair += 1
        dF = F[j][1] - F[i][1]
        rough.append(np.sqrt((dF ** 2).reshape(-1, 3).sum(1).mean()) / rmsd)
        W = -0.5 * np.dot(F[i][1] + F[j][1], F[j][0] - F[i][0])
        flip = (F[i][3] > 0.05) != (F[j][3] > 0.05)
        if abs((F[j][2] - F[i][2]) - W) > 0.020 or flip:
            jumps += 1
    if npair < 20:
        continue
    s2 = np.array([f[3] for f in F])
    rows.append(dict(rxn=rx, nfod=fod[rx], n_frames=len(F), n_pairs=npair, roughness=float(np.median(rough)),
                     jump_rate=jumps / npair, frac_bs_frames=float((s2 > 0.05).mean()),
                     **{f'err_{m}': err[m].get(rx, np.nan) for m in M}))
print(len(rows), 'reactions with enough neighbour pairs', flush=True)

with open(f'{RES}/label_roughness.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})

n = np.array([r['nfod'] for r in rows])
ro = np.array([r['roughness'] for r in rows])
ju = np.array([r['jump_rate'] for r in rows])
print('roughness median %.2f eV/A^2; jump rate median %.3f, reactions with any jump %d' % (np.median(ro), np.median(ju), (ju > 0).sum()))
print('roughness vs N_FOD: rho %.2f p %.2g' % spearmanr(ro, n))
print('jump rate vs N_FOD: rho %.2f p %.2g' % spearmanr(ju, n))
for lo, hi in ((0, .3), (.3, .6), (.6, 1.), (1., 9)):
    s = (n >= lo) & (n < hi)
    if s.sum():
        print(f'  N_FOD {lo}-{hi}: n={s.sum():3d} roughness {np.median(ro[s]):.2f} jump rate {np.mean(ju[s]):.3f}')
def partial(y, x, z):
    """Spearman of y and x with z removed (rank residuals, as in mr_vs_stretch.py)."""
    from scipy.stats import rankdata
    Z = np.column_stack([rankdata(z), np.ones(len(y))])
    ey = rankdata(y) - Z @ np.linalg.lstsq(Z, rankdata(y), rcond=None)[0]
    ex = rankdata(x) - Z @ np.linalg.lstsq(Z, rankdata(x), rcond=None)[0]
    return float(np.corrcoef(ey, ex)[0, 1])


for m in M:
    e = np.array([r[f'err_{m}'] for r in rows])
    ok = np.isfinite(e)
    print(f'{m:10s} error vs roughness rho %.2f p %.2g (N_FOD held fixed %.2f) | vs jump rate rho %.2f p %.2g | vs N_FOD rho %.2f'
          % (*spearmanr(e[ok], ro[ok]), partial(e[ok], ro[ok], n[ok]), *spearmanr(e[ok], ju[ok]), spearmanr(e[ok], n[ok])[0]))
