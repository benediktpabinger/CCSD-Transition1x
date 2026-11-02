"""Does force error grow with distance from the training path?

For each structure of the DFT test (results/d24_forces_<model>.csv), the RMSD to the
nearest frame of that reaction in Transition1x (all frames, atom order fixed, mirror
images not allowed). Force error against that distance, by group.

Output results/force_vs_path_distance.csv

    python pipeline/force_vs_path_distance.py
"""
import csv
import os

import h5py
import numpy as np
from ase.io import read
from scipy.stats import spearmanr

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
S = {r['rxn']: r for r in csv.DictReader(open(f'{RES}/d24_set.csv'))}


def nearest(x, P):
    A = x - x.mean(0)
    Bs = P - P.mean(1, keepdims=True)
    C = np.einsum('ji,kjl->kil', A, Bs)
    U, s, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ (np.eye(3)[None] * np.stack([np.ones_like(d), np.ones_like(d), d], 1)[:, None, :]) @ Vt
    D = A[None] @ Rm - Bs
    return float(np.sqrt((D ** 2).sum(2).mean(1)).min())


paths = {}
with h5py.File(f'{H}/data/Transition1x.h5', 'r') as f:
    for split in f:
        for formula in f[split]:
            for rx in f[split][formula]:
                if rx in S and rx not in paths:
                    paths[rx] = np.array(f[split][formula][rx]['positions'])
print(len(paths), 'paths, median frames', int(np.median([len(p) for p in paths.values()])))
out = []
for model in ('esen', 'uma-m'):
    for r in csv.DictReader(open(f'{RES}/d24_forces_{model}.csv')):
        if r['met'] != '1':
            continue
        rx = r['rxn']
        x = read(f"{H}/{S[rx]['root']}/t1x/{model}/{rx}/transition_state.xyz").positions
        out.append(dict(model=model, rxn=rx, group=r['group'], nfod=float(r['nfod']), dist=nearest(x, paths[rx]),
                        f_mae=float(r['f_err_mae']), f_max=float(r['f_err_max']), f_dft=float(r['f_dft'])))
with open(f'{RES}/force_vs_path_distance.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    for r in out:
        w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
for model in ('esen', 'uma-m'):
    T = [r for r in out if r['model'] == model]
    d = np.array([r['dist'] for r in T])
    print(f'\n{model}: n={len(T)} distance median {np.median(d):.3f} A, 90th pct {np.percentile(d, 90):.3f}, max {d.max():.3f}; share above 0.05 A {100*(d>.05).mean():.0f} %, above 0.1 A {100*(d>.1).mean():.0f} %')
    for g in ('baseline', 'closed', 'bs'):
        U = [r for r in T if r['group'] == g]
        if not U:
            continue
        x, y = np.array([r['dist'] for r in U]), np.array([r['f_mae'] for r in U])
        m = np.median(x)
        print('  %-9s n=%3d dist median %.3f  Spearman error-distance %.2f (p=%.2g)  error near half %.4f, far half %.4f, ratio %.2f'
              % (g, len(U), m, spearmanr(x, y).correlation, spearmanr(x, y).pvalue, np.median(y[x <= m]), np.median(y[x > m]), np.median(y[x > m]) / np.median(y[x <= m])))
    A = np.array([r['dist'] for r in T]); Y = np.array([r['f_mae'] for r in T]); N = np.array([r['nfod'] for r in T])
    for lab, sel in (('near half', A <= np.median(A)), ('far half', A > np.median(A))):
        hi, lo = Y[sel & (N >= .5)], Y[sel & (N < .5)]
        if len(lo) >= 5 and len(hi) >= 5:
            print(f'  {lab}: N_FOD >= 0.5 error {np.median(hi):.4f} (n={len(hi)}), N_FOD < 0.5 {np.median(lo):.4f} (n={len(lo)}), ratio {np.median(hi)/np.median(lo):.2f}')
