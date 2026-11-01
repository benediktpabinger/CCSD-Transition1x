"""Force error of the low-MR twin pairs, split by breaking depth.

Each low-MR broken-symmetry reaction (results/twins_selection.csv, N_FOD < 0.5)
is compared with its closed-shell twin of equal N_FOD at the stored Transition1x
transition state, f_mae from results/model_sp_ts_all.csv against the OMol25 label.
The depth E_RKS - E_BS comes from our ORCA 6.0.1 runs in ~/orca6_s2pos/<rxn>/.

Thesis prediction (Section "Explanation, and what remains open"): if the shape of
the surface is the cause, the error is largest close to the onset of symmetry
breaking and fades with growing depth.

Output results/twins_by_depth.csv: model, depth bin, n_pairs, medians, ratio of
the medians with a 95 % bootstrap interval over pairs (each pair is one reaction
pair, so this is resampling over reactions), share of pairs with BS worse.

    python pipeline/twins_by_depth.py
"""
import csv
import os
import re

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
HA_MEV = 27211.386245988
BINS = [('<1', -1e9, 1), ('1-10', 1, 10), ('10-43', 10, 43), ('43-100', 43, 100), ('>100', 100, 1e9)]
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')


def energy(p):
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return None
    return float(E_RE.findall(t)[-1])


tw = {r['rxn']: r['partner_bs'] for r in csv.DictReader(open(f'{RES}/twins_selection.csv'))}
depth = {}
for b in set(tw.values()):
    e_r, e_u = energy(f'{H}/orca6_s2pos/{b}/rks_sp.out'), energy(f'{H}/orca6_s2pos/{b}/uks_engrad.out')
    if e_r is not None and e_u is not None:
        depth[b] = (e_r - e_u) * HA_MEV
print(f'{len(depth)} of {len(set(tw.values()))} low-MR broken-symmetry reactions have a depth')

rows = list(csv.DictReader(open(f'{RES}/model_sp_ts_all.csv')))
rng = np.random.default_rng(0)
out = []
for m in ('uma-s', 'uma-m', 'esen', 'mace-omol'):
    f = {r['rxn']: float(r['f_mae']) for r in rows if r['model'] == m}
    for name, lo, hi in BINS:
        x = np.array([(f[b], f[c]) for c, b in tw.items()
                      if b in depth and lo <= depth[b] < hi and b in f and c in f])
        idx = rng.integers(0, len(x), (5000, len(x)))
        boot = np.median(x[idx, 0], axis=1) / np.median(x[idx, 1], axis=1)
        out.append(dict(model=m, depth_mev=name, n_pairs=len(x),
                        f_mae_bs=np.median(x[:, 0]), f_mae_cs=np.median(x[:, 1]),
                        ratio=np.median(x[:, 0]) / np.median(x[:, 1]),
                        ratio_lo=np.percentile(boot, 2.5), ratio_hi=np.percentile(boot, 97.5),
                        pct_bs_worse=100 * np.mean(x[:, 0] > x[:, 1])))
        r = out[-1]
        print('%-10s %-7s n=%3d  bs %.4f cs %.4f  ratio %.2f [%.2f, %.2f]  bs worse %2.0f %%'
              % (m, name, r['n_pairs'], r['f_mae_bs'], r['f_mae_cs'], r['ratio'],
                 r['ratio_lo'], r['ratio_hi'], r['pct_bs_worse']))

with open(f'{RES}/twins_by_depth.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    for r in out:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
print('results/twins_by_depth.csv')
