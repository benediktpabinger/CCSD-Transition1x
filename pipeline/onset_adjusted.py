"""D38: is the onset pattern of the force error a side effect of molecule size or bond stretch?

(1) Twin pairs (results/twins_selection.csv, low MR, equal N_FOD): per model, the log ratio
    f_mae(BS) / f_mae(closed-shell twin) at the stored transition state is regressed on the
    breaking-depth bins of twins_by_depth.py and on the pair differences in log atom count and
    in stretch. Adjusted ratio per bin = exp(bin coefficient) at zero difference.
(2) Every stored TS with our own stability analysis (stab_*_orca6.csv): per model, log f_mae on
    N_FOD bins, log atom count, stretch and class/depth bin (closed-shell is the reference).

Stretch as in mr_vs_stretch.py: largest d / (r_i + r_j) at the stored TS over the atom pairs
whose bond changes between the Transition1x reactant and product. Force error f_mae against
the OMol25 label (results/model_sp_ts_all.csv). Intervals: 2000 bootstraps over pairs (1) or
reactions (2). Reading fixed in D38.

Output results/onset_adjusted.csv

    python pipeline/onset_adjusted.py       (reads the 6.5 GB dataset, run as a short job)
"""
import csv
import json
import os
import sys

import h5py
import numpy as np
from ase.data import covalent_radii

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_check import bonds  # noqa: E402  (same bond graph and H-H rule as mr_vs_stretch.py)

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
M = ('uma-s', 'uma-m', 'esen', 'mace-omol')
DBINS = [('<1', -1e9, 1), ('1-10', 1, 10), ('10-43', 10, 43), ('43-100', 43, 100), ('100-300', 100, 300), ('>300', 300, 1e9)]
TWIN_BINS = DBINS[:4] + [('>100', 100, 1e9)]          # the bins of twins_by_depth.py
NBINS = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.), (1., 1.2), (1.2, 9)]

stab = {}
for s in ('s2pos', 'cshigh', 'twins', 'base'):
    for r in csv.DictReader(open(f'{RES}/stab_{s}_orca6.csv')):
        stab[r['rxn']] = (r['group_ours'], float(r['depth_mev']))
tw = {r['rxn']: r['partner_bs'] for r in csv.DictReader(open(f'{RES}/twins_selection.csv'))}
fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
need = set(stab) | set(tw) | set(tw.values())

ts = json.load(open(f'{H}/t1x_ts_all.json'))
geo = {}
with h5py.File(f'{H}/data/Transition1x.h5', 'r') as f:
    for split in f:
        for formula in f[split]:
            for rx in f[split][formula]:
                if rx not in need or rx in geo or rx not in ts:
                    continue
                g = f[split][formula][rx]
                Z = g['atomic_numbers'][:]
                x = np.array(ts[rx]['positions'])
                pairs = bonds(Z, g['positions'][0]) ^ bonds(Z, g['positions'][9])
                st = max(np.linalg.norm(x[i] - x[j]) / (covalent_radii[Z[i]] + covalent_radii[Z[j]]) for i, j in pairs) if pairs else np.nan
                geo[rx] = (len(Z), st)
print(len(geo), 'of', len(need), 'reactions with geometry', flush=True)

F = {}
for r in csv.DictReader(open(f'{RES}/model_sp_ts_all.csv')):
    if r['f_mae']:
        F[(r['model'], r['rxn'])] = float(r['f_mae'])
rng = np.random.default_rng(0)
out = []


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


# (1) twin pairs ---------------------------------------------------------------------------
for m in M:
    P = []
    for c, b in tw.items():
        if b in stab and (m, b) in F and (m, c) in F and b in geo and c in geo and np.isfinite(geo[b][1]) and np.isfinite(geo[c][1]):
            d = stab[b][1]
            k = next(i for i, (_, lo, hi) in enumerate(TWIN_BINS) if lo <= d < hi)
            P.append((k, np.log(F[(m, b)] / F[(m, c)]), np.log(geo[b][0] / geo[c][0]), geo[b][1] - geo[c][1]))
    P = np.array(P)
    X = np.column_stack([(P[:, 0] == k).astype(float) for k in range(len(TWIN_BINS))] + [P[:, 2], P[:, 3]])
    beta = fit(X, P[:, 1])
    boot = np.array([fit(X[i], P[i, 1]) for i in (rng.integers(0, len(P), len(P)) for _ in range(2000))])
    for k, (name, _, _) in enumerate(TWIN_BINS):
        raw = np.exp(np.median(P[P[:, 0] == k, 1]))
        out.append(dict(analysis='twins', model=m, term=f'depth {name} meV', n=int((P[:, 0] == k).sum()),
                        ratio=np.exp(beta[k]), lo=np.exp(np.percentile(boot[:, k], 2.5)),
                        hi=np.exp(np.percentile(boot[:, k], 97.5)), unadjusted_median_ratio=raw))
    out.append(dict(analysis='twins', model=m, term='per 10 % more atoms', n=len(P), ratio=np.exp(beta[-2] * np.log(1.1)),
                    lo=np.exp(np.percentile(boot[:, -2], 2.5) * np.log(1.1)), hi=np.exp(np.percentile(boot[:, -2], 97.5) * np.log(1.1)),
                    unadjusted_median_ratio=np.nan))
    out.append(dict(analysis='twins', model=m, term='per 0.1 stretch', n=len(P), ratio=np.exp(beta[-1] * 0.1),
                    lo=np.exp(np.percentile(boot[:, -1], 2.5) * 0.1), hi=np.exp(np.percentile(boot[:, -1], 97.5) * 0.1),
                    unadjusted_median_ratio=np.nan))
    lo1 = [o for o in out if o['analysis'] == 'twins' and o['model'] == m and o['term'] == 'depth <1 meV'][0]
    hi1 = [o for o in out if o['analysis'] == 'twins' and o['model'] == m and o['term'] == 'depth >100 meV'][0]
    print(f'twins {m:10s} n={len(P)}  adjusted <1 meV {lo1["ratio"]:.2f} [{lo1["lo"]:.2f}, {lo1["hi"]:.2f}] (raw {lo1["unadjusted_median_ratio"]:.2f})'
          f'  >100 meV {hi1["ratio"]:.2f} [{hi1["lo"]:.2f}, {hi1["hi"]:.2f}] (raw {hi1["unadjusted_median_ratio"]:.2f})', flush=True)

held = sum(1 for m in M if (lambda a, b: a['lo'] > 1 and a['ratio'] > b['ratio'])(
    *[o for o in out if o['analysis'] == 'twins' and o['model'] == m and o['term'] in ('depth <1 meV',)],
    *[o for o in out if o['analysis'] == 'twins' and o['model'] == m and o['term'] in ('depth >100 meV',)]))
print(f'onset pattern after adjustment holds on {held} of 4 models -> D38 reading: {"holds" if held >= 3 else "does not hold"}', flush=True)

# (2) all TS with our stability analysis ----------------------------------------------------
TERMS = [f'N_FOD {lo}-{hi}' for lo, hi in NBINS[1:]] + ['log atoms', 'stretch'] + [f'BS {n} meV' for n, _, _ in DBINS]
for m in M:
    rows = []
    for rx, (cls, d) in stab.items():
        if (m, rx) in F and rx in geo and rx in fod and np.isfinite(geo[rx][1]):
            nb = [float(lo <= fod[rx] < hi) for lo, hi in NBINS[1:]]
            db = [float(cls == 'bs' and lo <= d < hi) for _, lo, hi in DBINS]
            rows.append([1.0] + nb + [np.log(geo[rx][0]), geo[rx][1]] + db + [np.log(F[(m, rx)])])
    A = np.array(rows)
    X, y = A[:, :-1], A[:, -1]
    beta = fit(X, y)
    boot = np.array([fit(X[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))])
    for t, name in enumerate(TERMS, start=1):
        scale = 0.1 if name == 'stretch' else np.log(1.1) if name == 'log atoms' else 1.0
        n_t = int(X[:, t].sum()) if name not in ('log atoms', 'stretch') else len(y)
        out.append(dict(analysis='all', model=m, term=name if name not in ('log atoms', 'stretch') else
                        ('per 10 % more atoms' if name == 'log atoms' else 'per 0.1 stretch'), n=n_t,
                        ratio=np.exp(beta[t] * scale), lo=np.exp(np.percentile(boot[:, t], 2.5) * scale),
                        hi=np.exp(np.percentile(boot[:, t], 97.5) * scale), unadjusted_median_ratio=np.nan))
    s = [o for o in out if o['analysis'] == 'all' and o['model'] == m]
    print(f'all   {m:10s} n={len(y)}  ' + '  '.join(f'{o["term"].replace(" meV", "")} {o["ratio"]:.2f}' for o in s if o['term'].startswith('BS')), flush=True)

with open(f'{RES}/onset_adjusted.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    for r in out:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
print('results/onset_adjusted.csv')
