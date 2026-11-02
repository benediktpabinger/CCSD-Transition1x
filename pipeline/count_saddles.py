"""How many distinct DFT saddles has each reaction shown?

Pools every converged DFT transition-state optimisation of the project (PySCF
with geomeTRIC, ~/bs_tsopt_*, ~/tsopt_from_model, ~/tsopt_null) into one catalogue
and counts distinct saddles per reaction, closed-shell against broken-symmetry.

Rule, fixed before counting: two saddles are the same if the RMSD is <= 0.05 A
(mirror images allowed, atom order fixed) and |dE| <= 5 meV. Single linkage within
a reaction. A looser rule (0.10 A, 10 meV) shows whether the count depends on it.
Pairs within 1 meV but above the RMSD limit are flagged: they may be the same
saddle with equivalent atoms swapped, so the count is an upper limit there.

Runs marked BS_LOST or COLLAPSED are excluded: they left the broken-symmetry
solution. A reaction is closed-shell if no saddle has <S^2> > 0.05.

Limits: the runs were not designed for this. Numbers of starts differ per reaction,
closed-shell reactions have few starts, and no saddle was followed down to check
that it connects the same reactant and product.

Output results/saddle_catalogue.csv, results/saddle_counts.csv

    python pipeline/count_saddles.py
"""
import csv
import glob
import json
import os

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
HA_MEV = 27211.386245988
SETS = ['bs_tsopt_batch', 'bs_tsopt_v2', 'bs_tsopt_retry', 'bs_tsopt_umam', 'bs_tsopt_fromneb',
        'tsopt_from_model', 'tsopt_null']


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    return np.array([[float(v) for v in x.split()[1:4]] for x in L[2:2 + n]])


def rmsd(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    out = []
    for d in (1, -1):                       # proper rotation, then mirror image
        D = np.diag([1, 1, d * np.sign(np.linalg.det(U @ Vt))])
        out.append(float(np.sqrt(((A @ (U @ D @ Vt) - B) ** 2).sum(1).mean())))
    return min(out)


cat, skipped = [], {}
for s in SETS:
    for f in sorted(glob.glob(f'{H}/{s}/*/result.json')):
        r = json.load(open(f))
        d = os.path.dirname(f)
        st = str(r.get('status'))
        if not r.get('opt_converged') or st in ('BS_LOST', 'COLLAPSED'):
            skipped[st] = skipped.get(st, 0) + 1
            continue
        e = r.get('e_uks_final', r.get('e_final', r.get('e_rks_end')))
        xyz = next((p for p in (f'{d}/ts_opt.xyz', f'{d}/ts_null.xyz') if os.path.exists(p)), None)
        if e is None or xyz is None:
            skipped['no energy or structure'] = skipped.get('no energy or structure', 0) + 1
            continue
        s2 = r.get('s2_final')
        cat.append(dict(rxn=r['rxn'], source=s, start=r.get('model') or r.get('start_model') or r.get('mode') or 'rks_ref',
                        e_ha=float(e), s2=0.0 if s2 is None else float(s2), X=read_xyz(xyz)))


def clusters(S, r_max, e_max):
    n = len(S)
    lab = list(range(n))

    def find(i):
        while lab[i] != i:
            i = lab[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if abs(S[i]['e_ha'] - S[j]['e_ha']) * HA_MEV <= e_max and rmsd(S[i]['X'], S[j]['X']) <= r_max:
                lab[find(i)] = find(j)
    return [find(i) for i in range(n)]


rows, flags = [], 0
for rx in sorted({c['rxn'] for c in cat}):
    S = [c for c in cat if c['rxn'] == rx]
    a, b = clusters(S, 0.05, 5), clusters(S, 0.10, 10)
    for c, l in zip(S, a):
        c['cluster'] = l
    E = np.array([c['e_ha'] for c in S])
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if a[i] != a[j] and abs(E[i] - E[j]) * HA_MEV <= 1:
                flags += 1
    s2max = max(c['s2'] for c in S)
    rows.append(dict(rxn=rx, group='bs' if s2max > 0.05 else 'closed', n_runs=len(S),
                     n_starts=len({(c['source'], c['start']) for c in S}), n_saddles=len(set(a)),
                     n_saddles_loose=len(set(b)), span_mev=(E.max() - E.min()) * HA_MEV,
                     s2_min=min(c['s2'] for c in S), s2_max=s2max))

print('%d converged saddles, %d reactions; skipped %s; %d pairs within 1 meV counted as different'
      % (len(cat), len(rows), skipped, flags))
print('%-9s %-7s %4s %5s %8s %7s %9s  %s' % ('rxn', 'group', 'runs', 'starts', 'saddles', 'loose', 'span meV', 's2 range'))
for r in sorted(rows, key=lambda r: (r['group'], -r['n_saddles'])):
    print('%-9s %-7s %4d %5d %8d %7d %9.1f  %.2f..%.2f'
          % (r['rxn'], r['group'], r['n_runs'], r['n_starts'], r['n_saddles'], r['n_saddles_loose'],
             r['span_mev'], r['s2_min'], r['s2_max']))
for g in ('closed', 'bs'):
    S = [r for r in rows if r['group'] == g]
    M = [r for r in S if r['n_runs'] >= 2]
    print('%-7s reactions %2d, with 2+ runs %2d: of those, more than one saddle %2d (loose rule %2d)'
          % (g, len(S), len(M), sum(r['n_saddles'] > 1 for r in M), sum(r['n_saddles_loose'] > 1 for r in M)))

with open(os.path.join(REPO, 'results', 'saddle_counts.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.3f' % v if isinstance(v, float) else v) for k, v in r.items()})
with open(os.path.join(REPO, 'results', 'saddle_catalogue.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'source', 'start', 'e_ha', 's2', 'cluster'])
    for c in sorted(cat, key=lambda c: (c['rxn'], c['e_ha'])):
        w.writerow([c['rxn'], c['source'], c['start'], '%.8f' % c['e_ha'], '%.4f' % c['s2'], c['cluster']])
