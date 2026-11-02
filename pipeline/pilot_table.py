"""Twin pilot, evaluated as fixed in D22 (docs/paper_decisions.md).

For each reaction and model: the barrier from the Transition1x start and from the
interpolated start. A pair of runs counts only if both exist and both met the criterion.
    start-sensitive  at least 2 of 4 models have barrier(idpp) <= barrier(t1x) - 0.043 eV
    unresolved       fewer than 2 models have a usable pair, once all runs are in
    pending          runs are still missing and the reaction could still change
Main test: paired exact McNemar test over the 100 pairs on start-sensitive yes/no.

Output results/pilot_runs.csv, results/pilot_reactions.csv

    python pipeline/pilot_table.py
"""
import csv
import json
import os
from math import comb

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
M = ('uma-s', 'uma-m', 'esen', 'mace-omol')
GAP = 0.043


def get(start, m, rx):
    p = f'{H}/neb_pilot/{start}/{m}/{rx}/result.json'
    return json.load(open(p)) if os.path.exists(p) else None


pairs = list(csv.DictReader(open(f'{RES}/pilot_pairs.csv')))
runs, rx_rows = [], {}
for p in pairs:
    for group, rx in (('bs', p['rxn_bs']), ('closed', p['rxn_cs'])):
        n_usable, n_lower, n_higher, n_missing, gaps = 0, 0, 0, 0, []
        for m in M:
            a, b = get('t1x', m, rx), get('idpp', m, rx)
            if a is None or b is None:
                n_missing += 1
                continue
            ok = a['status'] == b['status'] == 'done' and a['criterion_met'] and b['criterion_met']
            d = b.get('barrier_ev', np.nan) - a.get('barrier_ev', np.nan)
            runs.append(dict(rxn=rx, group=group, bin=p['bin'], model=m, barrier_t1x=a.get('barrier_ev'),
                             barrier_idpp=b.get('barrier_ev'), met_t1x=int(bool(a['criterion_met'])),
                             met_idpp=int(bool(b['criterion_met'])), steps_t1x=a['n_steps'], steps_idpp=b['n_steps']))
            if ok:
                n_usable += 1
                gaps.append(d)
                n_lower += d <= -GAP
                n_higher += d >= GAP
        status = ('sensitive' if n_lower >= 2 else
                  'pending' if n_missing and n_lower + n_missing >= 2 else
                  'unresolved' if n_usable < 2 else 'not_sensitive')
        rx_rows[rx] = dict(rxn=rx, group=group, bin=p['bin'], n_usable=n_usable, n_missing=n_missing, n_lower=n_lower,
                           n_higher=n_higher, status=status, median_gap_ev=float(np.median(gaps)) if gaps else None,
                           best_gap_ev=min(gaps) if gaps else None)

print('reaction status    ', {g: {s: sum(r['group'] == g and r['status'] == s for r in rx_rows.values())
                               for s in ('sensitive', 'not_sensitive', 'unresolved', 'pending')} for g in ('bs', 'closed')})
for b in ('10-43', '43-300', '>300'):
    print(' bin %-7s' % b, {g: {s: sum(r['group'] == g and r['bin'] == b and r['status'] == s for r in rx_rows.values())
                                for s in ('sensitive', 'not_sensitive', 'unresolved', 'pending')} for g in ('bs', 'closed')})
done = [p for p in pairs if {rx_rows[p['rxn_bs']]['status'], rx_rows[p['rxn_cs']]['status']} <= {'sensitive', 'not_sensitive'}]
b = sum(rx_rows[p['rxn_bs']]['status'] == 'sensitive' and rx_rows[p['rxn_cs']]['status'] != 'sensitive' for p in done)
c = sum(rx_rows[p['rxn_bs']]['status'] != 'sensitive' and rx_rows[p['rxn_cs']]['status'] == 'sensitive' for p in done)
n = b + c
pv = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
s_bs = sum(rx_rows[p['rxn_bs']]['status'] == 'sensitive' for p in done)
s_cs = sum(rx_rows[p['rxn_cs']]['status'] == 'sensitive' for p in done)
print('complete pairs %d: start-sensitive BS %d, closed-shell %d; only BS %d, only closed %d; exact McNemar p = %.4f'
      % (len(done), s_bs, s_cs, b, c, pv))
R = [r for r in runs if r['met_t1x'] and r['met_idpp']]
for g in ('bs', 'closed'):
    S = [r for r in runs if r['group'] == g]
    T = [r for r in R if r['group'] == g]
    d = np.array([r['barrier_idpp'] - r['barrier_t1x'] for r in T])
    print('%-7s run pairs %4d, both converged %4d: lower by >43 meV %3d, higher %3d, largest drop %.2f eV' % (g, len(S), len(T), (d <= -GAP).sum(), (d >= GAP).sum(), d.min()))
    print('        converged share t1x %.0f %%  idpp %.0f %%' % (100 * np.mean([r['met_t1x'] for r in S]), 100 * np.mean([r['met_idpp'] for r in S])))
for name, rows in (('pilot_runs.csv', runs), ('pilot_reactions.csv', list(rx_rows.values()))):
    with open(f'{RES}/{name}', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
