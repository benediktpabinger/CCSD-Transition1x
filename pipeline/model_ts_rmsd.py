"""Wie weit liegen die drei Modell-TS einer Reaktion auseinander?

Paarweise Kabsch-RMSD zwischen den Uebergangszustaenden, die UMA-S, UMA-M und
eSEN fuer dieselbe Reaktion vorhergesagt haben. Rein geometrisch, keine
Energie. Gegenstueck zur Spannweite der DFT-Barriere ueber dieselben drei
Punkte.

results/model_ts_rmsd.csv
"""
import csv
import itertools
import os

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
MD = {'uma-s': 'uma_neb_results', 'uma-m': 'uma_m_neb_results',
      'esen': 'esen_neb_results'}


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0])
        xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B):
    if len(A) != len(B):
        return float('nan')
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


rxns = sorted({r['rxn'] for r in
               csv.DictReader(open(f'{OUT}/omol25_model_geoms.csv'))})
rows = []
for rx in rxns:
    G, ok = {}, True
    for m, dn in MD.items():
        p = f'{H}/{dn}/{rx}/transition_state.xyz'
        if not os.path.exists(p):
            ok = False
            break
        G[m] = read_xyz(p)
    if not ok:
        continue
    syms = [tuple(G[m][0]) for m in MD]
    if len(set(syms)) != 1:
        print('%s: Atomreihenfolge unterscheidet sich zwischen den Modellen'
              % rx)
        continue
    d = {'%s|%s' % (a, b): kabsch(G[a][1], G[b][1])
         for a, b in itertools.combinations(MD, 2)}
    rows.append({'rxn': rx,
                 'rmsd_max': max(d.values()),
                 'rmsd_med': float(np.median(list(d.values()))),
                 **{'rmsd_' + k.replace('|', '_'): v for k, v in d.items()}})

COLS = ['rxn', 'rmsd_max', 'rmsd_med', 'rmsd_uma-s_uma-m',
        'rmsd_uma-s_esen', 'rmsd_uma-m_esen']
with open(f'{OUT}/model_ts_rmsd.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in rows:
        w.writerow([r['rxn']] + ['%.5f' % r[c] for c in COLS[1:]])

v = np.array([r['rmsd_max'] for r in rows])
print('%d Reaktionen' % len(rows))
print('groesste paarweise RMSD [A]: min %.4f  median %.4f  p90 %.4f  max %.4f'
      % (v.min(), np.median(v), np.percentile(v, 90), v.max()))
print('geschrieben: results/model_ts_rmsd.csv')
