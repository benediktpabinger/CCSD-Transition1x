"""DFT test at found structures, read as fixed in D24 (and D26 for other models).

For each run: largest force norm on any atom on the lowest UKS solution, model force
error at the same structure, DFT and model barrier, class at the found structure.
Output results/d24_forces_<model>.csv

    python pipeline/d24_table.py esen ~/orca6_d24
"""
import csv
import json
import os
import re
import sys
from math import comb

import numpy as np
from ase.io import read

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
HA, EVA = 27.211386245988, 51.42208
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')
GAP, FMAX = 0.043, 0.05
model = sys.argv[1]
orca = os.path.expanduser(sys.argv[2])


def energy(p):
    if not os.path.exists(p):
        return None
    e = E_RE.findall(open(p, errors='replace').read())
    return float(e[-1]) * HA if e else None


def forces(p):
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    if 'CARTESIAN GRADIENT' not in t:
        return None
    G = []
    for line in t[t.find('CARTESIAN GRADIENT'):].split('\n')[3:]:
        f = line.split()
        if len(f) < 6:
            break
        try:
            G.append([float(v) for v in f[3:6]])
        except ValueError:
            break
    return -np.array(G) * EVA


def norm(F):
    return float(np.sqrt((F ** 2).sum(1)).max())


rows, missing = {}, []
for r in csv.DictReader(open(f'{RES}/d24_set.csv')):
    rx = r['rxn']
    d = f"{H}/{r['root']}/t1x/{model}/{rx}"
    if not os.path.exists(f'{d}/result.json'):
        continue
    res = json.load(open(f'{d}/result.json'))
    F = forces(f'{orca}/{rx}_ts/uks_engrad.out')
    if F is None:
        missing.append((rx, r['group'], int(bool(res['criterion_met']))))
        continue
    Fm = read(f'{d}/transition_state.xyz').get_forces()
    et, er = energy(f'{orca}/{rx}_ts/uks_engrad.out'), energy(f'{orca}/{rx}_r/uks_engrad.out')
    ek = energy(f'{orca}/{rx}_ts/rks_sp.out')
    s = S_RE.findall(open(f'{orca}/{rx}_ts/uks_sp.out', errors='replace').read())
    rows[rx] = dict(rxn=rx, group=r['group'], bin=r['bin'], partner=r['partner'], nfod=float(r['nfod']),
                    met=int(bool(res['criterion_met'])), f_dft=norm(F), f_model=norm(Fm), f_err_max=norm(F - Fm),
                    f_err_mae=float(np.abs(F - Fm).mean()), s2_found=float(s[-1]) if s else 0.0,
                    depth_found_mev=(ek - et) * 1e3 if ek and et else np.nan,
                    barrier_dft=et - er if et and er else np.nan, barrier_model=res.get('barrier_ev', np.nan))
R = list(rows.values())
print(len(R), 'structures with DFT,', sum(r['met'] for r in R), 'converged,', model)
print('missing (no DFT gradient):', len(missing), missing)
with open(f'{RES}/d24_forces_{model}_missing.csv', 'w') as fh:
    fh.write('rxn,group,converged\n' + ''.join('%s,%s,%d\n' % m for m in missing))


def line(name, S):
    if not S:
        return
    x = np.array([r['f_dft'] for r in S])
    e = np.array([r['f_err_max'] for r in S])
    m = np.array([r['f_err_mae'] for r in S])
    b = np.array([abs(r['barrier_model'] - r['barrier_dft']) for r in S])
    b = b[np.isfinite(b)]
    print('%-16s n=%3d  F_DFT median %.3f  >0.05 %3.0f %%  >0.10 %3.0f %%  err max %.3f mae %.4f  barrier err median %s, >43 meV %s'
          % (name, len(S), np.median(x), 100 * (x > FMAX).mean(), 100 * (x > 0.10).mean(), np.median(e), np.median(m),
             '%.0f meV' % (np.median(b) * 1e3) if len(b) else '-', '%.0f %%' % (100 * (b > GAP).mean()) if len(b) else '-'))


C = [r for r in R if r['met']]
for g in ('baseline', 'closed', 'bs'):
    line(g, [r for r in C if r['group'] == g])
print('by depth bin')
for b in ('1-10', '10-43', '43-300', '>300'):
    for g in ('closed', 'bs'):
        line(f'  {b} {g}', [r for r in C if r['group'] == g and r['bin'] == b])
print('failed runs, top image')
for g in ('baseline', 'closed', 'bs'):
    S = [r for r in R if r['group'] == g and not r['met']]
    if S:
        print('  %-9s n=%3d  err max %.3f mae %.4f' % (g, len(S), np.median([r['f_err_max'] for r in S]), np.median([r['f_err_mae'] for r in S])))
P = [(r, rows[r['partner']]) for r in C if r['group'] == 'bs' and r['partner'] in rows and rows[r['partner']]['met']]


def mc(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


b = sum(a['f_dft'] > FMAX >= t['f_dft'] for a, t in P)
c = sum(t['f_dft'] > FMAX >= a['f_dft'] for a, t in P)
print('D24 main test: %d pairs, above 0.05 BS %d, twin %d, only BS %d, only twin %d, exact McNemar p = %.3f'
      % (len(P), sum(a['f_dft'] > FMAX for a, t in P), sum(t['f_dft'] > FMAX for a, t in P), b, c, mc(b, c)))
if P:
    q = [a['f_dft'] / t['f_dft'] for a, t in P]
    print('paired ratio F_DFT BS/twin, median %.2f, BS larger in %d of %d' % (np.median(q), sum(v > 1 for v in q), len(q)))
with open(f'{RES}/d24_forces_{model}.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(R[0].keys()))
    w.writeheader()
    for r in R:
        w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
