"""How far apart are the three OMol25 models from each other?

Every other comparison here needs a reference whose own correctness is under
discussion. This one does not: it asks only how much UMA-S, UMA-M and eSEN
disagree among themselves. Agreement does not make them right -- three models
trained on the same data can share a bias -- but disagreement is decisive
evidence that at least two of them are wrong, and it can be read without any
reference at all.

Three quantities per reaction, each the largest of the three pairwise values:
geometric spread (Kabsch RMSD), spread in the reactive coordinate, and spread in
energy. The energies are ground-state energies at each model's own geometry, so
the spread mixes geometric disagreement with energetic -- which is the point,
since a barrier calculation would inherit exactly that.
"""
import itertools
import json
import os

import numpy as np

H = '/home/energy/s242862'
HA_MEV = 27211.386
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results'}

res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
TOP = [res[i]['rxn'] for i in range(26)]
MID = [res[i - 1]['rxn'] for i in [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
LOW = [res[i]['rxn'] for i in range(n - 10, n)]
grp = {}
for r in TOP: grp[r] = 'high'
for r in MID: grp.setdefault(r, 'mid')
for r in LOW: grp.setdefault(r, 'low')
nf = {x['rxn']: x['nfod'] for x in res}


def read_xyz(p):
    L = open(p).read().split('\n')
    m = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + m]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B):
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def reactive(rx, sym):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1]) for e in rb[:2]]
    r, p = (f'{H}/orca_neb_results/{rx}/reactant.xyz',
            f'{H}/orca_neb_results/{rx}/product.xyz')
    if not (os.path.exists(r) and os.path.exists(p)):
        return []
    from ase.data import atomic_numbers, covalent_radii
    _, xr = read_xyz(r)
    _, xp = read_xyz(p)
    cand = []
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            dr = float(np.linalg.norm(xr[i] - xr[j]))
            dp = float(np.linalg.norm(xp[i] - xp[j]))
            rc = 1.3 * (covalent_radii[atomic_numbers[sym[i]]]
                        + covalent_radii[atomic_numbers[sym[j]]])
            if min(dr, dp) < rc:
                cand.append((abs(dp - dr), i, j))
    cand.sort(reverse=True)
    return [(i, j) for _, i, j in cand[:2]]


rows = []
for rx in sorted(grp, key=lambda r: -nf[r]):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    geo = {g['source']: g for g in d['geometries']}
    ref = geo.get('RKS-ref')
    if not ref or ref.get('ext_stable') is None:
        continue
    cls = 'MR' if ref['ext_stable'] is False else 'einfach'

    xs, es = {}, {}
    for m, dn in MODELS.items():
        f = f'{H}/{dn}/{rx}/transition_state.xyz'
        if os.path.exists(f):
            sym, x = read_xyz(f)
            xs[m] = (sym, x)
        g = geo.get(m)
        if g and g.get('ext_stable') is not None:
            if g['ext_stable']:
                es[m] = g.get('e_rks')
            else:
                es[m] = (g.get('bs') or {}).get('e_uks')
    if len(xs) < 2:
        continue
    sym = list(xs.values())[0][0]
    pairs = reactive(rx, sym)

    d_rmsd, d_rc = [], []
    for a, b in itertools.combinations(xs, 2):
        if xs[a][0] != xs[b][0]:
            continue
        d_rmsd.append(kabsch(xs[a][1], xs[b][1]))
        if pairs:
            d_rc.append(max(
                abs(float(np.linalg.norm(xs[a][1][i] - xs[a][1][j]))
                    - float(np.linalg.norm(xs[b][1][i] - xs[b][1][j])))
                for i, j in pairs))
    ev = [v for v in es.values() if v is not None]
    d_e = (max(ev) - min(ev)) * HA_MEV if len(ev) > 1 else None
    rows.append({'rxn': rx, 'cls': cls, 'nfod': nf[rx],
                 'rmsd': max(d_rmsd) if d_rmsd else None,
                 'rc': max(d_rc) if d_rc else None,
                 'de': d_e, 'n': len(xs)})

print(f'{len(rows)} Reaktionen\n')
print('Groesste paarweise Abweichung zwischen UMA-S, UMA-M und eSEN.\n')
print(f"{'Klasse':<10}{'n':>4}{'RMSD med':>10}{'RMSD max':>10}"
      f"{'RC med':>9}{'RC max':>9}{'dE med':>10}{'dE max':>11}")
print('-' * 73)
for cls in ('einfach', 'MR'):
    s = [r for r in rows if r['cls'] == cls]
    if not s:
        continue
    a = np.array([r['rmsd'] for r in s if r['rmsd'] is not None])
    b = np.array([r['rc'] for r in s if r['rc'] is not None])
    c = np.array([r['de'] for r in s if r['de'] is not None])
    print(f'{cls:<10}{len(s):>4}{np.median(a):>10.4f}{a.max():>10.4f}'
          f'{np.median(b):>9.4f}{b.max():>9.4f}'
          f'{np.median(c):>10.1f}{c.max():>11.1f}')

print('\n=== die 19 Multireferenz-Reaktionen einzeln ===')
print(f"{'rxn':<10}{'N_FOD':>7}{'RMSD':>9}{'RC':>9}{'dE [meV]':>11}")
for r in sorted([r for r in rows if r['cls'] == 'MR'], key=lambda x: -(x['de'] or 0)):
    print(f"{r['rxn']:<10}{r['nfod']:>7.3f}"
          f"{('—' if r['rmsd'] is None else f'{r[chr(114)+chr(109)+chr(115)+chr(100)]:.4f}'):>9}"
          f"{('—' if r['rc'] is None else f'{r[chr(114)+chr(99)]:.4f}'):>9}"
          f"{('—' if r['de'] is None else f'{r[chr(100)+chr(101)]:.1f}'):>11}")

print('\n=== wie viele MR-Reaktionen ueberschreiten welche Schwelle ===')
mr = [r for r in rows if r['cls'] == 'MR']
ez = [r for r in rows if r['cls'] == 'einfach']
for label, thr, key in (('RMSD > 0.1 A', 0.1, 'rmsd'),
                        ('RC   > 0.1 A', 0.1, 'rc'),
                        ('dE   > 50 meV', 50, 'de'),
                        ('dE   > 500 meV', 500, 'de')):
    a = sum(1 for r in ez if r[key] is not None and r[key] > thr)
    b = sum(1 for r in mr if r[key] is not None and r[key] > thr)
    print(f'  {label:<16} einfach {a:>2}/{len(ez)}   MR {b:>2}/{len(mr)}')
