"""What are these reactions, chemically?

The chapter names reaction ids throughout but never says what happens in them.
Formula comes from the endpoint xyz; the changing bonds come from the same
rule stage 3 uses -- the pairs with the largest |d_product - d_reactant|,
restricted to pairs bonded on at least one side.
"""
import json, os, glob
from collections import Counter
import numpy as np

H = '/home/energy/s242862'
MR = ('rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn6196 rxn0346 rxn4518 '
      'rxn3107 rxn8837 rxn7060 rxn5691 rxn1283 rxn8827 rxn4522 rxn1147 '
      'rxn0894 rxn7957 rxn5690').split()
COV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66}


def rd(p):
    L = open(p, errors='replace').read().splitlines()
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for l in L[2:2 + n]:
        f = l.split()
        if len(f) >= 4:
            sym.append(f[0])
            xyz.append([float(v) for v in f[1:4]])
    return sym, np.array(xyz)


def formula(sym):
    c = Counter(sym)
    out = ''
    for el in ('C', 'H', 'N', 'O'):
        if c.get(el):
            out += el + (str(c[el]) if c[el] > 1 else '')
    return out


def bonded(sym, x, i, j, slack=1.3):
    d = np.linalg.norm(x[i] - x[j])
    return d < slack * (COV.get(sym[i], 0.7) + COV.get(sym[j], 0.7))


def changes(rx):
    pr = f'{H}/orca_neb_results/{rx}/reactant.xyz'
    pp = f'{H}/orca_neb_results/{rx}/product.xyz'
    if not (os.path.exists(pr) and os.path.exists(pp)):
        return None
    s, a = rd(pr)
    _, b = rd(pp)
    n = len(s)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            da = np.linalg.norm(a[i] - a[j])
            db = np.linalg.norm(b[i] - b[j])
            if not (bonded(s, a, i, j) or bonded(s, b, i, j)):
                continue
            cand.append((abs(db - da), i, j, da, db))
    cand.sort(reverse=True)
    return s, cand[:2]


import json as _j
grp = {}
for p in sorted(glob.glob(H + '/stab_pipeline/rxn*/result.json')):
    rxx = os.path.basename(os.path.dirname(p))
    try:
        g = {x['source']: x for x in _j.load(open(p))['geometries']}.get('RKS-ref')
    except Exception:
        continue
    if g and g.get('ext_stable') is not None:
        grp[rxx] = 'MR' if g['ext_stable'] is False else 'Kontrolle'
fc = {'MR': Counter(), 'Kontrolle': Counter()}
for rxx, k in sorted(grp.items()):
    c = changes(rxx)
    if c is None:
        continue
    fc[k][formula(c[0])] += 1
print('Summenformeln je Gruppe')
for k in ('MR', 'Kontrolle'):
    print('  %-10s %s' % (k, dict(fc[k].most_common())))
print()
print('%-9s %-8s %-5s  %s' % ('rxn', 'Formel', 'Atome', 'was sich aendert'))
print('-' * 78)
for rx in MR:
    c = changes(rx)
    if c is None:
        print('%-9s  Endpunkte fehlen' % rx); continue
    s, top = c
    desc = []
    for d, i, j, da, db in top:
        lab = '%s%d-%s%d' % (s[i], i + 1, s[j], j + 1)
        verb = 'bricht ' if db > da else 'bildet '
        desc.append('%s %s %.2f->%.2f' % (lab, verb, da, db))
    print('%-9s %-8s %-5d  %s' % (rx, formula(s), len(s), ' | '.join(desc)))
