"""How far from stationary is the model's predicted transition state?

Every other measure in this project compares a model geometry against a
structure we optimised, so it inherits whatever doubt attaches to that
structure. The gradient does not: it asks only whether a force still acts at
the predicted geometry. A transition state has none. This therefore says
something about model quality without any reference at all, and survives even
if every broken-symmetry saddle we found turned out to be wrong.

The gradient is taken on whichever surface is the ground state there, decided
by the stability analysis: RKS where the restricted solution is externally
stable, the broken-symmetry solution where it is not. Taking the RKS gradient
in a region where RKS is not the ground state would measure the wrong thing.

Scale: our optimised saddles come out at 0.001-0.015 eV/A. The ORCA reference
geometries recompute in PySCF at 0.013-0.18 eV/A, which is the spread between
the two codes and the practical floor here.
"""
import json
import os

import numpy as np

H = '/home/energy/s242862'
MODELS = ['UMA-S', 'UMA-M', 'eSEN']

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
    ref_unstable = ref['ext_stable'] is False

    def ground_grad(g):
        """Gradient on whichever surface is the ground state at this geometry."""
        if g is None or g.get('ext_stable') is None:
            return None, None
        if g['ext_stable']:
            v = (g.get('rks_grad') or {}).get('max_evang')
            return v, 'RKS'
        v = ((g.get('bs') or {}).get('bs_grad') or {}).get('max_evang')
        if v is None:
            return None, None
        return v, 'BS'

    gref, sref = ground_grad(ref)
    for m in MODELS:
        v, s = ground_grad(geo.get(m))
        if v is None:
            continue
        rows.append({'rxn': rx, 'grp': grp[rx], 'nfod': nf[rx], 'model': m,
                     'grad': v, 'surface': s,
                     'class': 'MR' if ref_unstable else 'einfach',
                     'ref_grad': gref})

print(f'{len(rows)} Modellgeometrien\n')


def stats(v):
    v = np.asarray(v)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return len(v), q2, v.mean(), q1, q3, v.max()


print(f"{'Klasse':<12}{'n':>5}{'median':>9}{'mean':>9}{'Q1':>9}{'Q3':>9}"
      f"{'max':>9}{'>0.3':>7}")
print('-' * 69)
for cls in ('einfach', 'MR'):
    v = [r['grad'] for r in rows if r['class'] == cls]
    if not v:
        continue
    n_, med, mean, q1, q3, mx = stats(v)
    print(f'{cls:<12}{n_:>5}{med:>9.4f}{mean:>9.4f}{q1:>9.4f}{q3:>9.4f}'
          f'{mx:>9.4f}{int((np.array(v) > 0.3).sum()):>7}')

print(f'\n{"Modell":<12}{"einfach: median":>17}{"MR: median":>13}{"Faktor":>9}')
for m in MODELS:
    a = [r['grad'] for r in rows if r['model'] == m and r['class'] == 'einfach']
    b = [r['grad'] for r in rows if r['model'] == m and r['class'] == 'MR']
    if not (a and b):
        continue
    ma, mb = float(np.median(a)), float(np.median(b))
    print(f'{m:<12}{ma:>17.4f}{mb:>13.4f}{mb/ma:>9.1f}')

print('\n=== zum Vergleich: die Referenzgeometrie selbst ===')
for cls in ('einfach', 'MR'):
    v = [r['ref_grad'] for r in rows if r['class'] == cls
         and r['ref_grad'] is not None]
    v = sorted(set(round(x, 6) for x in v))
    if v:
        print(f'  {cls:<9} n={len(v):>3}  median {np.median(v):.4f}  '
              f'min {min(v):.4f}  max {max(v):.4f}')

print('\n=== auf welcher Flaeche gemessen ===')
for cls in ('einfach', 'MR'):
    c = {}
    for r in rows:
        if r['class'] == cls:
            c[r['surface']] = c.get(r['surface'], 0) + 1
    print(f'  {cls:<9} ' + '  '.join(f'{k}: {v}' for k, v in sorted(c.items())))

print('\n=== groesste Abweichungen in der MR-Gruppe ===')
for r in sorted([r for r in rows if r['class'] == 'MR'],
                key=lambda x: -x['grad'])[:8]:
    print(f"  {r['rxn']:<10}{r['model']:<8}{r['surface']:<4}{r['grad']:>8.3f}")
