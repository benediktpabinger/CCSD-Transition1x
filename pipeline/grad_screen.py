"""Sort every model geometry by how stationary it is, and let that decide the
Hessian list.

The gradient at each of the 134 model geometries was computed by the stability
pipeline and has been sitting there unused for this purpose. It answers the
first half of "does the model sit on a saddle at all" for free:

  large gradient   the point is not stationary, so it is not a saddle. Done.
                   A Hessian there is meaningless -- n_imag at a non-stationary
                   point says nothing about transition states.
  small gradient   a stationary point is at or very near this geometry, and only
                   a Hessian says whether it is a saddle.

Both outcomes count towards the statistic. Only the second needs computing, so
the screen defines the work list instead of a guess about it.

Measured on whichever surface is the ground state at that geometry: BS where
the restricted solution is externally unstable, RKS where it is stable.
"""
import json
import os

import numpy as np

H = '/home/energy/s242862'
MODELS = ('UMA-S', 'UMA-M', 'eSEN')
# a confirmed saddle of ours sits at 0.006-0.011 eV/A in ORCA
BUCKETS = [(0.05, 'stationary'), (0.15, 'near'), (0.30, 'off'),
           (99.0, 'far off')]


def bucket(g):
    for lim, name in BUCKETS:
        if g < lim:
            return name
    return 'far off'


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


def has_freq(rx, m):
    return os.path.exists(f'{H}/freq_at_model/{rx}_{m}/result.json')


rows = []
for rx in grp:
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    geo = {g['source']: g for g in d['geometries']}
    ref = geo.get('RKS-ref')
    if not ref or ref.get('ext_stable') is None:
        continue
    cls = 'MR' if ref['ext_stable'] is False else 'simple'
    for m in MODELS:
        g = geo.get(m)
        if not g or g.get('ext_stable') is None:
            continue
        if g['ext_stable']:
            gr = (g.get('rks_grad') or {}).get('max_evang')
            surf = 'RKS'
        else:
            gr = ((g.get('bs') or {}).get('bs_grad') or {}).get('max_evang')
            surf = 'BS'
        if gr is None:
            continue
        rows.append({'rxn': rx, 'model': m, 'cls': cls, 'grad': gr,
                     'surf': surf, 'freq': has_freq(rx, m)})

print('GRADIENT SCREEN OVER EVERY MODEL GEOMETRY')
print('=' * 78)
print('A confirmed saddle of ours measures 0.006 to 0.011 eV/A in ORCA.')
print()
print(f'{"bucket":<12}{"range eV/A":<14}{"simple":>8}{"MR":>7}   meaning')
lo = 0.0
for lim, name in BUCKETS:
    s = sum(1 for r in rows if r['cls'] == 'simple' and bucket(r['grad']) == name)
    mr = sum(1 for r in rows if r['cls'] == 'MR' and bucket(r['grad']) == name)
    meaning = {'stationary': 'a stationary point is right here',
               'near': 'one is very close; Hessian informative',
               'off': 'not stationary',
               'far off': 'not stationary, badly'}[name]
    rng = f'{lo:.2f} - {lim:.2f}' if lim < 99 else f'> {lo:.2f}'
    print(f'{name:<12}{rng:<14}{s:>8}{mr:>7}   {meaning}')
    lo = lim

for cls in ('simple', 'MR'):
    v = np.array([r['grad'] for r in rows if r['cls'] == cls])
    print(f'\n{cls}: n={len(v)}  median {np.median(v):.3f}  '
          f'Q1 {np.percentile(v, 25):.3f}  Q3 {np.percentile(v, 75):.3f}  '
          f'max {v.max():.3f}')

print('\n\nWORK LIST: geometries where a Hessian would say something')
print('=' * 78)
print('cut at 0.15 eV/A -- above that the point is not stationary and the')
print('answer is already known')
print()
todo = [r for r in rows if r['grad'] < 0.15 and not r['freq']]
done = [r for r in rows if r['grad'] < 0.15 and r['freq']]
skip = [r for r in rows if r['grad'] >= 0.15]
print(f'  already computed      {len(done):>4}')
print(f'  to compute            {len(todo):>4}   '
      f'({sum(1 for r in todo if r["cls"] == "MR")} MR, '
      f'{sum(1 for r in todo if r["cls"] == "simple")} simple)')
print(f'  not stationary, skip  {len(skip):>4}   '
      f'({sum(1 for r in skip if r["cls"] == "MR")} MR, '
      f'{sum(1 for r in skip if r["cls"] == "simple")} simple)')

print('\nMR geometries to compute, by gradient:')
for r in sorted([r for r in todo if r['cls'] == 'MR'], key=lambda x: x['grad']):
    print(f'  {r["rxn"]:<9}{r["model"]:<7}{r["grad"]:7.3f}  {r["surf"]}')

print(f'\nsimple geometries to compute: {sum(1 for r in todo if r["cls"] == "simple")}')
print('  (three models per reaction agree to 0.0045 A median there, so one per')
print('   reaction is enough as a control group)')
seen = set()
for r in sorted([r for r in todo if r['cls'] == 'simple'],
                key=lambda x: x['grad']):
    if r['rxn'] in seen:
        continue
    seen.add(r['rxn'])
    print(f'  {r["rxn"]:<9}{r["model"]:<7}{r["grad"]:7.3f}  {r["surf"]}')
print(f'  -> {len(seen)} distinct reactions')
