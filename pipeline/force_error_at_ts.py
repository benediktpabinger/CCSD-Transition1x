"""How wrong are the model forces at the geometry the model itself stopped at?

The models reproduce the ground-state energy at their own predicted transition
state to about 20 meV, yet only 7 to 13 of 19 predictions are stationary points.
Energy right, position wrong. The force is where those two meet: a NEB stops
where *its* force is small, so whatever force the reference has left at that
point is precisely the model's force error, and it is what pushed the prediction
off the saddle.

Both numbers are already on disk and have never been put side by side:

    model force   the extxyz written by the NEB carries a forces array
    DFT force     the ORCA gradient at the same geometry, from the sweep

This measures the model's force error where it matters -- at a transition state
of a multireference reaction -- rather than averaged over a test set.

Sign convention: ORCA prints dE/dx, a force is minus that.
"""
import glob
import json
import os
import re

import numpy as np

H = '/home/energy/s242862'
EH_BOHR_TO_EV_A = 51.42208
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}


def read_extxyz_forces(p):
    """positions are not needed; the forces are the last three columns."""
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    if 'forces' not in L[1]:
        return None
    F = []
    for line in L[2:2 + n]:
        f = line.split()
        if len(f) < 7:
            return None
        F.append([float(x) for x in f[4:7]])
    return np.array(F)


def orca_gradient(label):
    for d in (f'{H}/orca_freq/{label}', f'{H}/orca_irc/{label}'):
        p = f'{d}/engrad.out'
        if not os.path.exists(p):
            continue
        t = open(p, errors='replace').read()
        i = t.find('CARTESIAN GRADIENT')
        if i < 0:
            continue
        G = []
        for line in t[i:].split('\n')[3:]:
            f = line.split()
            if len(f) < 6:
                break
            try:
                G.append([float(v) for v in f[3:6]])
            except ValueError:
                break
        if G:
            return np.array(G) * EH_BOHR_TO_EV_A
    return None


res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
sel = set([res[i]['rxn'] for i in range(26)]
          + [res[i - 1]['rxn'] for i in
             [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
          + [res[i]['rxn'] for i in range(n - 10, n)])
cls = {}
for rx in sel:
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        continue
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    if g and g.get('ext_stable') is not None:
        cls[rx] = 'MR' if g['ext_stable'] is False else 'simple'

rows = []
for rx, c in cls.items():
    for m, dn in MODELDIR.items():
        p = f'{H}/{dn}/{rx}/transition_state.xyz'
        if not os.path.exists(p):
            continue
        Fm = read_extxyz_forces(p)
        G = orca_gradient(f'{rx}_{m}')
        if Fm is None or G is None or len(Fm) != len(G):
            continue
        Fd = -G
        d = Fm - Fd
        rows.append({'rx': rx, 'model': m, 'cls': c,
                     'mae': float(np.abs(d).mean()),
                     'maxerr': float(np.abs(d).max()),
                     'fmodel': float(np.abs(Fm).max()),
                     'fdft': float(np.abs(Fd).max())})

print('MODEL FORCE ERROR AT THE MODEL\'S OWN PREDICTED TRANSITION STATE')
print('=' * 92)
print('All in eV/A. |F| model is what the model believes is left at the point')
print('it stopped at; |F| DFT is what is actually there.')
print()
print(f'{"group":<10}{"n":>4}{"MAE":>9}{"max comp":>10}{"|F| model":>11}'
      f'{"|F| DFT":>10}')
for c in ('simple', 'MR'):
    v = [r for r in rows if r['cls'] == c]
    if not v:
        continue
    print(f'{c:<10}{len(v):>4}'
          f'{np.median([r["mae"] for r in v]):>9.3f}'
          f'{np.median([r["maxerr"] for r in v]):>10.3f}'
          f'{np.median([r["fmodel"] for r in v]):>11.3f}'
          f'{np.median([r["fdft"] for r in v]):>10.3f}')

print()
print('per model, median MAE')
print(f'{"":<10}{"simple":>10}{"MR":>10}{"factor":>9}')
for m in MODELDIR:
    a = [r['mae'] for r in rows if r['model'] == m and r['cls'] == 'simple']
    b = [r['mae'] for r in rows if r['model'] == m and r['cls'] == 'MR']
    if a and b:
        ma, mb = np.median(a), np.median(b)
        print(f'{m:<10}{ma:>10.3f}{mb:>10.3f}{mb / ma:>8.1f}x')

print()
print('The comparison that matters: the model stopped because its own force was')
print('small. Whatever the reference still has there is the error that moved')
print('the prediction off the saddle.')
print()
mr = [r for r in rows if r['cls'] == 'MR']
sp = [r for r in rows if r['cls'] == 'simple']
for lab, v in (('simple', sp), ('MR', mr)):
    if not v:
        continue
    print(f'  {lab:<8} model believes {np.median([r["fmodel"] for r in v]):.3f},'
          f'  reference has {np.median([r["fdft"] for r in v]):.3f}')

print()
print('worst cases in the multireference group')
for r in sorted(mr, key=lambda x: -x['mae'])[:8]:
    print(f'  {r["rx"]:<9}{r["model"]:<7}MAE {r["mae"]:6.3f}   '
          f'max {r["maxerr"]:6.3f}   |F| model {r["fmodel"]:5.3f}   '
          f'DFT {r["fdft"]:5.3f}')
