"""Are the model geometry and ours the same structure, or two different ones?

Where they are the same, our frequency already covers both and no Hessian at the
model geometry is needed -- the missing calculation is not missing, it is
unnecessary. Where they differ, the gap is real.

This is decided from the geometries alone, at no computational cost, and it
should have been the first thing checked before counting 42 missing Hessians.

Two measures, because one hides the difference between the two failure modes:
  RMSD       whole molecule after Kabsch alignment. Catches conformational
             differences that do not touch the chemistry.
  reactive   largest change in the two bonds that make and break. This is the
             one that decides whether it is the same transition state.
"""
import glob
import json
import os

import numpy as np

H = '/home/energy/s242862'
HA_MEV = 27211.386
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results'}
SAME_RMSD = 0.05      # below this the two are the same structure
SAME_BOND = 0.03      # and the reactive bonds agree


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B):
    if len(A) != len(B):
        return float('nan')
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def ours(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        rp = f'{H}/{d}/{rx}/result.json'
        if not os.path.exists(rp):
            continue
        j = json.load(open(rp))
        if j.get('e_uks_final') is None:
            continue
        for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
            if any(p in os.path.basename(f).lower()
                   for p in ('ts', 'final', 'opt')):
                return f, j['e_uks_final']
    return None, None


def has_freq(rx, m):
    return os.path.exists(f'{H}/freq_at_model/{rx}_{m}/result.json')


res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
nf = {x['rxn']: x['nfod'] for x in res}

RX = ['rxn8837', 'rxn7949', 'rxn1147', 'rxn4113', 'rxn8885', 'rxn0894',
      'rxn5691', 'rxn4522', 'rxn7957',
      'rxn7060', 'rxn3107', 'rxn0346', 'rxn6196', 'rxn8832', 'rxn8827',
      'rxn1320', 'rxn4518', 'rxn1283', 'rxn5690']
GRP = {}
for r in RX[:6]:  GRP[r] = 'A ours'
for r in RX[6:9]: GRP[r] = 'B models'
for r in RX[9:15]: GRP[r] = 'C same?'
for r in RX[15:]: GRP[r] = 'D open'

print('IS THE MODEL GEOMETRY THE SAME STRUCTURE AS OURS?')
print('=' * 96)
print(f'same = RMSD < {SAME_RMSD} A and both reactive bonds agree to '
      f'{SAME_BOND} A')
print('where they are the same, our frequency already covers the model and no')
print('Hessian at the model geometry is needed')
print()
print(f'{"rxn":<9}{"group":<10}{"model":<7}{"RMSD":>7}{"d1":>8}{"d2":>8}'
      f'{"dE meV":>9}  {"freq":<6} verdict')
print('-' * 96)

need, nonneed = [], []
for rx in RX:
    og, oe = ours(rx)
    pairs = reactive(rx)
    if not og or not pairs:
        print(f'{rx:<9}{GRP[rx]:<10}no saddle of ours to compare against')
        for m in MODELS:
            if not has_freq(rx, m):
                need.append((rx, m))
        continue
    _, xo = read_xyz(og)
    do = [np.linalg.norm(xo[a] - xo[b]) for a, b, _ in pairs]
    for m, dn in MODELS.items():
        p = f'{H}/{dn}/{rx}/transition_state.xyz'
        if not os.path.exists(p):
            continue
        _, xm = read_xyz(p)
        r = kabsch(xo, xm)
        dm = [np.linalg.norm(xm[a] - xm[b]) for a, b, _ in pairs]
        dd = [abs(x - y) for x, y in zip(do, dm)]
        same = r < SAME_RMSD and max(dd) < SAME_BOND
        f = 'yes' if has_freq(rx, m) else 'NO'
        v = 'SAME structure' if same else 'different'
        print(f'{rx:<9}{GRP[rx]:<10}{m:<7}{r:7.3f}{dd[0]:8.3f}{dd[1]:8.3f}'
              f'{"":9}  {f:<6} {v}')
        if same:
            nonneed.append((rx, m))
        elif not has_freq(rx, m):
            need.append((rx, m))

print()
print(f'model geometries identical to ours (no Hessian needed): {len(nonneed)}')
for rx, m in nonneed:
    print(f'    {rx} {m}')
print(f'\nmodel geometries genuinely different and untested: {len(need)}')
byrx = {}
for rx, m in need:
    byrx.setdefault(rx, []).append(m)
for rx, ms in byrx.items():
    print(f'    {rx:<9}{" ".join(ms)}')
