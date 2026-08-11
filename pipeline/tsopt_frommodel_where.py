"""Where did the transition-state optimisations started from model geometries
actually land?

Eight of these ran (job 10720452) and were never looked at. They are the only
multi-start data in the project: every other transition-state optimisation began
at the RKS reference geometry, so every other structure we have inherits that
starting point. These eight started somewhere else.

Three outcomes are possible and they mean different things:

  back to our structure     two independent starting points converge on the same
                            saddle. That is the best evidence available that it
                            is the relevant one -- short of a proof, which does
                            not exist.
  stayed at the model       the model geometry was already at a stationary
                            point, and it is a different one from ours.
  somewhere else            a third structure nobody has looked at.

The distances alone do not say whether the endpoint is a saddle. A run that
slides into a minimum also reports convergence, because the criterion is on the
gradient -- that is what happened twice in bs_tsopt_retry. The <S^2> and the
energy are printed alongside so that case is visible.
"""
import glob
import json
import os

import numpy as np

H = '/home/energy/s242862'
HA_MEV = 27211.386
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results'}


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


# find every directory that looks like a from-model TS optimisation
cands = sorted(set(glob.glob(f'{H}/tsopt_*model*/*/') +
                   glob.glob(f'{H}/*frommodel*/*/') +
                   glob.glob(f'{H}/tsopt_from_model/*/')))
print('directories found:', len(cands))
for c in cands[:5]:
    print('   ', c)
print()

print(f'{"case":<20}{"vs ours":>9}{"vs start":>10}{"dE meV":>10}'
      f'{"S2":>8}{"steps":>7}  outcome')
print('-' * 88)
for c in cands:
    tag = os.path.basename(os.path.dirname(c))
    rp = os.path.join(c, 'result.json')
    if not os.path.exists(rp):
        continue
    j = json.load(open(rp))
    rx = j.get('rxn') or tag.split('_')[0]
    mod = j.get('start_model') or (tag.split('_')[1] if '_' in tag else '?')
    xyzs = [f for f in glob.glob(os.path.join(c, '*.xyz'))
            if any(p in os.path.basename(f).lower()
                   for p in ('ts', 'final', 'opt'))]
    if not xyzs:
        print(f'{tag:<20}no geometry written   status={j.get("status")}')
        continue
    _, xe = read_xyz(xyzs[0])
    og, oe = ours(rx)
    d_ours = kabsch(xe, read_xyz(og)[1]) if og else float('nan')
    sp = f'{H}/{MODELS.get(mod, "")}/{rx}/transition_state.xyz'
    d_start = kabsch(xe, read_xyz(sp)[1]) if os.path.exists(sp) else float('nan')
    e = j.get('e_uks_final')
    de = (e - oe) * HA_MEV if (e is not None and oe is not None) else float('nan')
    s2 = j.get('s2_final')
    n = j.get('n_geom_steps')
    if d_ours < 0.10:
        out = 'converged back to ours'
    elif d_start < 0.10:
        out = 'stayed at the model geometry'
    else:
        out = 'THIRD STRUCTURE'
    print(f'{tag:<20}{d_ours:9.3f}{d_start:10.3f}{de:10.1f}'
          f'{s2 if s2 is not None else float("nan"):8.3f}'
          f'{n if n is not None else -1:7d}  {out}')
