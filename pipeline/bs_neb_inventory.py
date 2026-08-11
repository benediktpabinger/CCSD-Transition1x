"""What did the broken-symmetry NEB actually find, and where does it sit?

The BS-NEB route was set aside on a precision argument: 0.669 A of scatter in
the null measurement against 0.021 A for the transition-state optimisation, and
coherent <S^2> profiles in only 5 of 11 bands. Precision is not correctness,
and the two failure modes are unrelated to each other.

The reason to look again is the starting point. Every transition-state
optimisation in this project began at the RKS reference geometry -- the point we
know is not the ground state -- so it finds whichever saddle lies downhill of
that, and the starting geometry decides the answer. A NEB does not start there.
It starts from the relaxed reactant and product and interpolates, so it carries
none of that bias. Even the frommodel sweep does not have this property: it
still starts from one prescribed structure.

And in the one case where the right answer is independently known -- rxn4113,
where a second basin lies 0.93 A from the reference -- the BS-NEB found it.

This script only inventories and measures. It does not judge: the three stages
still apply, and a NEB structure has to pass them like any other.
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


def neb_ts(rx):
    """The converged CI-NEB image, whatever it is called in that directory."""
    for pat in ('*NEB-CI_converged.xyz', '*NEB_CI_converged.xyz',
                '*_NEB-TS_converged.xyz', '*NEB-CI*.xyz'):
        g = sorted(glob.glob(f'{H}/bs_uks_neb_results/{rx}/{pat}'))
        if g:
            return g[0]
    return None


def s2_profile(rx):
    """<S^2> across the band, from whatever log the run left behind."""
    vals = []
    for p in sorted(glob.glob(f'{H}/bs_uks_neb_results/{rx}/*.out')
                    + glob.glob(f'{H}/bs_uks_neb_results/{rx}/*.log')):
        for line in open(p, errors='replace'):
            if '<S**2>' in line:
                try:
                    vals.append(float(line.split()[-1]))
                except ValueError:
                    pass
    return vals


rxns = sorted(os.path.basename(d) for d in
              glob.glob(f'{H}/bs_uks_neb_results/rxn*') if os.path.isdir(d))
print(f'{len(rxns)} reactions have a BS-NEB directory\n')
print(f'{"rxn":<9}{"CI image":<10}{"vs ours":>9}{"vs UMA-S":>10}'
      f'{"vs UMA-M":>10}{"vs eSEN":>9}   reactive bonds at the NEB structure')
print('-' * 108)

rows = []
for rx in rxns:
    f = neb_ts(rx)
    if not f:
        print(f'{rx:<9}no converged CI image')
        continue
    sym, x = read_xyz(f)
    og, oe = ours(rx)
    d = {}
    if og and os.path.exists(og):
        d['ours'] = kabsch(x, read_xyz(og)[1])
    for m, dn in MODELS.items():
        p = f'{H}/{dn}/{rx}/transition_state.xyz'
        if os.path.exists(p):
            d[m] = kabsch(x, read_xyz(p)[1])
    pairs = reactive(rx)
    bl = '  '.join(f'{nm} {np.linalg.norm(x[a] - x[b]):.3f}'
                   for a, b, nm in pairs) if pairs else '-'
    print(f'{rx:<9}{"yes":<10}'
          + ''.join(f'{d.get(k, float("nan")):>9.3f} ' if k in d
                    else f'{"-":>9} ' for k in ('ours', 'UMA-S', 'UMA-M', 'eSEN'))
          + f'  {bl}')
    rows.append((rx, d, bl))

print()
print('Interpretation: a NEB structure far from ours but close to a model')
print('geometry points at the same second basin the frommodel sweep is after,')
print('found independently -- the NEB never sees the reference saddle.')
print()
print('<S^2> across the band (a coherent profile means the broken solution')
print('survived the whole path; the ends should be near zero, that is correct)')
for rx, _, _ in rows:
    v = s2_profile(rx)
    if v:
        print(f'  {rx:<9}n={len(v):<4} min {min(v):6.3f}  max {max(v):6.3f}  '
              f'mean {sum(v) / len(v):6.3f}')
