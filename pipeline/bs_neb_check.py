"""The BS-NEB results, looked at properly this time.

Two things the first pass got wrong.

It used the CI image. ORCA's NEB-TS runs a third phase after the band converges:
a full saddle optimisation starting from the climbing image. Fourteen of these
directories hold bs_uks_neb_NEB-TS_converged.xyz, which is an optimised
transition state, not an interpolated band point. That is the structure to
compare.

It counted <S^2> wrongly. Every band shows a maximum of 2.006 to 2.014, which is
not a diradical singlet but a triplet: ORCA's BrokenSym converges the high-spin
state first and then flips it, so both values land in the log. Reading the
maximum as the broken-symmetry value inflates every profile, and the claim that
only 5 of 11 profiles were coherent may rest on that confusion. Here the
high-spin values are separated out.

The reason any of this matters: a NEB starts from the relaxed reactant and
product and interpolates. It never sees the RKS reference saddle, so it does not
inherit the one bias every transition-state optimisation in this project has.
That independence is worth more than the precision it gives up.

Nothing here decides anything. The three stages still apply, and a NEB structure
has to pass them like any other.
"""
import glob
import json
import os
import re

import numpy as np

H = '/home/energy/s242862'
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
                return f, j['e_uks_final'], d.replace('bs_tsopt_', '')
    return None, None, None


def neb_structures(rx):
    d = f'{H}/bs_uks_neb_results/{rx}'
    out = {}
    for key, pat in (('NEB-TS', '*NEB-TS_converged.xyz'),
                     ('CI', '*NEB-CI_converged.xyz')):
        g = sorted(glob.glob(f'{d}/{pat}'))
        if g:
            out[key] = g[0]
    return out


HS = re.compile(r'<S\*\*2>\s*:\s*([-\d.]+)')


def s2_split(rx):
    """Separate the high-spin reference values from the broken-symmetry ones.

    BrokenSym converges the high-spin state first, so values near S(S+1) = 2 for
    a triplet are the reference, not the band. Only the rest describe the path.
    """
    vals = []
    for p in sorted(glob.glob(f'{H}/bs_uks_neb_results/{rx}/*.out')):
        for line in open(p, errors='replace'):
            m = HS.search(line)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except ValueError:
                    pass
    hs = [v for v in vals if v > 1.8]
    bs = [v for v in vals if v <= 1.8]
    return hs, bs


rxns = sorted(os.path.basename(d) for d in
              glob.glob(f'{H}/bs_uks_neb_results/rxn*') if os.path.isdir(d))

print('THE ORCA BS-NEB-TS STRUCTURES')
print('=' * 104)
print('Distances in A. "ours" is the PySCF BS transition state, whose origin is')
print('given in the last column. The NEB never used it as a starting point.')
print()
print(f'{"rxn":<9}{"phase":<8}{"ours":>7}{"UMA-S":>8}{"UMA-M":>8}{"eSEN":>7}'
      f'   {"closest to":<12}  reactive bonds        ours from')
print('-' * 104)

agree = {'ours': [], 'model': [], 'neither': []}
for rx in rxns:
    st = neb_structures(rx)
    if not st:
        print(f'{rx:<9}no converged structure')
        continue
    key = 'NEB-TS' if 'NEB-TS' in st else 'CI'
    sym, x = read_xyz(st[key])
    og, oe, orig = ours(rx)
    d = {}
    if og and os.path.exists(og):
        d['ours'] = kabsch(x, read_xyz(og)[1])
    for m, dn in MODELS.items():
        p = f'{H}/{dn}/{rx}/transition_state.xyz'
        if os.path.exists(p):
            d[m] = kabsch(x, read_xyz(p)[1])
    if not d:
        continue
    near = min(d, key=lambda k: d[k])
    pairs = reactive(rx)
    bl = ' '.join(f'{nm} {np.linalg.norm(x[a] - x[b]):.2f}'
                  for a, b, nm in pairs) if pairs else '-'
    print(f'{rx:<9}{key:<8}'
          + ''.join(f'{d[k]:>7.3f} ' if k in d else f'{"-":>7} '
                    for k in ('ours', 'UMA-S', 'UMA-M', 'eSEN'))
          + f'  {near:<12}  {bl:<22}{orig or "-"}')
    if 'ours' in d:
        m_best = min((d[k] for k in MODELS if k in d), default=9e9)
        if d['ours'] < m_best - 0.05:
            agree['ours'].append(rx)
        elif m_best < d['ours'] - 0.05:
            agree['model'].append(rx)
        else:
            agree['neither'].append(rx)

print()
print('Which side the independent NEB lands nearer to (0.05 A tolerance):')
for k, v in agree.items():
    print(f'  {k:<9}{len(v):>3}   {" ".join(v)}')

print()
print('<S^2> WITH THE HIGH-SPIN REFERENCE SEPARATED OUT')
print('=' * 104)
print('BrokenSym converges the triplet first; those values sit near 2.0 and are')
print('not part of the path. Only the remainder describes the band.')
print()
print(f'{"rxn":<9}{"n_HS":>6}{"n_BS":>6}{"BS min":>9}{"BS max":>9}'
      f'{"BS mean":>9}{"BS>0.3":>8}')
for rx in rxns:
    hs, bs = s2_split(rx)
    if not bs:
        continue
    n_broken = sum(1 for v in bs if v > 0.3)
    print(f'{rx:<9}{len(hs):>6}{len(bs):>6}{min(bs):>9.3f}{max(bs):>9.3f}'
          f'{sum(bs) / len(bs):>9.3f}{n_broken:>8}')
