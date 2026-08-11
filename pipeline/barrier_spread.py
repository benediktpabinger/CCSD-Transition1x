"""Barrier disagreement between the three OMol25 models.

A forward barrier is E(TS) - E(reactant). Asking how much the models disagree
about it therefore has two contributions: they may place the transition state
differently, and they may place the reactant differently. If they agree on the
reactant, that term cancels in the difference between two models and the barrier
disagreement equals the disagreement in transition-state energy alone.

The script checks that assumption instead of assuming it: it reports the
geometric spread of the model reactants next to the spread of their transition
states. Where the reactants coincide, the transition-state energy spread IS the
barrier spread.

All energies are DFT single points at the models' own predicted geometries,
wB97M-V/def2-TZVP in PySCF, on whichever surface is the ground state there --
the broken-symmetry solution where the restricted one is externally unstable.
The DFT method is identical across all of them; every difference comes from the
models predicting different structures.

Groups:
  "single-reference"  the 26 reactions whose RKS reference TS is externally
                      stable -- no broken-symmetry solution exists, the ordinary
                      closed-shell picture holds
  "multireference"    the 19 where it does not
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


def spread(vals):
    v = [x for x in vals if x is not None]
    return (max(v) - min(v)) if len(v) > 1 else None


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
    cls = ('multireference' if ref['ext_stable'] is False
           else 'single-reference')

    e_ts, x_ts, x_r = {}, {}, {}
    for m, dn in MODELS.items():
        g = geo.get(m)
        if g and g.get('ext_stable') is not None:
            e_ts[m] = (g.get('e_rks') if g['ext_stable']
                       else (g.get('bs') or {}).get('e_uks'))
        f = f'{H}/{dn}/{rx}/transition_state.xyz'
        if os.path.exists(f):
            x_ts[m] = read_xyz(f)
        fr = f'{H}/{dn}/{rx}/reactant.xyz'
        if os.path.exists(fr):
            x_r[m] = read_xyz(fr)

    def geo_spread(xd):
        out = []
        for a, b in itertools.combinations(xd, 2):
            if xd[a][0] == xd[b][0]:
                out.append(kabsch(xd[a][1], xd[b][1]))
        return max(out) if out else None

    s = spread(list(e_ts.values()))
    rows.append({'rxn': rx, 'cls': cls, 'nfod': nf[rx],
                 'dE_ts': None if s is None else s * HA_MEV,
                 'rmsd_ts': geo_spread(x_ts),
                 'rmsd_react': geo_spread(x_r),
                 'n_react': len(x_r)})

print('Barrier disagreement between UMA-S, UMA-M and eSEN')
print('=' * 78)
print()
print('Every number is the largest pairwise difference between the three')
print('models. No DFT reference enters -- this is model against model.')
print()
print('  TS energy spread   max - min of the DFT energy evaluated at each')
print('                     model\'s own predicted transition state [meV].')
print('                     With a shared reactant this equals the spread in')
print('                     forward barrier, because the reactant cancels.')
print('  TS geometry        largest pairwise Kabsch RMSD of those transition')
print('                     states [A]')
print('  reactant geometry  same for the relaxed reactants, to check whether')
print('                     the reactant really does cancel')
print()

have_react = sum(1 for r in rows if r['rmsd_react'] is not None)
print(f'reactant geometries available for {have_react} of {len(rows)} reactions')
print()

print(f"{'group':<20}{'n':>4}{'TS energy spread [meV]':>26}"
      f"{'TS geometry [A]':>19}")
print(f"{'':<20}{'':>4}{'median':>11}{'max':>15}{'median':>11}{'max':>8}")
print('-' * 69)
for cls in ('single-reference', 'multireference'):
    s = [r for r in rows if r['cls'] == cls]
    e = np.array([r['dE_ts'] for r in s if r['dE_ts'] is not None])
    g = np.array([r['rmsd_ts'] for r in s if r['rmsd_ts'] is not None])
    print(f'{cls:<20}{len(s):>4}{np.median(e):>11.1f}{e.max():>15.1f}'
          f'{np.median(g):>11.4f}{g.max():>8.4f}')

if have_react:
    print()
    print(f"{'group':<20}{'n':>4}{'reactant geometry [A]':>25}")
    print(f"{'':<20}{'':>4}{'median':>13}{'max':>12}")
    print('-' * 49)
    for cls in ('single-reference', 'multireference'):
        v = np.array([r['rmsd_react'] for r in rows
                      if r['cls'] == cls and r['rmsd_react'] is not None])
        if len(v):
            print(f'{cls:<20}{len(v):>4}{np.median(v):>13.4f}{v.max():>12.4f}')

print()
print('How many reactions exceed a given disagreement')
print('-' * 62)
sr = [r for r in rows if r['cls'] == 'single-reference']
mr = [r for r in rows if r['cls'] == 'multireference']
for label, thr in (('TS energy spread >  10 meV', 10),
                   ('TS energy spread >  50 meV', 50),
                   ('TS energy spread > 250 meV', 250),
                   ('TS energy spread >   1 eV', 1000)):
    a = sum(1 for r in sr if r['dE_ts'] is not None and r['dE_ts'] > thr)
    b = sum(1 for r in mr if r['dE_ts'] is not None and r['dE_ts'] > thr)
    print(f'  {label:<28} single-reference {a:>2}/{len(sr)}'
          f'   multireference {b:>2}/{len(mr)}')

print()
print('Multireference reactions, sorted by disagreement')
print('-' * 62)
print(f"{'reaction':<11}{'N_FOD':>7}{'TS energy [meV]':>17}{'TS geom [A]':>13}")
for r in sorted(mr, key=lambda x: -(x['dE_ts'] or 0)):
    e = '—' if r['dE_ts'] is None else '{:.1f}'.format(r['dE_ts'])
    g = '—' if r['rmsd_ts'] is None else '{:.4f}'.format(r['rmsd_ts'])
    print('{:<11}{:>7.3f}{:>17}{:>13}'.format(r['rxn'], r['nfod'], e, g))
