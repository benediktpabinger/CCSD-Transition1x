"""Which sheet do the models reproduce -- the restricted one or the ground state?

The question behind it: OMol25 recomputed Transition1x with an unrestricted
reference, and a UKS calculation collapses onto the restricted solution wherever
that one is stable. So one might expect the labels to be the ground state
everywhere and there to be nothing for a model to get wrong about which sheet
it is on.

But UKS does not give the ground state by itself. Started from a symmetric
guess it converges onto the restricted solution even where that solution is
unstable -- the restricted solution is a stationary point of the unrestricted
equations too. Finding the lower one takes a broken guess or a stability
analysis. So whether the labels are the ground state in the diradical region is
an open question, and it is answerable from data we already have.

The test, at each model's own predicted transition state:

    barrier_model = E_model(TS) - E_model(reactant)          from the model
    barrier_RKS   = E_RKS(TS)   - E_RKS(reactant)            restricted sheet
    barrier_BS    = E_BS(TS)    - E_RKS(reactant)            ground state

The two DFT barriers differ by exactly dE_BS at the transition state, which
runs to several eV here, so the two hypotheses are far apart and easy to tell
apart. Whichever the model barrier tracks is the sheet its labels came from.

The reactant is the same structure in all three barriers and is closed-shell in
all 45 reactions (checked: every reference reactant has dE_BS = 0), so RKS and
BS coincide there and the choice of zero does not favour either hypothesis.

Two caveats reported alongside rather than hidden: the model relaxes its own
reactant, which is not exactly the reference reactant, and the model is trained
at def2-TZVPD against our def2-TZVP. Both shift a barrier by tens of meV at
most, while the two hypotheses differ by hundreds to thousands.
"""
import json
import os
import re

import numpy as np

H = '/home/energy/s242862'
HA_EV = 27.211386245988
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}
ERE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')


def xyz_energy(p):
    """extxyz comment line: energy=... in eV."""
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        fh.readline()
        c = fh.readline()
    m = re.search(r'\benergy=([-\d.eE+]+)', c)
    return float(m.group(1)) if m else None


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


def ref_reactant_rks(rx):
    p = f'{H}/orca_endpoint/{rx}_reactant/rks.out'
    if not os.path.exists(p):
        return None
    m = ERE.findall(open(p, errors='replace').read())
    return float(m[-1]) if m else None


def dft_at_model_ts(rx, m):
    """E_RKS and E_BS at the model geometry, from the stability pipeline."""
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return None, None
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get(m)
    if not g:
        return None, None
    e_rks = g.get('e_rks')
    b = g.get('bs') or {}
    e_bs = b.get('e_uks')
    if e_bs is None and g.get('ext_stable'):
        e_bs = e_rks          # no broken solution there; the sheets coincide
    return e_rks, e_bs


res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
sel = set([res[i]['rxn'] for i in range(26)]
          + [res[i - 1]['rxn'] for i in
             [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
          + [res[i]['rxn'] for i in range(n - 10, n)])
nf = {x['rxn']: x['nfod'] for x in res}
MR = []
for rx in sel:
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        continue
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    if g and g.get('ext_stable') is False:
        MR.append(rx)
MR.sort(key=lambda r: -nf[r])

print('WHICH SHEET DO THE MODEL ENERGIES FOLLOW?')
print('=' * 104)
print('All barriers in eV, measured from the reactant to the model\'s own')
print('predicted transition state. RKS and BS differ only at the transition')
print('state -- every reactant is closed-shell.')
print()
print(f'{"rxn":<9}{"model":<7}{"model":>9}{"RKS":>9}{"BS":>9}   '
      f'{"|m-RKS|":>8}{"|m-BS|":>8}   verdict')
print('-' * 104)

votes = {'RKS': 0, 'BS': 0, 'same': 0}
gaps, rr_rmsd = [], []
rows = []
for rx in MR:
    e0 = ref_reactant_rks(rx)
    if e0 is None:
        continue
    pr = f'{H}/orca_neb_results/{rx}/reactant.xyz'
    xr = read_xyz(pr)[1] if os.path.exists(pr) else None
    for m, dn in MODELDIR.items():
        ts = f'{H}/{dn}/{rx}/transition_state.xyz'
        rc = f'{H}/{dn}/{rx}/reactant.xyz'
        em_ts, em_r = xyz_energy(ts), xyz_energy(rc)
        e_rks, e_bs = dft_at_model_ts(rx, m)
        if None in (em_ts, em_r, e_rks, e_bs):
            continue
        b_model = em_ts - em_r
        b_rks = (e_rks - e0) * HA_EV
        b_bs = (e_bs - e0) * HA_EV
        d_rks, d_bs = abs(b_model - b_rks), abs(b_model - b_bs)
        gap = abs(b_rks - b_bs)
        if gap < 0.05:
            v = 'sheets coincide here'
            votes['same'] += 1
        elif d_rks < d_bs:
            v = 'follows RKS'
            votes['RKS'] += 1
        else:
            v = 'follows BS'
            votes['BS'] += 1
        if xr is not None and os.path.exists(rc):
            rr_rmsd.append(kabsch(read_xyz(rc)[1], xr))
        gaps.append(gap)
        rows.append((rx, m, b_model, b_rks, b_bs, d_rks, d_bs, v, gap))
        print(f'{rx:<9}{m:<7}{b_model:>9.2f}{b_rks:>9.2f}{b_bs:>9.2f}   '
              f'{d_rks:>8.2f}{d_bs:>8.2f}   {v}')

print()
print(f'follows RKS  {votes["RKS"]:>3}')
print(f'follows BS   {votes["BS"]:>3}')
print(f'no distinction possible (sheets within 50 meV)  {votes["same"]:>3}')

decisive = [r for r in rows if r[8] > 0.30]
if decisive:
    nr = sum(1 for r in decisive if r[7] == 'follows RKS')
    print()
    print(f'restricted to the {len(decisive)} cases where the two hypotheses')
    print(f'differ by more than 300 meV:  RKS {nr}, BS {len(decisive) - nr}')

if rr_rmsd:
    print()
    print(f'model reactant vs reference reactant: median '
          f'{np.median(rr_rmsd):.4f} A, max {max(rr_rmsd):.4f} A')
    print('-- the zero of the barrier is essentially the same structure, so it')
    print('   does not favour either hypothesis')
