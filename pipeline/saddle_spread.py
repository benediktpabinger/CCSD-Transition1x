"""How far apart are the confirmed saddle points of one and the same reaction?

"Confirmed" is used strictly: a frequency calculation exists and found exactly
one imaginary mode. That restricts the comparison to the reactions where model
frequencies were actually computed -- five of them -- plus, listed separately,
the ORCA NEB-TS structures, which ORCA converged as saddles but which have not
been through our own three stages.

The number this produces is the honest answer to "how much does the answer
depend on where you started", because every one of these structures is a valid
first-order saddle of the same molecule at the same level of theory.
"""
import glob
import itertools
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


def confirmed(rx):
    """Every structure for this reaction with a verified single imaginary mode."""
    out = {}
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        rp = f'{H}/{d}/{rx}/result.json'
        if not os.path.exists(rp):
            continue
        j = json.load(open(rp))
        if j.get('e_uks_final') is None:
            continue
        ni = None
        for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
            q = f'{H}/{fd}/{rx}/result.json'
            if os.path.exists(q):
                ni = json.load(open(q)).get('n_imag', ni)
        if ni != 1:
            break
        for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
            if any(p in os.path.basename(f).lower()
                   for p in ('ts', 'final', 'opt')):
                out['ours'] = (f, j['e_uks_final'])
                break
        break
    for m, dn in MODELS.items():
        fp = f'{H}/freq_at_model/{rx}_{m}/result.json'
        g = f'{H}/{dn}/{rx}/transition_state.xyz'
        if os.path.exists(fp) and os.path.exists(g):
            if json.load(open(fp)).get('n_imag') == 1:
                out[m] = (g, None)
    return out


def neb_ts(rx):
    g = sorted(glob.glob(f'{H}/bs_uks_neb_results/{rx}/*NEB-TS_converged.xyz'))
    return g[0] if g else None


rxns = sorted({os.path.basename(os.path.dirname(p))
               for p in glob.glob(f'{H}/freq_at_model/*/result.json')})
rxns = sorted({r.rsplit('_', 1)[0] for r in rxns})

print('CONFIRMED SADDLES OF THE SAME REACTION, PAIRWISE DISTANCE')
print('=' * 78)
print('Every structure below is a first-order saddle of the same molecule at')
print('the same level of theory, with exactly one imaginary mode verified.')
print()
allv = []
for rx in rxns:
    c = confirmed(rx)
    if len(c) < 2:
        print(f'{rx}: fewer than two confirmed saddles ({", ".join(c) or "none"})')
        continue
    pairs = reactive(rx)
    geoms = {k: read_xyz(v[0]) for k, v in c.items()}
    print(f'--- {rx}   {len(c)} confirmed: {", ".join(c)}')
    vals = []
    for a, b in itertools.combinations(c, 2):
        r = kabsch(geoms[a][1], geoms[b][1])
        vals.append((r, a, b))
        allv.append(r)
    for r, a, b in sorted(vals, reverse=True):
        print(f'      {a:<7} vs {b:<7} {r:7.3f} A')
    if pairs:
        print('      reactive bonds:')
        for k in c:
            x = geoms[k][1]
            print(f'        {k:<7} ' + '  '.join(
                f'{nm} {np.linalg.norm(x[a] - x[b]):.3f}' for a, b, nm in pairs))
    print()

if allv:
    a = np.array(allv)
    print(f'over {len(a)} pairs of confirmed saddles: median {np.median(a):.3f} A, '
          f'min {a.min():.3f}, max {a.max():.3f}')

print()
print('THE ORCA NEB-TS STRUCTURES')
print('=' * 78)
print('ORCA converged these as saddles but they have not been through our own')
print('three stages, so they are listed apart from the numbers above.')
print()
for rx in sorted({os.path.basename(os.path.dirname(p))
                  for p in glob.glob(f'{H}/bs_uks_neb_results/*/')}):
    f = neb_ts(rx)
    c = confirmed(rx)
    if not f or 'ours' not in c:
        continue
    x = read_xyz(f)[1]
    o = read_xyz(c['ours'][0])[1]
    print(f'  {rx}   NEB-TS vs ours {kabsch(x, o):.3f} A')
