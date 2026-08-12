"""Two statistics about the three OMol25 models, now that every prediction of
the multireference set has a Hessian.

  1. How often does a model land on a first-order saddle at all?
     Per model, and split by whether the restricted reference is valid.

  2. When they do, do the three land on the same saddle or on different ones?
     Disagreement here proves at least two of the three are wrong without any
     reference entering, which is the only kind of statement about these models
     that does not depend on our own structures being right.

Both are reference-free. Neither asks whether the saddle is the right one.

A structure counts as a saddle only if it is stationary as well: a Hessian at a
point with a large residual force has eigenvalues but no meaning as a
transition state. Threshold 0.15 eV/A, against 0.006-0.011 at our confirmed
saddles.

Caveat on the control group: only one model geometry per single-reference
reaction was given a Hessian, chosen as the one nearest stationary, because the
three agree there to 0.0045 A median and are the same structure. So the control
group supports an aggregate number but not a per-model split, and question 2
cannot be asked of it -- the answer is fixed by the geometries.
"""
import glob
import json
import os
import re

import numpy as np
from ase.data import atomic_masses, atomic_numbers

H = '/home/energy/s242862'
BOHR = 0.529177210903
CM = 5140.4871
MODELS = ('UMA-S', 'UMA-M', 'eSEN')
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}
GRAD_OK = 0.15
SAME_RMSD = 0.10        # same structure
SAME_BOND = 0.05        # same chemistry, conformation allowed to differ


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B):
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def read_orca_hess(path):
    lines = open(path).read().split('\n')
    i = next(k for k, l in enumerate(lines) if l.strip() == '$hessian')
    n = int(lines[i + 1].split()[0])
    Hm = np.zeros((n, n))
    k, cols = i + 2, []
    while True:
        t = lines[k].split()
        k += 1
        if not t:
            continue
        if all(x.lstrip('-').isdigit() for x in t) and len(t) <= 8:
            cols = [int(x) for x in t]
            continue
        r = int(t[0])
        for c, v in zip(cols, t[1:]):
            Hm[r, c] = float(v)
        if r == n - 1 and cols and cols[-1] == n - 1:
            break
    return Hm


def n_imag_of(hess, sym, xyz):
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    msqrt = np.sqrt(m)
    w = np.repeat(1.0 / msqrt, 3)
    Hm = hess * w[:, None] * w[None, :]
    nat = len(sym)
    c = xyz / BOHR
    c = c - (c * (msqrt ** 2)[:, None]).sum(0) / (msqrt ** 2).sum()
    B = []
    for k in range(3):
        v = np.zeros((nat, 3)); v[:, k] = msqrt
        B.append(v.ravel())
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        B.append((np.cross(np.tile(e, (nat, 1)), c) * msqrt[:, None]).ravel())
    U, s, _ = np.linalg.svd(np.array(B).T, full_matrices=False)
    P = U[:, s > 1e-8]
    Q = np.eye(len(Hm)) - P @ P.T
    ev = np.linalg.eigvalsh(Q @ Hm @ Q)
    fr = np.sign(ev) * np.sqrt(np.abs(ev)) * CM
    return int((fr < -20).sum()), float(fr.min())


def hess_of(rx, m):
    for d in (f'{H}/orca_freq/{rx}_{m}', f'{H}/orca_irc/{rx}_{m}'):
        p = f'{d}/numfreq.hess'
        if os.path.exists(p):
            return p
    p = f'{H}/freq_at_model/{rx}_{m}/hessian.npy'
    return p if os.path.exists(p) else None


def grad_of(rx, m):
    for d in (f'{H}/orca_freq/{rx}_{m}', f'{H}/orca_irc/{rx}_{m}'):
        p = f'{d}/engrad.out'
        if os.path.exists(p):
            t = open(p, errors='replace').read()
            i = t.find('CARTESIAN GRADIENT')
            if i > 0:
                mx = 0.0
                for line in t[i:].split('\n')[3:]:
                    f = line.split()
                    if len(f) < 6:
                        break
                    for v in f[3:6]:
                        try:
                            mx = max(mx, abs(float(v)))
                        except ValueError:
                            pass
                if mx:
                    return mx * 51.42208
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if os.path.exists(p):
        g = {x['source']: x
             for x in json.load(open(p))['geometries']}.get(m)
        if g and g.get('ext_stable') is not None:
            if g['ext_stable']:
                return (g.get('rks_grad') or {}).get('max_evang')
            return ((g.get('bs') or {}).get('bs_grad') or {}).get('max_evang')
    return None


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
sel = set([res[i]['rxn'] for i in range(26)]
          + [res[i - 1]['rxn'] for i in
             [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
          + [res[i]['rxn'] for i in range(n - 10, n)])
nf = {x['rxn']: x['nfod'] for x in res}
MR, SIMPLE = [], []
for rx in sel:
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        continue
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    if not g or g.get('ext_stable') is None:
        continue
    (MR if g['ext_stable'] is False else SIMPLE).append(rx)
MR.sort(key=lambda r: -nf[r])

# ---------------------------------------------------------------- 1
print('=' * 76)
print('1  HOW OFTEN DOES A MODEL LAND ON A FIRST-ORDER SADDLE?')
print('=' * 76)
print(f'stationary means gradient < {GRAD_OK} eV/A; our confirmed saddles are')
print('at 0.006 to 0.011. A point that is not stationary is not a saddle,')
print('whatever its Hessian says.')
print()
print(f'{"multireference":<16}{"n":>4}{"stationary":>12}{"1 imag":>9}'
      f'{"both":>7}{"share":>8}')
per = {}
for m in MODELS:
    tot = st = im = both = 0
    per[m] = {}
    for rx in MR:
        hp = hess_of(rx, m)
        g = f'{H}/{MODELDIR[m]}/{rx}/transition_state.xyz'
        if not hp or not os.path.exists(g):
            continue
        sym, xyz = read_xyz(g)
        hs = read_orca_hess(hp) if hp.endswith('.hess') else np.load(hp)
        ni, lo = n_imag_of(hs, sym, xyz)
        gr = grad_of(rx, m)
        tot += 1
        a = gr is not None and gr < GRAD_OK
        b = ni == 1
        st += a; im += b
        if a and b:
            both += 1
        per[m][rx] = {'saddle': a and b, 'ni': ni, 'grad': gr, 'imag': lo}
    print(f'{m:<16}{tot:>4}{st:>12}{im:>9}{both:>7}'
          f'{both / tot * 100 if tot else 0:>7.0f}%')

ns = sc = 0
for rx in SIMPLE:
    for m in MODELS:
        hp = hess_of(rx, m)
        g = f'{H}/{MODELDIR[m]}/{rx}/transition_state.xyz'
        if not hp or not os.path.exists(g):
            continue
        sym, xyz = read_xyz(g)
        hs = read_orca_hess(hp) if hp.endswith('.hess') else np.load(hp)
        ni, lo = n_imag_of(hs, sym, xyz)
        gr = grad_of(rx, m)
        ns += 1
        if gr is not None and gr < GRAD_OK and ni == 1:
            sc += 1
print()
print(f'{"single-reference":<16}{ns:>4}{"":>12}{"":>9}{sc:>7}'
      f'{sc / ns * 100 if ns else 0:>7.0f}%   (one geometry per reaction)')


# ---------------------------------------------------------------- 1b
def ours_struct(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        rp = f'{H}/{d}/{rx}/result.json'
        if not os.path.exists(rp):
            continue
        if json.load(open(rp)).get('e_uks_final') is None:
            continue
        for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
            if any(p in os.path.basename(f).lower()
                   for p in ('ts', 'final', 'opt')):
                return f
    return None


def ours_hess(rx):
    for d in (f'{H}/orca_freq/ours_{rx}', f'{H}/orca_irc/{rx}_ours'):
        p = f'{d}/numfreq.hess'
        if os.path.exists(p):
            return p
    for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
        p = f'{H}/{fd}/{rx}/hessian.npy'
        if os.path.exists(p):
            return p
    return None


def ours_grad(rx):
    for d in (f'{H}/orca_freq/ours_{rx}', f'{H}/orca_irc/{rx}_ours'):
        p = f'{d}/engrad.out'
        if os.path.exists(p):
            t = open(p, errors='replace').read()
            i = t.find('CARTESIAN GRADIENT')
            if i > 0:
                mx = 0.0
                for line in t[i:].split('\n')[3:]:
                    f = line.split()
                    if len(f) < 6:
                        break
                    for v in f[3:6]:
                        try:
                            mx = max(mx, abs(float(v)))
                        except ValueError:
                            pass
                if mx:
                    return mx * 51.42208
    for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
        p = f'{H}/{fd}/{rx}/result.json'
        if os.path.exists(p):
            v = json.load(open(p)).get('max_grad_ha_bohr')
            if v is not None:
                return float(v) * 51.42208
    return None


def ref_grad(rx):
    """The benchmark's own reference transition state, measured on whichever
    surface is the ground state there -- BS where the restricted solution is
    externally unstable."""
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return None
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    if not g or g.get('ext_stable') is None:
        return None
    if g['ext_stable']:
        return (g.get('rks_grad') or {}).get('max_evang')
    return ((g.get('bs') or {}).get('bs_grad') or {}).get('max_evang')


print()
print('=' * 76)
print('1b  THE SAME QUESTION FOR OUR OWN STRUCTURES AND FOR THE BENCHMARK')
print('    REFERENCE')
print('=' * 76)
print('The reference is the ORCA NEB transition state the whole benchmark is')
print('scored against. Its gradient is measured on the ground-state surface,')
print('which for the multireference reactions is the broken-symmetry one -- so')
print('the number says whether the reference is a stationary point of the')
print('surface the reaction actually runs on. No Hessian was ever computed at')
print('the reference, so its stage 2 is unknown by construction.')
print()
print(f'{"":22}{"n":>4}{"stationary":>12}{"1 imag":>9}{"saddle":>8}{"share":>8}')

for cls, RXS in (('multireference', MR), ('single-reference', SIMPLE)):
    tot = st = im = both = 0
    for rx in RXS:
        g = ours_struct(rx)
        hp = ours_hess(rx)
        if not g or not hp:
            continue
        sym, xyz = read_xyz(g)
        hs = read_orca_hess(hp) if hp.endswith('.hess') else np.load(hp)
        ni, lo = n_imag_of(hs, sym, xyz)
        gr = ours_grad(rx)
        tot += 1
        a = gr is not None and gr < GRAD_OK
        b = ni == 1
        st += a; im += b
        if a and b:
            both += 1
    if tot:
        print(f'{"ours, " + cls:<22}{tot:>4}{st:>12}{im:>9}{both:>8}'
              f'{both / tot * 100:>7.0f}%')

for cls, RXS in (('multireference', MR), ('single-reference', SIMPLE)):
    gs = [ref_grad(rx) for rx in RXS]
    gs = [g for g in gs if g is not None]
    st = sum(1 for g in gs if g < GRAD_OK)
    print(f'{"reference, " + cls:<22}{len(gs):>4}{st:>12}{"unknown":>9}'
          f'{"unknown":>8}{"":>8}   median grad {np.median(gs):.3f} eV/A')

print()
print('Our structures are compared on their own terms: they were optimised to')
print('be saddles, so a high rate here is expected and is a check on the')
print('optimisation, not a result about the models. The reference row is the')
print('one that carries information -- it is what every RMSD in the benchmark')
print('is measured against.')

# ---------------------------------------------------------------- 2
print()
print('=' * 76)
print('2  DO THE THREE LAND ON THE SAME SADDLE?')
print('=' * 76)
print(f'same structure: RMSD < {SAME_RMSD} A')
print(f'same chemistry: both reactive bonds within {SAME_BOND} A, conformation')
print('                allowed to differ')
print()
print(f'{"rxn":<9}{"saddles":>8}   {"pairwise RMSD":<24}{"reaktive Abweichung":<22} '
      f'verdict')
tally = {}
for rx in MR:
    pairs = reactive(rx)
    good = [m for m in MODELS if per[m].get(rx, {}).get('saddle')]
    if len(good) == 0:
        v = 'no model is a saddle'
        tally[v] = tally.get(v, 0) + 1
        print(f'{rx:<9}{0:>8}   {"":<24}{"":<22} {v}')
        continue
    xs = {}
    for m in good:
        xs[m] = read_xyz(f'{H}/{MODELDIR[m]}/{rx}/transition_state.xyz')[1]
    rms, dbs = [], []
    ks = list(xs)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            rms.append(kabsch(xs[ks[i]], xs[ks[j]]))
            if pairs:
                dbs.append(max(abs(np.linalg.norm(xs[ks[i]][a] - xs[ks[i]][b])
                                   - np.linalg.norm(xs[ks[j]][a] - xs[ks[j]][b]))
                               for a, b, _ in pairs))
    # The reactive bonds decide first. Whole-molecule RMSD can read "same"
    # while the bond that makes and breaks differs by 0.2 A -- rxn1283 does
    # exactly that -- and the bonds are what the question is about.
    if len(good) == 1:
        v = 'only one model is a saddle'
    elif dbs and max(dbs) >= SAME_BOND:
        v = '*** DIFFERENT SADDLES ***'
    elif rms and max(rms) < SAME_RMSD:
        v = 'same saddle'
    else:
        v = 'same chemistry, different conformation'
    tally[v] = tally.get(v, 0) + 1
    r = f'{min(rms):.3f} - {max(rms):.3f}' if rms else '—'
    d = f'{min(dbs):.3f} - {max(dbs):.3f}' if dbs else '—'
    print(f'{rx:<9}{len(good):>8}   {r:<24}{d:<22} {v}')

print()
for k, v in sorted(tally.items(), key=lambda x: -x[1]):
    print(f'  {v:>3}   {k}')

print()
print('For the single-reference reactions the question does not arise: the')
print('three models agree there to 0.0045 A median over all 26, which is one')
print('structure by any measure.')
