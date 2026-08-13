"""One line per reaction: who found a saddle, who did not and why.

The short reasons matter more than the symbols. "not stationary" and "two
imaginary modes" are different failures -- the first means the structure is not
at a stationary point at all, the second that it is at one but of the wrong
kind -- and both had previously been collapsed into a blank cell together with
"nobody tested this".

The last column compares every saddle found for that reaction against every
other, so a reaction where several methods agree looks different from one where
they found separate structures. The comparison is on the reactive bonds, not on
whole-molecule RMSD: two conformers of the same transition state are the same
saddle for this purpose, while 0.06 A of RMSD can hide a reactive bond that
differs by 0.2 A.
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
SAME_BOND = 0.05


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


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
    return int((fr < -20).sum())


def engrad(d):
    p = f'{d}/engrad.out'
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    i = t.find('CARTESIAN GRADIENT')
    if i < 0:
        return None
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
    return mx * 51.42208 if mx else None


def stab_entry(rx, src):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return None
    return {x['source']: x for x in json.load(open(p))['geometries']}.get(src)


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def candidate(rx, who):
    """geometry, gradient, hessian path -- or None where the structure does not
    exist."""
    if who == 'Referenz':
        g = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
        e = stab_entry(rx, 'RKS-ref')
        gr = None
        if e and e.get('ext_stable') is not None:
            gr = ((e.get('rks_grad') or {}).get('max_evang') if e['ext_stable']
                  else ((e.get('bs') or {}).get('bs_grad') or {}).get('max_evang'))
        return (g if os.path.exists(g) else None), gr, None
    if who == 'unsere':
        g = hp = None
        for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
            rp = f'{H}/{d}/{rx}/result.json'
            if os.path.exists(rp) and json.load(open(rp)).get('e_uks_final'):
                for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
                    if any(p in os.path.basename(f).lower()
                           for p in ('ts', 'final', 'opt')):
                        g = f
                        break
            if g:
                break
        for d in (f'{H}/orca_freq/ours_{rx}', f'{H}/orca_irc/{rx}_ours'):
            if os.path.exists(f'{d}/numfreq.hess'):
                return g, engrad(d), f'{d}/numfreq.hess'
        for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
            p = f'{H}/{fd}/{rx}/hessian.npy'
            if os.path.exists(p):
                j = json.load(open(f'{H}/{fd}/{rx}/result.json'))
                v = j.get('max_grad_ha_bohr')
                return g, (float(v) * 51.42208 if v is not None else None), p
        return g, None, None
    if who == 'NEB':
        gs = sorted(glob.glob(f'{H}/bs_uks_neb_results/{rx}/*NEB-TS_converged.xyz'))
        if not gs:
            return None, None, None
        d = f'{H}/orca_freq/nebts_{rx}'
        if os.path.exists(f'{d}/numfreq.hess'):
            return gs[0], engrad(d), f'{d}/numfreq.hess'
        return gs[0], None, None
    # a model
    g = f'{H}/{MODELDIR[who]}/{rx}/transition_state.xyz'
    if not os.path.exists(g):
        return None, None, None
    hp = None
    for d in (f'{H}/orca_freq/{rx}_{who}', f'{H}/orca_irc/{rx}_{who}'):
        if os.path.exists(f'{d}/numfreq.hess'):
            hp = f'{d}/numfreq.hess'
            gr = engrad(d)
            break
    else:
        p = f'{H}/freq_at_model/{rx}_{who}/hessian.npy'
        hp = p if os.path.exists(p) else None
        gr = None
    if gr is None:
        e = stab_entry(rx, who)
        if e and e.get('ext_stable') is not None:
            gr = ((e.get('rks_grad') or {}).get('max_evang') if e['ext_stable']
                  else ((e.get('bs') or {}).get('bs_grad') or {}).get('max_evang'))
    return g, gr, hp


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
    e = stab_entry(rx, 'RKS-ref')
    if e and e.get('ext_stable') is False:
        MR.append(rx)
MR.sort(key=lambda r: -nf[r])

WHO = ['Referenz', 'unsere', 'UMA-S', 'UMA-M', 'eSEN', 'NEB']
CW = 15

print('SATTELPUNKT GEFUNDEN?   19 Multireferenz-Reaktionen')
print('=' * 118)
print('ja        stationaer (Gradient < 0.15 eV/A) und genau eine imaginaere Mode')
print('n.stat.   Gradient zu gross -- kein Stationaerpunkt, also kein Sattelpunkt')
print('Minimum   stationaer, aber keine imaginaere Mode')
print('N imag    stationaer, aber N imaginaere Moden -- Sattelpunkt hoeherer Ordnung')
print('n.gepr.   Struktur existiert, aber es wurde nie eine Hesse gerechnet')
print('--        Struktur existiert nicht')
print()
def product_broken(rx):
    """Is the relaxed product spin-broken? A marker, not a column: the product
    is not a candidate for the transition state, but a reaction whose product
    sits on the wrong surface is a different object from one whose does not."""
    p = f'{H}/orca_endpoint/{rx}_product/sp.out'
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return None
    m = re.findall(r'<S\*\*2>\s*:\s*([-\d.]+)', t)
    if not m:
        return None
    v = float(m[-1])
    return v if abs(v) > 0.05 else None


print('rxn      ' + ''.join(f'{w:<{CW}}' for w in WHO) + 'gleicher Sattel?')
print('-' * 118)

for rx in MR:
    pairs = reactive(rx)
    cells, found = [], {}
    for who in WHO:
        g, gr, hp = candidate(rx, who)
        if g is None:
            cells.append('--')
            continue
        # The gradient decides before the Hessian does. A point with a large
        # residual force is not stationary and therefore not a saddle, which is
        # a result and not a gap -- reporting it as "never tested" put the
        # reference column in a state that contradicted the figure.
        if gr is not None and gr >= GRAD_OK:
            cells.append(f'n.stat. {gr:.2f}')
            continue
        if hp is None:
            cells.append(f'n.gepr. {gr:.2f}' if gr is not None else 'n.gepr.')
            continue
        sym, xyz = read_xyz(g)
        hs = read_orca_hess(hp) if hp.endswith('.hess') else np.load(hp)
        ni = n_imag_of(hs, sym, xyz)
        if gr is None:
            cells.append('? kein Grad')
        elif ni == 0:
            cells.append('Minimum')
        elif ni > 1:
            cells.append(f'{ni} imag')
        else:
            cells.append('ja')
            found[who] = xyz
    # do the ones that found a saddle agree?
    if len(found) < 2:
        cmp = '—' if len(found) == 0 else f'nur {list(found)[0]}'
    elif not pairs:
        cmp = 'keine reaktiven Bindungen notiert'
    else:
        ks = list(found)
        groups = []
        for k in ks:
            for grp in groups:
                a = found[k]; b = found[grp[0]]
                if max(abs(np.linalg.norm(a[i] - a[j])
                           - np.linalg.norm(b[i] - b[j]))
                       for i, j, _ in pairs) < SAME_BOND:
                    grp.append(k)
                    break
            else:
                groups.append([k])
        if len(groups) == 1:
            cmp = f'alle {len(ks)} gleich'
        else:
            cmp = ('VERSCHIEDEN: '
                   + ' | '.join('+'.join(g) for g in groups))
    pb = product_broken(rx)
    if pb is not None:
        cmp += f'   [Produkt spingebrochen, S2 {pb:.2f}]'
    print(f'{rx:<9}' + ''.join(f'{c:<{CW}}' for c in cells) + cmp)
