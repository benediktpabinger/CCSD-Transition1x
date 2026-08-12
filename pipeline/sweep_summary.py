"""The two questions the sweep was run to answer.

1. Do the models sit on saddle points at all -- even wrong ones? Reference-free:
   it needs no opinion about which structure is correct, only the gradient and
   the Hessian at the geometry the model predicted. The single-reference group
   is the control; without it the multireference number means nothing.

2. Which candidate holds the transition state of each multireference reaction,
   now that every candidate has been tested rather than a selected few.

A structure counts as a first-order saddle only if it is stationary as well.
n_imag at a point with a large residual force says nothing about transition
states, so the two conditions are reported together and never separately.
"""
import glob
import json
import os
import re

import numpy as np
from ase.data import atomic_masses, atomic_numbers

H = '/home/energy/s242862'
HA_MEV = 27211.386
BOHR = 0.529177210903
CM = 5140.4871
MODELS = ('UMA-S', 'UMA-M', 'eSEN')
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}
GRAD_OK = 0.15          # above this the point is not stationary
FRAC_MIN, RATE_MIN = 0.10, 0.05


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


def trans_rot(msqrt, xyz_bohr):
    nat = len(msqrt)
    w2 = msqrt ** 2
    c = xyz_bohr - (xyz_bohr * w2[:, None]).sum(0) / w2.sum()
    B = []
    for k in range(3):
        v = np.zeros((nat, 3)); v[:, k] = msqrt
        B.append(v.ravel())
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        B.append((np.cross(np.tile(e, (nat, 1)), c) * msqrt[:, None]).ravel())
    U, s, _ = np.linalg.svd(np.array(B).T, full_matrices=False)
    return U[:, s > 1e-8]


def analyse(hess, sym, xyz, pairs):
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    msqrt = np.sqrt(m)
    w = np.repeat(1.0 / msqrt, 3)
    Hm = hess * w[:, None] * w[None, :]
    P = trans_rot(msqrt, xyz / BOHR)
    Q = np.eye(len(Hm)) - P @ P.T
    ev, vec = np.linalg.eigh(Q @ Hm @ Q)
    fr = np.sign(ev) * np.sqrt(np.abs(ev)) * CM
    k = int(np.argmin(ev))
    q = vec[:, k].reshape(-1, 3)
    q = q / np.linalg.norm(q)
    out = {'n_imag': int((fr < -20).sum()), 'imag': float(fr[k])}
    if pairs:
        idx = sorted({i for a, b, _ in pairs for i in (a, b)})
        bonds = [(nm, abs(float(np.dot(q[a] - q[b],
                  (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b])))),
                  float(np.linalg.norm(xyz[a] - xyz[b]))) for a, b, nm in pairs]
        out.update({'frac': float((q[idx] ** 2).sum()), 'bonds': bonds,
                    'maxrate': max(b[1] for b in bonds)})
    return out


def orca(label):
    for d in (f'{H}/orca_freq/{label}', f'{H}/orca_irc/{label}'):
        if os.path.isdir(d) and os.path.exists(f'{d}/numfreq.hess'):
            g = None
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
                    g = mx * 51.42208
            e = None
            p = f'{d}/bs_sp.out'
            if os.path.exists(p):
                m = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)',
                               open(p, errors='replace').read())
                if m:
                    e = float(m[-1])
            return {'hess': f'{d}/numfreq.hess', 'grad': g, 'e': e}
    return None


def stab_grad(rx, m):
    """The gradient from the stability pipeline, on whichever surface is the
    ground state there.

    Needed as a fallback: the fifteen model structures whose Hessian came from
    PySCF have no ORCA engrad, and treating a missing gradient as a failed
    stage 1 silently discarded thirteen candidates.
    """
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return None
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get(m)
    if not g or g.get('ext_stable') is None:
        return None
    if g['ext_stable']:
        return (g.get('rks_grad') or {}).get('max_evang')
    return ((g.get('bs') or {}).get('bs_grad') or {}).get('max_evang')


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

print('=' * 78)
print('1  ARE THE MODEL PREDICTIONS SADDLE POINTS AT ALL?')
print('=' * 78)
print('A structure counts only if it is stationary (gradient < '
      f'{GRAD_OK} eV/A) and')
print('has exactly one imaginary mode. Nothing here needs a reference.')
print()
print(f'{"":22}{"tested":>8}{"stationary":>12}{"+1 imag":>10}{"= saddle":>10}'
      f'{"share":>8}')

rows = {}
for cls, RXS in (('single-reference', SIMPLE), ('multireference', MR)):
    tested = stat = one = saddle = 0
    detail = []
    for rx in sorted(RXS, key=lambda r: -nf[r]):
        pairs = reactive(rx)
        for m in MODELS:
            lbl = f'{rx}_{m}'
            o = orca(lbl)
            if not o:
                p = f'{H}/freq_at_model/{lbl}/hessian.npy'
                if not os.path.exists(p):
                    continue
                o = {'hess': p, 'grad': None, 'e': None}
            g = f'{H}/{MODELDIR[m]}/{rx}/transition_state.xyz'
            if not os.path.exists(g):
                continue
            sym, xyz = read_xyz(g)
            hs = (read_orca_hess(o['hess']) if o['hess'].endswith('.hess')
                  else np.load(o['hess']))
            a = analyse(hs, sym, xyz, pairs)
            if o['grad'] is None:
                o['grad'] = stab_grad(rx, m)
            tested += 1
            st = o['grad'] is not None and o['grad'] < GRAD_OK
            im = a['n_imag'] == 1
            stat += st; one += im
            if st and im:
                saddle += 1
            detail.append((rx, m, o['grad'], a['n_imag'], a['imag'],
                           a.get('frac'), a.get('maxrate')))
    rows[cls] = detail
    print(f'{cls:<22}{tested:>8}{stat:>12}{one:>10}{saddle:>10}'
          f'{saddle / tested * 100 if tested else 0:>7.0f}%')

print()
print('The control group is the point: the same models, the same procedure, on')
print('reactions where the restricted reference is valid.')

print()
print('=' * 78)
print('2  DOES THE SADDLE BELONG TO THE REACTION?  (multireference only)')
print('=' * 78)
print(f'stage 3 needs mode fraction >= {FRAC_MIN} and bond rate >= {RATE_MIN}')
print()
print(f'{"rxn":<9}{"model":<7}{"grad":>7}{"n_imag":>8}{"imag":>10}'
      f'{"frac":>7}{"rate":>7}   verdict')
ok3 = 0
for rx, m, g, ni, im, fr, rt in rows['multireference']:
    if g is None:
        v = 'no gradient'
    elif g >= GRAD_OK:
        v = 'not stationary'
    elif ni != 1:
        v = f'{ni} imaginary modes'
    elif fr is None:
        v = 'no reactive bonds recorded'
    elif fr >= FRAC_MIN and rt >= RATE_MIN:
        v = '*** CLEARS ALL THREE STAGES ***'
        ok3 += 1
    else:
        v = 'mode does not belong to this reaction'
    print(f'{rx:<9}{m:<7}{g if g is not None else float("nan"):7.3f}{ni:>8}'
          f'{im:>10.1f}{fr if fr is not None else float("nan"):7.2f}'
          f'{rt if rt is not None else float("nan"):7.3f}   {v}')
print(f'\n{ok3} model geometries clear all three stages')

print()
print('=' * 78)
print('3  THE OTHER CANDIDATES: NEB-TS AND THE FROM-MODEL OPTIMISATIONS')
print('=' * 78)
print(f'{"structure":<24}{"grad":>7}{"n_imag":>8}{"imag":>10}{"frac":>7}'
      f'{"rate":>7}   verdict')
for pre, pat in (('nebts_', f'{H}/bs_uks_neb_results/{{}}/*NEB-TS_converged.xyz'),
                 ('tsopt_', None)):
    for d in sorted(glob.glob(f'{H}/orca_freq/{pre}*/')):
        lbl = os.path.basename(os.path.dirname(d))
        if not os.path.exists(f'{d}/numfreq.hess'):
            continue
        rx = re.search(r'rxn\d+', lbl).group(0)
        gp = f'{d}/start.xyz'
        if not os.path.exists(gp):
            continue
        sym, xyz = read_xyz(gp)
        pairs = reactive(rx)
        a = analyse(read_orca_hess(f'{d}/numfreq.hess'), sym, xyz, pairs)
        o = orca(lbl)
        g = o['grad'] if o else None
        if g is None:
            v = 'no gradient'
        elif g >= GRAD_OK:
            v = 'not stationary'
        elif a['n_imag'] == 0:
            v = 'MINIMUM, not a saddle'
        elif a['n_imag'] != 1:
            v = f'{a["n_imag"]} imaginary modes'
        elif a.get('frac') is None:
            v = 'no reactive bonds recorded'
        elif a['frac'] >= FRAC_MIN and a['maxrate'] >= RATE_MIN:
            v = '*** CLEARS ALL THREE STAGES ***'
        else:
            v = 'mode does not belong to this reaction'
        print(f'{lbl:<24}{g if g is not None else float("nan"):7.3f}'
              f'{a["n_imag"]:>8}{a["imag"]:>10.1f}'
              f'{a.get("frac", float("nan")):7.2f}'
              f'{a.get("maxrate", float("nan")):7.3f}   {v}')
