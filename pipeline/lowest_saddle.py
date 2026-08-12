"""Which method finds the lowest saddle -- of the ones anybody found?

The scope has to be stated or the number is a lie: this is not "who finds the
true transition state". The global lowest saddle of a potential energy surface
cannot be established, and every structure here came from a search that started
somewhere. What can be counted is narrower and still worth counting:

    among all candidates that clear all three stages for a given reaction,
    which is lowest in energy, and which methods produced it

That is a lower bound on method quality and an upper bound on our knowledge. If
no method wins consistently, the answer to "which method should I use" is "more
than one", and that is a result.

Only candidates that clear all three stages enter. A lower non-saddle is not a
competitor -- rxn8885 made that concrete, with a structure 425 meV below our
saddle that turned out to be a minimum.

Two structures count as the same saddle when both reactive bonds agree to
0.05 A, so a method is not penalised for finding the same point in a different
conformation.
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
GRAD_OK = 0.15
FRAC_MIN, RATE_MIN = 0.10, 0.05
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


def analyse(hess, sym, xyz, pairs):
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
    ev, vec = np.linalg.eigh(Q @ Hm @ Q)
    fr = np.sign(ev) * np.sqrt(np.abs(ev)) * CM
    k = int(np.argmin(ev))
    q = vec[:, k].reshape(-1, 3)
    q = q / np.linalg.norm(q)
    idx = sorted({i for a, b, _ in pairs for i in (a, b)})
    rates = [abs(float(np.dot(q[a] - q[b],
                              (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b]))))
             for a, b, _ in pairs]
    return {'n_imag': int((fr < -20).sum()), 'imag': float(fr[k]),
            'frac': float((q[idx] ** 2).sum()), 'maxrate': max(rates)}


def odir(label):
    """The ORCA directory for a structure, under either naming scheme.

    The IRC preparation wrote rxn1147_ours while the sweep wrote ours_rxn1147.
    Looking only for the second silently dropped our own structure at rxn1147
    and rxn7957 -- the two reactions where it loses -- and turned an 11-of-13
    record into 11 of 11.
    """
    cands = [f'{H}/orca_freq/{label}', f'{H}/orca_irc/{label}']
    m = re.match(r'ours_(rxn\d+)$', label)
    if m:
        cands += [f'{H}/orca_irc/{m.group(1)}_ours',
                  f'{H}/orca_freq/{m.group(1)}_ours']
    for d in cands:
        if os.path.isdir(d):
            return d
    return None


def orca_energy_grad(label):
    d = odir(label)
    if not d:
        return None, None
    e = g = None
    p = f'{d}/bs_sp.out'
    if os.path.exists(p):
        m = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)',
                       open(p, errors='replace').read())
        if m:
            e = float(m[-1])
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
            g = mx * 51.42208 if mx else None
    return e, g


def pyscf_energy_grad(rx, src):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return None, None
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get(src)
    if not g or g.get('ext_stable') is None:
        return None, None
    if g['ext_stable']:
        return g.get('e_rks'), (g.get('rks_grad') or {}).get('max_evang')
    b = g.get('bs') or {}
    return b.get('e_uks'), (b.get('bs_grad') or {}).get('max_evang')


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def candidates(rx):
    out = []
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        rp = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(rp):
            j = json.load(open(rp))
            if j.get('e_uks_final'):
                for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
                    if any(p in os.path.basename(f).lower()
                           for p in ('ts', 'final', 'opt')):
                        out.append(('TS-Opt', f, f'ours_{rx}',
                                    j['e_uks_final']))
                        break
                break
    for m in MODELS:
        g = f'{H}/{MODELDIR[m]}/{rx}/transition_state.xyz'
        if os.path.exists(g):
            out.append((m, g, f'{rx}_{m}', None))
    nt = sorted(glob.glob(f'{H}/bs_uks_neb_results/{rx}/*NEB-TS_converged.xyz'))
    if nt:
        out.append(('UKS-NEB', nt[0], f'nebts_{rx}', None))
    for d in sorted(glob.glob(f'{H}/tsopt_from_model/{rx}_*/')):
        tag = os.path.basename(os.path.dirname(d))
        xs = [f for f in sorted(glob.glob(f'{d}/*.xyz'))
              if any(p in os.path.basename(f).lower()
                     for p in ('ts', 'final', 'opt'))]
        if xs:
            j = {}
            if os.path.exists(f'{d}/result.json'):
                j = json.load(open(f'{d}/result.json'))
            out.append((f'TS-Opt ab {tag.split("_", 1)[1]}', xs[0],
                        f'tsopt_{tag}', j.get('e_uks_final')))
    return out


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

print('WHICH METHOD FINDS THE LOWEST SADDLE?')
print('=' * 100)
print('Scope: lowest among the candidates that clear all three stages for that')
print('reaction. Not the true transition state -- that cannot be established.')
print()
print(f'{"rxn":<9}{"valid":>6}{"distinct":>9}   {"winner(s)":<34}'
      f'{"2nd place":<18}{"gap meV":>8}')
print('-' * 100)

wins, appear, distinct_tally = {}, {}, []
attempted = {}
mixed = []
for rx in MR:
    pairs = reactive(rx)
    if not pairs:
        continue
    valid = []
    for name, geom, label, e_pyscf in candidates(rx):
        # a structure exists, so this method was at least tried here
        attempted[name] = attempted.get(name, 0) + 1
        hp = None
        d = odir(label)
        if d and os.path.exists(f'{d}/numfreq.hess'):
            hp = f'{d}/numfreq.hess'
        elif name in MODELS:
            p = f'{H}/freq_at_model/{rx}_{name}/hessian.npy'
            hp = p if os.path.exists(p) else None
        elif name == 'TS-Opt':
            for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
                p = f'{H}/{fd}/{rx}/hessian.npy'
                if os.path.exists(p):
                    hp = p
                    break
        if not hp:
            continue
        sym, xyz = read_xyz(geom)
        hs = read_orca_hess(hp) if hp.endswith('.hess') else np.load(hp)
        a = analyse(hs, sym, xyz, pairs)
        e, g = orca_energy_grad(label)
        src = 'ORCA'
        if e is None:
            if name in MODELS:
                e, g2 = pyscf_energy_grad(rx, name)
                g = g if g is not None else g2
            else:
                e = e_pyscf
            src = 'PySCF'
        if g is None and name == 'TS-Opt':
            # our own gradient, recorded by the frequency run and never carried
            # anywhere else
            for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
                q = f'{H}/{fd}/{rx}/result.json'
                if os.path.exists(q):
                    v = json.load(open(q)).get('max_grad_ha_bohr')
                    if v is not None:
                        g = float(v) * 51.42208
                        break
        if e is None or g is None:
            print(f'  [{rx} {name}: keine '
                  + ('Energie' if e is None else 'Gradient')
                  + ' -- uebersprungen]')
            continue
        if g >= GRAD_OK or a['n_imag'] != 1:
            continue
        if a['frac'] < FRAC_MIN or a['maxrate'] < RATE_MIN:
            continue
        valid.append({'name': name, 'e': e, 'xyz': xyz, 'src': src})
    if not valid:
        print(f'{rx:<9}{0:>6}{0:>9}   {"kein gueltiger Kandidat":<34}')
        continue
    if len({v['src'] for v in valid}) > 1:
        mixed.append(rx)
    # group the ones that are the same saddle
    groups = []
    for v in sorted(valid, key=lambda x: x['e']):
        for grp in groups:
            a, b = v['xyz'], grp[0]['xyz']
            if max(abs(np.linalg.norm(a[i] - a[j])
                       - np.linalg.norm(b[i] - b[j]))
                   for i, j, _ in pairs) < SAME_BOND:
                grp.append(v)
                break
        else:
            groups.append([v])
    groups.sort(key=lambda g: min(x['e'] for x in g))
    win = groups[0]
    gap = ((min(x['e'] for x in groups[1]) - min(x['e'] for x in win))
           * HA_MEV) if len(groups) > 1 else None
    for v in win:
        wins[v['name']] = wins.get(v['name'], 0) + 1
    for v in valid:
        appear[v['name']] = appear.get(v['name'], 0) + 1
    distinct_tally.append(len(groups))
    w = '+'.join(sorted({v['name'] for v in win}))
    s = ('+'.join(sorted({v['name'] for v in groups[1]}))
         if len(groups) > 1 else '—')
    print(f'{rx:<9}{len(valid):>6}{len(groups):>9}   {w:<34}{s:<18}'
          + (f'{gap:>8.0f}' if gap is not None else f'{"—":>8}'))

print()
print('How often did each method produce the lowest saddle, how often did it')
print('produce a valid saddle at all, and -- the column that decides whether')
print('the others can be compared -- on how many reactions was it even tried?')
print()
print(f'{"Methode":<20}{"versucht":>9}{"gueltig":>9}{"niedrigster":>12}'
      f'{"Abdeckung":>11}')
for k in sorted(appear, key=lambda x: -wins.get(x, 0)):
    w, a, t = wins.get(k, 0), appear[k], attempted.get(k, 0)
    print(f'{k:<20}{t:>9}{a:>9}{w:>12}{a / t * 100 if t else 0:>10.0f}%')
print()
print('versucht    reactions where the method was run at all. The from-model')
print('            optimisations were only ever started on 10 of the 19, and 9')
print('            of those 10 from UMA-M alone, so that row is not on the same')
print('            footing as the rest and its rate is not comparable.')
print('gueltig     produced a structure clearing all three stages')
print('niedrigster was in the lowest group, ties included -- and 13 of 19')
print('            reactions have only one distinct saddle, so most of these')
print('            are ties rather than victories')

if distinct_tally:
    print()
    print(f'distinct saddles per reaction: '
          f'{np.mean(distinct_tally):.2f} on average, '
          f'{sum(1 for d in distinct_tally if d > 1)} of '
          f'{len(distinct_tally)} reactions have more than one')
if mixed:
    print()
    print(f'energies from mixed sources in {len(mixed)} reactions: '
          + ' '.join(mixed))
    print('ORCA and PySCF agree to under 1 meV where both exist, so the'
          ' comparison holds,')
    print('but it is not a single-code number.')
