"""One figure: which method found which saddle, and why the others did not.

One panel, one row per reaction, one cell per method. The cell says what
happened, and its height says where that saddle sits:

  low      this method found the lowest saddle of that reaction
  high     it found a valid saddle, but a higher one; the number above the cell
           says by how much, in meV
  centre   it found no saddle at all -- no energy, so no height

A failure is not a blank. "Not stationary", "two imaginary modes" and "never
computed" are different outcomes and get different cells, because collapsing
them is what let untested candidates read as refuted ones.

Design notes, since they were choices:

  The two greens are one hue at two steps, an ordinal pair: "lowest" and
  "higher" are ordered states of one outcome, not two outcomes. Height and the
  printed number carry the same distinction, so nothing rests on telling the
  greens apart.

  Height is schematic. Drawing +20 meV and +1214 meV proportionally would make
  the first invisible; drawing them equally would misstate the size. Height
  carries the order, the number carries the size.

  An earlier version put a separate energy panel beside the cells. It said the
  same thing twice.
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ase.data import atomic_masses, atomic_numbers

H = '/home/energy/s242862'
HA_MEV = 27211.386
BOHR = 0.529177210903
CM = 5140.4871
GRAD_OK = 0.15
FRAC_MIN, RATE_MIN = 0.10, 0.05
SAME_BOND = 0.05

# surface and ink
SURF = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
INK3 = '#8a8985'
GRID = '#e6e5e1'
# Status palette, fixed and distinct from any series hue. The two greens are one
# hue at two steps: an ordinal pair, since "lowest saddle" and "a saddle but a
# higher one" are ordered states of the same outcome, not different outcomes.
# Height and the printed offset carry the same distinction, so nothing rests on
# the two greens being told apart.
ST = {'low': '#076b07', 'ok': '#0ca30c', 'grad': '#d03b3b',
      'nimag': '#ec835a', 'mode': '#fab219', 'none': '#dedcd6'}

METHODS = [('Reference', 'R'), ('TS-Opt', 'T'), ('UMA-S', 'S'),
           ('UMA-M', 'M'), ('eSEN', 'E'), ('UKS-NEB', 'N')]
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}
LETTER = dict(METHODS)


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
    return {'n_imag': int((fr < -20).sum()),
            'frac': float((q[idx] ** 2).sum()), 'maxrate': max(rates)}


def odir(label):
    cands = [f'{H}/orca_freq/{label}', f'{H}/orca_irc/{label}']
    m = re.match(r'ours_(rxn\d+)$', label)
    if m:
        cands += [f'{H}/orca_irc/{m.group(1)}_ours']
    for d in cands:
        if os.path.isdir(d):
            return d
    return None


def orca_eg(label):
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


def stab(rx, src):
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


def collect(rx):
    """Per method: geometry, energy, gradient, Hessian -- or what is missing."""
    out = {}
    g = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
    e, gr = stab(rx, 'RKS-ref')
    out['Reference'] = {'geom': g if os.path.exists(g) else None,
                        'e': e, 'grad': gr, 'hess': None}
    geom = hp = e = None
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        rp = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(rp):
            j = json.load(open(rp))
            if j.get('e_uks_final'):
                for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
                    if any(p in os.path.basename(f).lower()
                           for p in ('ts', 'final', 'opt')):
                        geom, e = f, j['e_uks_final']
                        break
                break
    eo, go = orca_eg(f'ours_{rx}')
    d = odir(f'ours_{rx}')
    if d and os.path.exists(f'{d}/numfreq.hess'):
        hp = f'{d}/numfreq.hess'
    if hp is None:
        for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
            p = f'{H}/{fd}/{rx}/hessian.npy'
            if os.path.exists(p):
                hp = p
                q = json.load(open(f'{H}/{fd}/{rx}/result.json'))
                if go is None and q.get('max_grad_ha_bohr') is not None:
                    go = float(q['max_grad_ha_bohr']) * 51.42208
                break
    out['TS-Opt'] = {'geom': geom, 'e': eo if eo is not None else e,
                     'grad': go, 'hess': hp}
    for m in ('UMA-S', 'UMA-M', 'eSEN'):
        g = f'{H}/{MODELDIR[m]}/{rx}/transition_state.xyz'
        e, gr = orca_eg(f'{rx}_{m}')
        if e is None:
            e, gr2 = stab(rx, m)
            gr = gr if gr is not None else gr2
        d = odir(f'{rx}_{m}')
        hp = (f'{d}/numfreq.hess' if d and os.path.exists(f'{d}/numfreq.hess')
              else (f'{H}/freq_at_model/{rx}_{m}/hessian.npy'
                    if os.path.exists(f'{H}/freq_at_model/{rx}_{m}/hessian.npy')
                    else None))
        out[m] = {'geom': g if os.path.exists(g) else None, 'e': e,
                  'grad': gr, 'hess': hp}
    nt = sorted(glob.glob(f'{H}/bs_uks_neb_results/{rx}/*NEB-TS_converged.xyz'))
    e, gr = orca_eg(f'nebts_{rx}')
    d = odir(f'nebts_{rx}')
    out['UKS-NEB'] = {'geom': nt[0] if nt else None, 'e': e, 'grad': gr,
                      'hess': (f'{d}/numfreq.hess'
                               if d and os.path.exists(f'{d}/numfreq.hess')
                               else None)}
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

rows = []
for rx in MR:
    pairs = reactive(rx)
    info = collect(rx)
    state, valid = {}, []
    for name, _ in METHODS:
        d = info[name]
        if not d['geom']:
            state[name] = 'none'
            continue
        if d['grad'] is None:
            state[name] = 'none'
            continue
        if d['grad'] >= GRAD_OK:
            state[name] = 'grad'
            continue
        if not d['hess']:
            state[name] = 'none'
            continue
        try:
            sym, xyz = read_xyz(d['geom'])
            hs = (read_orca_hess(d['hess']) if d['hess'].endswith('.hess')
                  else np.load(d['hess']))
            a = analyse(hs, sym, xyz, pairs) if pairs else None
        except Exception:
            state[name] = 'none'
            continue
        if a is None or a['n_imag'] != 1:
            state[name] = 'nimag'
            continue
        if a['frac'] < FRAC_MIN or a['maxrate'] < RATE_MIN:
            state[name] = 'mode'
            continue
        if d['e'] is None:
            state[name] = 'none'
            continue
        state[name] = 'ok'
        valid.append({'name': name, 'e': d['e'], 'xyz': xyz})
    groups = []
    for v in sorted(valid, key=lambda x: x['e']):
        for grp in groups:
            a, b = v['xyz'], grp[0]['xyz']
            if pairs and max(abs(np.linalg.norm(a[i] - a[j])
                                 - np.linalg.norm(b[i] - b[j]))
                             for i, j, _ in pairs) < SAME_BOND:
                grp.append(v)
                break
        else:
            groups.append([v])
    groups.sort(key=lambda g: min(x['e'] for x in g))
    e0 = min(x['e'] for x in valid) if valid else None
    levels = [(min(x['e'] for x in g) - e0) * HA_MEV
              if e0 is not None else 0.0 for g in groups]
    # which level each method sits on, so the cell itself can carry it
    lvl = {}
    for k, grp in enumerate(groups):
        for v in grp:
            lvl[v['name']] = k
    for name in lvl:
        state[name] = 'low' if lvl[name] == 0 else 'ok'
    rows.append({'rx': rx, 'nfod': nf[rx], 'state': state, 'level': lvl,
                 'groups': groups, 'levels': levels})
# ------------------------------------------------------------------ figure
NR = len(rows)
# One panel. The cells carry the energy themselves now: a cell sits lower when
# its method found the lowest saddle of that reaction, higher when it found a
# valid one that is not the lowest, and at the row centre when it found no
# saddle at all -- height only exists where an energy exists. A separate energy
# panel beside this said the same thing twice.
fig = plt.figure(figsize=(9.6, 0.60 * NR + 3.4), facecolor=SURF)
axS = fig.add_axes([0.175, 0.100, 0.780, 0.740])
axS.set_facecolor(SURF)
for sp in axS.spines.values():
    sp.set_visible(False)

# Schematic offset: drawing +20 meV and +1214 meV at proportional heights would
# make the first invisible, and drawing them equal would be a lie about size.
# So height carries the order and the printed number carries the size.
DY, CH = 0.155, 0.215
GLYPH = {'low': '✓', 'ok': '✓', 'grad': 'g', 'nimag': 'n', 'mode': 'm',
         'none': '·'}

axS.set_xlim(-0.5, len(METHODS) - 0.5)
axS.set_ylim(NR - 0.5, -0.5)
axS.set_xticks(range(len(METHODS)))
axS.set_xticklabels([m for m, _ in METHODS], fontsize=9.5, color=INK,
                    rotation=34, ha='left')
axS.xaxis.set_ticks_position('top')
axS.tick_params(axis='x', length=0, pad=5)
axS.set_yticks(range(NR))
axS.set_yticklabels([f'{r["rx"]}   {r["nfod"]:.2f}' for r in rows],
                    fontsize=9.5, color=INK, family='DejaVu Sans')
axS.tick_params(axis='y', length=0, pad=8)
for i in range(1, NR):
    axS.axhline(i - 0.5, color=GRID, lw=0.7, zorder=0)

for i, r in enumerate(rows):
    nlev = len(r['groups'])
    for j, (name, _) in enumerate(METHODS):
        s = r['state'].get(name, 'none')
        # Only a higher saddle leaves the line. Lowering the lowest as well
        # made the thirteen uncontested rows ragged for no reason, and put the
        # failure cells -- which have no energy at all -- at an apparent height
        # between the two.
        k = r.get('level', {}).get(name)
        y = i if not k else i - k * 2 * DY
        axS.add_patch(Rectangle((j - 0.42, y - CH), 0.84, 2 * CH,
                                facecolor=ST[s], edgecolor=SURF, lw=1.8,
                                zorder=2))
        axS.text(j, y, GLYPH[s], fontsize=9.0, ha='center', va='center',
                 color='white' if s != 'none' else INK3, zorder=3,
                 weight='bold')
        if k is not None and k > 0:
            axS.text(j, y + CH + 0.115, f'+{r["levels"][k]:.0f}',
                     fontsize=8.2, ha='center', va='center', color=INK2,
                     zorder=3, family='DejaVu Sans')

fig.text(0.045, 0.968, 'Which saddle, found by whom — and why not',
         fontsize=16.5, color=INK, weight='bold', ha='left')
for k, line in enumerate([
        'The 19 reactions whose restricted reference solution is externally '
        'unstable. Valid means: stationary (gradient < 0.15 eV/Å),',
        'exactly one imaginary mode, and that mode moves this reaction’s '
        'bonds. A cell is raised only when its method found a valid saddle',
        'that is not the lowest one for that reaction; the number above it '
        'says by how much, in meV. Everything else sits on the line.']):
    fig.text(0.045, 0.940 - 0.0172 * k, line, fontsize=9.5, color=INK2,
             ha='left')

fig.text(0.045, 0.878,
         'Thirteen of the nineteen have every successful method on one level — '
         'there the methods that find a saddle find the same one.',
         fontsize=9.3, color=INK3, ha='left')

items = [[('low', 'lowest saddle found'),
          ('ok', 'a valid saddle, but higher'),
          ('grad', 'not stationary')],
         [('nimag', 'wrong number of imaginary modes'),
          ('mode', 'mode does not belong to this reaction'),
          ('none', 'no structure, or no Hessian computed')]]
for r_i, row_items in enumerate(items):
    x0 = 0.045
    y = 0.042 - 0.022 * r_i
    for key, lab in row_items:
        fig.patches.append(Rectangle((x0, y), 0.0135, 0.0125,
                                     transform=fig.transFigure,
                                     facecolor=ST[key], edgecolor=SURF,
                                     lw=1.2))
        fig.text(x0 + 0.0165, y + 0.0014, f'{GLYPH[key]}  {lab}', fontsize=9,
                 color=INK2, ha='left')
        x0 += 0.0235 + 0.0086 * len(lab)

fig.savefig(f'{H}/saddle_landscape.png', dpi=200, facecolor=SURF)
print('written saddle_landscape.png')
print(f'{NR} reactions, '
      f'{sum(1 for r in rows if len(r["groups"]) > 1)} with more than one '
      f'saddle, {sum(1 for r in rows if not r["groups"])} with none')
