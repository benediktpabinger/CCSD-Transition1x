"""One figure: which method found which saddle, and why the others did not.

Two coupled panels sharing a row per reaction.

  left    the energy of every distinct saddle found for that reaction, relative
          to the lowest one. Methods that landed on the same saddle sit on the
          same level, so agreement is a cluster and disagreement is a vertical
          gap you can measure against the axis.
  right   one cell per method: what happened. A failure is not a blank -- "not
          stationary" and "two imaginary modes" and "never computed" are
          different outcomes and get different cells.

Together they make the summary counts checkable: every number in the tables can
be recounted off this figure.

Design notes, because they were choices:

  No categorical colour for the methods. Six hues in a scatter-like form cannot
  clear the colourblind separation floors, and identity is already carried by
  the letter in each marker. Colour is spent on emphasis instead -- the lowest
  saddle of each reaction against the ones above it -- which is what the figure
  is about.

  The state cells use the status palette only, kept distinct from any series
  hue, and each carries a glyph as well, so nothing depends on colour alone.

  A symmetric-log energy axis. The gaps run from 20 to 1214 meV and most
  reactions sit at exactly zero; a linear axis would put everything on the
  baseline and a plain log axis cannot show the zero.
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
# emphasis: the lowest saddle against the rest
LOW = '#2a78d6'
HIGH = '#9a9995'
# status palette, fixed, distinct from any series hue
ST = {'ok': '#0ca30c', 'grad': '#d03b3b', 'nimag': '#ec835a',
      'mode': '#fab219', 'none': '#dedcd6'}

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
    rows.append({'rx': rx, 'nfod': nf[rx], 'state': state,
                 'groups': groups, 'levels': levels})

# ------------------------------------------------------------------ figure
NR = len(rows)
# The energy panel is deliberately narrower than the status panel: thirteen of
# the nineteen reactions have every method on one level at zero, so most of its
# width would be empty. The emptiness is the message, but it does not need
# sixty percent of the figure to make it.
fig = plt.figure(figsize=(11.6, 0.60 * NR + 3.6), facecolor=SURF)
axE = fig.add_axes([0.150, 0.095, 0.245, 0.760])
axS = fig.add_axes([0.500, 0.095, 0.360, 0.760], sharey=axE)

# The energy is drawn as a staircase, not on a scale. A logarithmic axis was
# hard to read a position off, and thirteen of the nineteen reactions sit at
# exactly zero, so most of its width carried nothing. Height now means only
# "higher", and the amount is written next to the point where it applies.
STEP_X, STEP_Y = 1.0, 0.21
for ax in (axE, axS):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_visible(False)
axE.set_xlim(-0.35, 2.5)
axE.set_ylim(NR - 0.5, -0.5)
axE.set_yticks(range(NR))
axE.set_yticklabels([f'{r["rx"]}   {r["nfod"]:.2f}' for r in rows],
                    fontsize=9.5, color=INK, family='DejaVu Sans')
axE.tick_params(axis='y', length=0, pad=6)
axE.set_xticks([])
for i in range(NR):
    axE.axhline(i, color=GRID, lw=0.7, zorder=0)

for i, r in enumerate(rows):
    if not r['groups']:
        axE.text(0.0, i, 'kein gültiger Sattelpunkt', fontsize=9,
                 color=INK3, va='center', style='italic')
        continue
    # a staircase: every further saddle one step up and to the right
    pts = [(k * STEP_X, i - k * STEP_Y) for k in range(len(r['groups']))]
    if len(pts) > 1:
        px, py = zip(*pts)
        axE.plot(px, py, color=GRID, lw=1.6, zorder=1, solid_capstyle='round')
    for k, (grp, (x, y)) in enumerate(zip(r['groups'], pts)):
        col = LOW if k == 0 else HIGH
        # in the column order of the panel on the right, so the eye can move
        # between the two halves without re-sorting
        order = [m for m, _ in METHODS]
        names = sorted((v['name'] for v in grp), key=order.index)
        # One marker per saddle, not per method: four stacked markers inside a
        # row overlap at any readable marker size, and the level is the object
        # the figure is about. Who found it goes beside it as letters.
        axE.scatter([x], [y], s=112, marker='o', facecolor=col,
                    edgecolor=SURF, linewidth=1.8, zorder=4)
        axE.annotate(' '.join(LETTER[n] for n in names),
                     xy=(x, y), xytext=(10, 0), textcoords='offset points',
                     fontsize=8.6, color=INK if k == 0 else INK2,
                     va='center', ha='left', zorder=5,
                     weight='bold' if k == 0 else 'normal',
                     family='DejaVu Sans')
        if k > 0:
            axE.annotate(f'+{r["levels"][k]:.0f} meV', xy=(x, y),
                         xytext=(0, 12), textcoords='offset points',
                         fontsize=8.2, color=INK2, va='center', ha='center',
                         family='DejaVu Sans')


GLYPH = {'ok': '✓', 'grad': 'g', 'nimag': 'n', 'mode': 'm', 'none': '·'}
axS.set_xlim(-0.5, len(METHODS) - 0.5)
axS.set_xticks(range(len(METHODS)))
axS.set_xticklabels([m for m, _ in METHODS], fontsize=9, color=INK,
                    rotation=38, ha='left')
axS.xaxis.set_ticks_position('top')
axS.xaxis.set_label_position('top')
axS.tick_params(axis='x', length=0, pad=4)
axS.tick_params(axis='y', length=0, labelleft=False)
for i, r in enumerate(rows):
    for j, (name, _) in enumerate(METHODS):
        s = r['state'].get(name, 'none')
        axS.add_patch(Rectangle((j - 0.42, i - 0.36), 0.84, 0.72,
                                facecolor=ST[s], edgecolor=SURF, lw=1.6,
                                zorder=2))
        axS.text(j, i, GLYPH[s], fontsize=8.6, ha='center', va='center',
                 color='white' if s != 'none' else INK3, zorder=3,
                 weight='bold')

fig.text(0.075, 0.968, 'Which saddle, found by whom — and why not',
         fontsize=16.5, color=INK, weight='bold', ha='left')
for k, line in enumerate([
        'The 19 reactions whose restricted reference solution is externally '
        'unstable. Each point on the left is a distinct saddle and the letters',
        'beside it name the methods that landed on it. Valid means: '
        'stationary (gradient < 0.15 eV/Å), exactly one imaginary mode, and '
        'that mode',
        'moves this reaction’s bonds. Height is schematic — it means only '
        '"higher"; the amount is written next to the point it applies to.']):
    fig.text(0.075, 0.945 - 0.0165 * k, line, fontsize=9.5, color=INK2,
             ha='left')

fig.text(0.075, 0.888,
         'T  TS-Opt (ours)    S  UMA-S    M  UMA-M    E  eSEN    N  UKS-NEB'
         '          (the reference is never a valid saddle, so it appears '
         'only on the right)',
         fontsize=9, color=INK3, ha='left')

leg = [Line2D([], [], marker='o', ls='', markersize=8.5, markerfacecolor=LOW,
              markeredgecolor=SURF, label='lowest saddle found'),
       Line2D([], [], marker='o', ls='', markersize=8.5, markerfacecolor=HIGH,
              markeredgecolor=SURF, label='higher saddle')]
fig.legend(handles=leg, loc='lower left', bbox_to_anchor=(0.073, 0.862),
           bbox_transform=fig.transFigure, ncol=2, frameon=False,
           fontsize=9.2, handletextpad=0.4, columnspacing=2.4)

# Two rows, so the longest label cannot run off the right edge.
items = [[('ok', 'valid saddle'),
          ('grad', 'not stationary'),
          ('nimag', 'wrong number of imaginary modes')],
         [('mode', 'mode does not belong to this reaction'),
          ('none', 'no structure, or no Hessian computed')]]
for r_i, row_items in enumerate(items):
    x0 = 0.075
    y = 0.040 - 0.021 * r_i
    for key, lab in row_items:
        fig.patches.append(Rectangle((x0, y), 0.0125, 0.0115,
                                     transform=fig.transFigure,
                                     facecolor=ST[key], edgecolor=SURF,
                                     lw=1.2))
        fig.text(x0 + 0.0150, y + 0.0012, f'{GLYPH[key]}  {lab}', fontsize=9,
                 color=INK2, ha='left')
        x0 += 0.0205 + 0.0071 * len(lab)

fig.savefig(f'{H}/saddle_landscape.png', dpi=200, facecolor=SURF)
print('written saddle_landscape.png')
print(f'{NR} reactions, '
      f'{sum(1 for r in rows if len(r["groups"]) > 1)} with more than one '
      f'saddle, {sum(1 for r in rows if not r["groups"])} with none')
