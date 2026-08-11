"""Reliability table as a figure: who holds the transition state, per reaction.

Grouped by outcome rather than sorted by number, because the reason a row lands
where it does is the content. Each row carries the evidence that decided it, so
the table can be read without the surrounding text.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ase.data import atomic_masses, atomic_numbers

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


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1]) for e in rb[:2]]
    return []


def mode_stats(hp, gp, pairs):
    if not (hp and gp and pairs and os.path.exists(hp) and os.path.exists(gp)):
        return None
    hess = np.load(hp)
    sym, xyz = read_xyz(gp)
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    w = np.repeat(1.0 / np.sqrt(m), 3)
    ev, vec = np.linalg.eigh(hess * w[:, None] * w[None, :])
    q = vec[:, int(np.argmin(ev))].reshape(-1, 3)
    q = q / np.linalg.norm(q)
    idx = sorted({i for a, b in pairs for i in (a, b)})
    rates = []
    for a, b in pairs:
        u = (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b])
        rates.append(abs(float(np.dot(q[a] - q[b], u))))
    return {'frac': float((q[idx] ** 2).sum()), 'maxrate': max(rates)}


def ours(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        rp = f'{H}/{d}/{rx}/result.json'
        if not os.path.exists(rp):
            continue
        j = json.load(open(rp))
        if j.get('e_uks_final') is None:
            continue
        g = None
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    g = f
                    break
            if g:
                break
        hp = ni = None
        for fd in ('bs_freq', 'bs_freq_v2', 'bs_freq_fromneb'):
            q = f'{H}/{fd}/{rx}/result.json'
            if os.path.exists(q):
                jj = json.load(open(q))
                if 'n_imag' in jj:
                    ni, hp = jj['n_imag'], f'{H}/{fd}/{rx}/hessian.npy'
        return {'e': j['e_uks_final'], 'geom': g, 'hess': hp, 'nimag': ni}
    return None


VERDICT = {
    'rxn7949': ('OURS', 'rival mode rates 0.003-0.011; both bonds already settled'),
    'rxn8832': ('OURS', 'no model lower (closest +12 meV)'),
    'rxn4113': ('OURS', 'UMA-M lower but not stationary (grad 0.185)'),
    'rxn8885': ('OURS', 'UMA-S lower but not stationary (grad 0.484)'),
    'rxn6196': ('OURS', 'no model lower (closest +8 meV)'),
    'rxn0346': ('OURS', 'no model lower (closest +2 meV)'),
    'rxn3107': ('OURS', 'no model lower (closest +1 meV)'),
    'rxn8837': ('OURS', 'models +1034 to +5469 meV, none stationary'),
    'rxn7060': ('OURS', 'no model lower (closest +0 meV)'),
    'rxn8827': ('OURS', 'no model lower (closest +20 meV)'),
    'rxn1147': ('OURS', 'models past the TS: C1-O5 already 1.497 A, rate 0.06'),
    'rxn0894': ('OURS', 'models +68 / +319 meV, neither stationary'),
    'rxn5691': ('MODELS', 'our mode dead (rate 0.014); UMA-S rate 0.53, -164 meV'),
    'rxn4522': ('MODELS', 'no saddle of ours; all three models valid, -1845 meV'),
    'rxn7957': ('MODELS', 'we are past the TS: C5-H7 already 1.120 A, rate 0.06'),
    'rxn1320': ('NEITHER', 'our mode fraction 0.00; no model candidate'),
    'rxn4518': ('NEITHER', 'our mode fraction 0.03; no model candidate'),
    'rxn1283': ('NEITHER', 'our optimisation never converged; no model candidate'),
    'rxn5690': ('NEITHER', 'dE_BS only -1.3 meV, not a multireference case'),
}

rows = []
for rx, (v, why) in VERDICT.items():
    p = f'{H}/stab_pipeline/{rx}/result.json'
    d = json.load(open(p))
    geo = {g['source']: g for g in d['geometries']}
    pairs = reactive(rx)
    o = ours(rx)
    oms = mode_stats(o['hess'], o['geom'], pairs) if o else None
    best = None
    for m in MODELS:
        g = geo.get(m)
        if not g or g.get('ext_stable') is None or not o:
            continue
        e = (g.get('e_rks') if g['ext_stable']
             else (g.get('bs') or {}).get('e_uks'))
        gr = ((g.get('rks_grad') or {}).get('max_evang') if g['ext_stable']
              else ((g.get('bs') or {}).get('bs_grad') or {}).get('max_evang'))
        if e is None:
            continue
        de = (e - o['e']) * HA_MEV
        if best is None or de < best[1]:
            best = (m, de, gr)
    rows.append({'rxn': rx, 'nfod': nf[rx], 'v': v, 'why': why,
                 'ni': (o or {}).get('nimag'), 'oms': oms, 'best': best})

ORD = {'OURS': 0, 'MODELS': 1, 'NEITHER': 2}
rows.sort(key=lambda r: (ORD[r['v']], -r['nfod']))

TITLE = {'OURS': 'Our broken-symmetry saddle is the better answer',
         'MODELS': 'The model prediction is the better answer',
         'NEITHER': 'Neither side survives the test'}
COL = {'OURS': '#4878a8', 'MODELS': '#c0553b', 'NEITHER': '#8a8a8a'}

fig = plt.figure(figsize=(15.5, 11.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

X = {'rxn': 0.035, 'nfod': 0.115, 'ni': 0.175, 'frac': 0.235, 'rate': 0.305,
     'model': 0.385, 'de': 0.475, 'grad': 0.555, 'why': 0.615}
y = 0.945
ax.text(0.035, 0.975, 'Which side holds the transition state',
        fontsize=19, weight='bold', va='center')
ax.text(0.035, 0.947,
        '19 reactions whose restricted reference solution is externally '
        'unstable, so a lower spin-broken solution exists there.',
        fontsize=10.5, color='#444', va='center')
ax.text(0.035, 0.925,
        'A structure is the transition state only if it is stationary and '
        'lower in energy, has exactly one imaginary frequency, and that mode '
        'moves this reaction’s bonds.',
        fontsize=10.5, color='#444', va='center')
ax.text(0.035, 0.903,
        'The same test is applied to both sides and has rejected structures on '
        'both.', fontsize=10.5, color='#444', va='center')

y = 0.862
hdr = [('rxn', 'reaction'), ('nfod', 'N_FOD'), ('ni', 'ν_imag'),
       ('frac', 'mode frac.'), ('rate', 'max bond rate'),
       ('model', 'closest model'), ('de', 'ΔE [meV]'),
       ('grad', 'grad [eV/Å]'), ('why', 'deciding evidence')]
for k, lab in hdr:
    ax.text(X[k], y, lab, fontsize=9.5, weight='bold', color='#222', va='center')
ax.plot([0.03, 0.975], [y - 0.012, y - 0.012], color='#222', lw=1.1)
ax.text(0.16, y + 0.024, 'our saddle', fontsize=9, style='italic',
        color='#666', ha='center')
ax.text(0.47, y + 0.024, 'best rival among UMA-S / UMA-M / eSEN',
        fontsize=9, style='italic', color='#666', ha='center')

y -= 0.036
cur = None
for r in rows:
    if r['v'] != cur:
        cur = r['v']
        y -= 0.014
        ax.add_patch(Rectangle((0.03, y - 0.011), 0.945, 0.026,
                               facecolor=COL[cur], alpha=0.14, lw=0))
        nrx = sum(1 for q in rows if q['v'] == cur)
        ax.text(0.038, y + 0.002, f'{TITLE[cur]}   —   {nrx} reactions',
                fontsize=11.5, weight='bold', color=COL[cur], va='center')
        y -= 0.034
    b = r['best']
    cells = {
        'rxn': r['rxn'], 'nfod': f"{r['nfod']:.3f}",
        'ni': '—' if r['ni'] is None else str(r['ni']),
        'frac': '—' if not r['oms'] else f"{r['oms']['frac']:.2f}",
        'rate': '—' if not r['oms'] else f"{r['oms']['maxrate']:.3f}",
        'model': '—' if not b else b[0],
        'de': '—' if not b else f'{b[1]:+.0f}',
        'grad': '—' if not b or b[2] is None else f'{b[2]:.3f}',
        'why': r['why']}
    for k, v in cells.items():
        weak = (k == 'frac' and r['oms'] and r['oms']['frac'] < 0.10) or \
               (k == 'rate' and r['oms'] and r['oms']['maxrate'] < 0.05)
        ax.text(X[k], y, v, fontsize=9.3, va='center',
                color='#b3261e' if weak else '#111',
                weight='bold' if weak or k == 'rxn' else 'normal',
                family='DejaVu Sans')
    y -= 0.0295

y -= 0.012
ax.plot([0.03, 0.975], [y, y], color='#ccc', lw=0.8)
ax.text(0.035, y - 0.026,
        'mode frac. — share of the imaginary mode sitting on the four '
        'reactive atoms.   max bond rate — fastest change of a reactive '
        'bond along that mode.',
        fontsize=9, color='#555', va='center')
ax.text(0.035, y - 0.05,
        'Values in red fail the test: below 0.10 the motion sits elsewhere in '
        'the molecule, below 0.05 it does not touch the reaction coordinate at '
        'all.', fontsize=9, color='#555', va='center')
ax.text(0.035, y - 0.074,
        'ΔE is the model energy minus ours, both DFT at wB97M-V/def2-TZVP '
        'on the ground-state surface. Negative means the model lies lower.',
        fontsize=9, color='#555', va='center')

fig.savefig('/home/energy/s242862/ts_reliability_table.png', dpi=190,
            facecolor='white')
print('written')
