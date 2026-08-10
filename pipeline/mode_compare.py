"""Do two competing saddle points describe the same reaction?

Energy decides which of two saddles the reaction takes -- but only if both sit
on the same path. A lower saddle belonging to a different rearrangement is no
competitor at all, and the endpoint test that was meant to settle this turned
out to give false negatives.

The imaginary mode carries the answer. If both saddles are on the same path,
their modes must move the same bonds in the same sense: the bond that breaks
must lengthen on both, the one that forms must shorten on both. Differing
patterns mean different reactions.

Compared here: the mode fraction on the reactive atoms, the rate of change of
each reactive bond along the mode, and the sign pattern. Sign matters -- a mode
that shortens where the other lengthens is the same motion traversed backwards,
which is fine, so the comparison is made up to an overall sign.
"""
import glob
import json
import os
import sys

import numpy as np
from ase.data import atomic_masses, atomic_numbers

H = '/home/energy/s242862'
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def mode_of(hess_path, geom_path, pairs):
    hess = np.load(hess_path)
    sym, xyz = read_xyz(geom_path)
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    w = np.repeat(1.0 / np.sqrt(m), 3)
    ev, vec = np.linalg.eigh(hess * w[:, None] * w[None, :])
    q = vec[:, int(np.argmin(ev))].reshape(-1, 3)
    q = q / np.linalg.norm(q)
    idx = sorted({i for a, b, _ in pairs for i in (a, b)})
    frac = float((q[idx] ** 2).sum())
    rates = []
    for a, b, nm in pairs:
        u = (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b])
        rates.append((nm, float(np.dot(q[a] - q[b], u)),
                      float(np.linalg.norm(xyz[a] - xyz[b]))))
    return frac, rates, float(ev[int(np.argmin(ev))])


def our_saddle(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    for fd in ('bs_freq', 'bs_freq_v2', 'bs_freq_fromneb'):
                        hp = f'{H}/{fd}/{rx}/hessian.npy'
                        if os.path.exists(hp):
                            return f, hp
    return None, None


rx = sys.argv[1]
pairs = reactive(rx)
if not pairs:
    print(f'{rx}: keine reaktiven Bindungen hinterlegt'); sys.exit(1)

g, hp = our_saddle(rx)
entries = []
if g and hp:
    entries.append(('unser BS-TS', g, hp))
for m in MODELDIR:
    hpm = f'{H}/freq_at_model/{rx}_{m}/hessian.npy'
    gm = f'{H}/{MODELDIR[m]}/{rx}/transition_state.xyz'
    if os.path.exists(hpm) and os.path.exists(gm):
        entries.append((m, gm, hpm))

print(f'=== {rx} ===')
print(f'reaktive Bindungen: ' + ', '.join(n for _, _, n in pairs))
print()
print(f"{'Struktur':<14}{'Anteil':>8}" +
      ''.join(f'{n + " d/dQ":>14}{n + " [A]":>13}' for _, _, n in pairs))
print('-' * (22 + 27 * len(pairs)))
store = {}
for name, gp, hpp in entries:
    frac, rates, ev = mode_of(hpp, gp, pairs)
    # fix the overall sign so the first reactive bond lengthens
    s = 1.0 if rates[0][1] >= 0 else -1.0
    rates = [(n, s * r, d) for n, r, d in rates]
    store[name] = rates
    line = f'{name:<14}{frac:>8.3f}'
    for n, r, d in rates:
        line += f'{r:>14.3f}{d:>13.3f}'
    print(line)

if 'unser BS-TS' in store and len(store) > 1:
    print('\nVergleich der Vorzeichenmuster (nach Angleichung des Gesamtvorzeichens):')
    ours = [np.sign(r) for _, r, _ in store['unser BS-TS']]
    for name, rates in store.items():
        if name == 'unser BS-TS':
            continue
        sg = [np.sign(r) for _, r, _ in rates]
        same = sg == ours
        weak = any(abs(r) < 0.05 for _, r, _ in rates)
        print(f'  {name:<12} {"gleich" if same else "VERSCHIEDEN":<12}'
              + ('   (mindestens eine Rate unter 0.05, also kaum aussagekraeftig)'
                 if weak else ''))
