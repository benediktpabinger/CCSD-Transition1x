"""Does the imaginary mode actually move the reactive bonds?

One imaginary frequency proves a structure is a first-order saddle point. It does
not say what the saddle is a saddle *for*: a hindered methyl rotation is also a
first-order saddle. The transition state of interest is the one whose imaginary
mode stretches the bonds that break and form.

Test, per structure: diagonalise the stored Hessian, take the eigenvector of the
imaginary mode, and ask how much of it lies along the reactive bonds. Two
numbers are reported.

  d(bond)/dQ   the rate at which each reactive bond length changes along the
               mode, in A per unit of mass-weighted displacement. Large means
               the mode really is that bond breaking.
  Anteil       the fraction of the mode's squared amplitude carried by the four
               atoms of the two reactive bonds. Low means the motion sits
               somewhere else in the molecule.

No new quantum chemistry -- this reads hessian.npy from the frequency jobs.
"""
import glob
import json
import os

import numpy as np
from ase.data import atomic_masses, atomic_numbers

H = '/home/energy/s242862'
DIRS = ['bs_freq', 'bs_freq_v2', 'bs_freq_fromneb']


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def geom_of(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    return f
    return None


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


print(f"{'rxn':<10}{'v_imag':>8}{'Anteil reaktiv':>16}   "
      f"{'d(Bindung)/dQ  [A pro Einheit]':<40} Befund")
print('-' * 108)

for rx_dir in DIRS:
    for p in sorted(glob.glob(f'{H}/{rx_dir}/*/result.json')):
        rx = os.path.basename(os.path.dirname(p))
        j = json.load(open(p))
        if j.get('n_imag') != 1:
            continue
        hp = os.path.join(os.path.dirname(p), 'hessian.npy')
        g = geom_of(rx)
        pairs = reactive(rx)
        if not (os.path.exists(hp) and g and pairs):
            print(f'{rx:<10} (Hessian oder reaktive Bindungen fehlen)')
            continue
        hess = np.load(hp)                     # Ha / Bohr^2, 3N x 3N
        sym, xyz = read_xyz(g)
        m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
        w = np.repeat(1.0 / np.sqrt(m), 3)
        hm = hess * w[:, None] * w[None, :]    # mass-weighted
        ev, vec = np.linalg.eigh(hm)
        k = int(np.argmin(ev))                 # the one negative eigenvalue
        q = vec[:, k].reshape(-1, 3)
        q = q / np.linalg.norm(q)

        # how much of the mode sits on the reactive atoms
        idx = sorted({i for a, b, _ in pairs for i in (a, b)})
        frac = float((q[idx] ** 2).sum())

        # rate of change of each reactive bond along the mode
        rates = []
        for a, b, name in pairs:
            u = xyz[a] - xyz[b]
            u = u / np.linalg.norm(u)
            rates.append((name, float(np.dot(q[a] - q[b], u))))

        biggest = max(abs(r) for _, r in rates)
        if frac > 0.5 and biggest > 0.3:
            verdict = 'reaktive Mode'
        elif frac > 0.3 or biggest > 0.2:
            verdict = 'teilweise reaktiv'
        else:
            verdict = '*** ANDERE BEWEGUNG ***'
        txt = '  '.join(f'{n} {r:+.3f}' for n, r in rates)
        print(f"{rx:<10}{(j.get('imag_freq') or [0])[0]:>8.0f}{frac:>16.2f}   "
              f"{txt:<40} {verdict}")
