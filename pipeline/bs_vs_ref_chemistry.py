"""Are the broken-symmetry transition states chemically different from the RKS
reference, or only conformationally?

The control rxn7945 showed the trap: 0.669 A all-atom RMSD, yet no bond differed
by more than 0.025 A -- same structure, rotated hydroxyl. The same test applied
to the confirmed BS transition states decides whether the reference is wrong in
a way that matters.

Reported per reaction: all-atom RMSD, heavy-atom RMSD, the largest bond-length
difference anywhere, and the two reactive bonds specifically -- those are the
coordinates that define the transition state.
"""
import glob
import json
import os

import numpy as np
from ase.data import atomic_numbers, covalent_radii

H = '/home/energy/s242862'
CONFIRMED = ['rxn0346', 'rxn0894', 'rxn1147', 'rxn1320', 'rxn4518', 'rxn5691',
             'rxn7949', 'rxn8827', 'rxn8837', 'rxn3107', 'rxn7957', 'rxn8832',
             'rxn8885']
# the null measurement, for scale
CONTROL = ['rxn7945', 'rxn1150']


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B, sel=None):
    if sel is None:
        sel = np.arange(len(A))
    Ac, Bc = A - A[sel].mean(0), B - B[sel].mean(0)
    V, S, W = np.linalg.svd(Ac[sel].T @ Bc[sel])
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    d = (Ac @ (V @ D @ W))[sel] - Bc[sel]
    return float(np.sqrt((d ** 2).sum() / len(sel)))


def max_bond_diff(sym, x1, x2, scale=1.3):
    worst, where = 0.0, None
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            rc = scale * (covalent_radii[atomic_numbers[sym[i]]]
                          + covalent_radii[atomic_numbers[sym[j]]])
            d1 = float(np.linalg.norm(x1[i] - x1[j]))
            d2 = float(np.linalg.norm(x2[i] - x2[j]))
            if min(d1, d2) < rc and abs(d1 - d2) > worst:
                worst, where = abs(d1 - d2), f'{sym[i]}{i}-{sym[j]}{j}'
    return worst, where


def bs_ts(rx):
    for d in ('bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    return f, d
    return None, None


def reactive(rx):
    for d in ('bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return rb
    return []


print(f"{'rxn':<10}{'RMSD all':>10}{'RMSD schwer':>13}{'max d(bond)':>13}"
      f"{'wo':>10}   reaktive Bindungen: RKS-ref -> BS-TS")
print('-' * 100)
rows = []
for rx in CONFIRMED:
    a, src = bs_ts(rx)
    b = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
    if not (a and os.path.exists(b)):
        print(f'{rx:<10}(fehlt)'); continue
    s1, x1 = read_xyz(a)
    s2, x2 = read_xyz(b)
    if s1 != s2:
        print(f'{rx:<10}ATOMREIHENFOLGE'); continue
    heavy = np.array([i for i, s in enumerate(s1) if s != 'H'])
    ra = kabsch(x1, x2)
    rh = kabsch(x1, x2, heavy)
    db, where = max_bond_diff(s1, x1, x2)
    rb = reactive(rx)
    txt = []
    for e in rb[:2]:
        i, j = e['pair']
        d_ref = float(np.linalg.norm(x2[i] - x2[j]))
        d_bs = float(np.linalg.norm(x1[i] - x1[j]))
        txt.append(f"{e['sym']} {d_ref:.3f}->{d_bs:.3f} ({d_bs-d_ref:+.3f})")
    rows.append((rx, ra, rh, db))
    print(f'{rx:<10}{ra:>10.4f}{rh:>13.4f}{db:>13.4f}{where:>10}   '
          + '  '.join(txt))

print(f'\n--- Kontrolle: RKS extern stabil, es gibt nichts zu brechen ---')
for rx in CONTROL:
    a = f'{H}/bs_uks_neb_results/{rx}/bs_uks_neb_NEB-TS_converged.xyz'
    b = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
    if not (os.path.exists(a) and os.path.exists(b)):
        continue
    s1, x1 = read_xyz(a); s2, x2 = read_xyz(b)
    heavy = np.array([i for i, s in enumerate(s1) if s != 'H'])
    db, where = max_bond_diff(s1, x1, x2)
    print(f'{rx:<10}{kabsch(x1, x2):>10.4f}{kabsch(x1, x2, heavy):>13.4f}'
          f'{db:>13.4f}{where:>10}')

if rows:
    A = np.array([[r[1], r[2], r[3]] for r in rows])
    print(f'\nn = {len(rows)} bestaetigte BS-Uebergangszustaende')
    for k, name in enumerate(('RMSD all', 'RMSD schwer', 'max d(bond)')):
        print(f'  {name:<13} median {np.median(A[:,k]):.4f}   '
              f'min {A[:,k].min():.4f}   max {A[:,k].max():.4f}')
    print(f'\n  mit max d(bond) > 0.10 A (chemisch verschieden): '
          f'{int((A[:,2] > 0.10).sum())}/{len(rows)}')
    print(f'  mit max d(bond) < 0.05 A (chemisch gleich):       '
          f'{int((A[:,2] < 0.05).sum())}/{len(rows)}')
