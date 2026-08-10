"""Do the two independent routes to a broken-symmetry TS agree?

Route B: PySCF TS optimisation from the RKS reference geometry, broken symmetry
         carried step to step as a density matrix. Produced the 13 confirmed
         structures, each with exactly one imaginary frequency.
Route A: ORCA NEB-TS with BrokenSym 1,1 -- a full path search from relaxed
         endpoints, re-deriving the broken symmetry at every SCF from scratch.

Nothing is shared between them: different code, different starting point,
different way of breaking the symmetry, different optimiser. Agreement is
therefore real evidence; disagreement localises the problem.

Reported against the RKS reference as well, so the size of any discrepancy can
be read against the effect being claimed.
"""
import glob
import os

import numpy as np
from ase.data import atomic_numbers, covalent_radii

H = '/home/energy/s242862'


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0])
        xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch_rmsd(A, B, sel=None):
    if sel is None:
        sel = np.arange(len(A))
    Ac, Bc = A - A[sel].mean(0), B - B[sel].mean(0)
    V, S, W = np.linalg.svd(Ac[sel].T @ Bc[sel])
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    d = (Ac @ (V @ D @ W))[sel] - Bc[sel]
    return float(np.sqrt((d ** 2).sum() / len(sel)))


def bond_max_diff(s, x1, x2, scale=1.3):
    worst, where = 0.0, None
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            rc = (covalent_radii[atomic_numbers[s[i]]]
                  + covalent_radii[atomic_numbers[s[j]]]) * scale
            d1 = float(np.linalg.norm(x1[i] - x1[j]))
            d2 = float(np.linalg.norm(x2[i] - x2[j]))
            if min(d1, d2) < rc and abs(d1 - d2) > worst:
                worst, where = abs(d1 - d2), f'{s[i]}{i}-{s[j]}{j}'
    return worst, where


def pyscf_ts(rx):
    for d in ('bs_tsopt_v2', 'bs_tsopt_batch'):
        c = glob.glob(f'{H}/{d}/{rx}/*.xyz')
        for pat in ('ts', 'final', 'opt'):
            for f in c:
                if pat in os.path.basename(f).lower():
                    return f
    return None


CONFIRMED = ['rxn0346', 'rxn0894', 'rxn1147', 'rxn1320', 'rxn4518', 'rxn5691',
             'rxn7949', 'rxn8827', 'rxn8837', 'rxn3107', 'rxn7957', 'rxn8832',
             'rxn8885']

print(f"{'rxn':<10}{'RMSD A-B':>10}{'schwer':>9}{'max d(bond)':>13}"
      f"{'B vs RKS':>10}{'A vs RKS':>10}  groesste Abweichung")
print('-' * 78)
pairs = []
for rx in CONFIRMED:
    a = f'{H}/bs_uks_neb_results/{rx}/bs_uks_neb_NEB-TS_converged.xyz'
    b = pyscf_ts(rx)
    ref = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
    if not (os.path.exists(a) and b and os.path.exists(b)):
        print(f'{rx:<10}(NEB laeuft noch)')
        continue
    sa, xa = read_xyz(a)
    sb, xb = read_xyz(b)
    sr, xr = read_xyz(ref)
    if not (sa == sb == sr):
        print(f'{rx:<10}ATOMREIHENFOLGE WEICHT AB')
        continue
    heavy = np.array([i for i, s in enumerate(sa) if s != 'H'])
    r_ab = kabsch_rmsd(xa, xb)
    r_ab_h = kabsch_rmsd(xa, xb, heavy)
    db, where = bond_max_diff(sa, xa, xb)
    r_b = kabsch_rmsd(xb, xr)
    r_a = kabsch_rmsd(xa, xr)
    pairs.append((rx, r_ab, r_ab_h, db, r_b, r_a))
    print(f'{rx:<10}{r_ab:>10.4f}{r_ab_h:>9.4f}{db:>13.4f}'
          f'{r_b:>10.4f}{r_a:>10.4f}  {where}')

if pairs:
    A = np.array([[p[1], p[2], p[3]] for p in pairs])
    print(f'\nn = {len(pairs)}')
    for k, name in enumerate(('RMSD alle', 'RMSD schwer', 'max d(bond)')):
        print(f'  {name:<13} median {np.median(A[:, k]):.4f}   '
              f'min {A[:, k].min():.4f}   max {A[:, k].max():.4f}')
    ok = sum(1 for p in pairs if p[3] < 0.05)
    print(f'\nBindungen stimmen auf < 0.05 A: {ok}/{len(pairs)}')
