"""How much of the benchmark's model error is chemistry, and how much is
hydrogen positions and group rotations?

The benchmark scores a model TS by the all-atom Kabsch RMSD against the ORCA
reference, with 0.3 A as the failure threshold. That metric is dominated by
light atoms: a rotated methyl or hydroxyl moves several hydrogens by more than
an Angstrom while every bond length stays put. rxn7945 is the clear case --
0.669 A all-atom, yet no bond differs by more than 0.025 A.

Three metrics per (reaction, model), all Kabsch-aligned:
  rmsd_all     what the benchmark currently reports
  rmsd_heavy   non-hydrogen atoms only, aligned on heavy atoms
  d_bond_max   largest difference in any covalent bond length

The last is the one that answers "is this the same transition state".
"""
import json
import os

import numpy as np
from ase.data import atomic_numbers, covalent_radii

H = '/home/energy/s242862'
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results', 'MACE': 'mace_bare_neb_results',
          'MACE+delta': 'mace_delta_neb_results_fw2'}
THR = 0.3


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
    """RMSD over `sel`, with the rotation fitted on `sel` too."""
    if sel is None:
        sel = np.arange(len(A))
    Ac = A - A[sel].mean(0)
    Bc = B - B[sel].mean(0)
    V, S, W = np.linalg.svd(Ac[sel].T @ Bc[sel])
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    A2 = Ac @ (V @ D @ W)
    d = A2[sel] - Bc[sel]
    return float(np.sqrt((d ** 2).sum() / len(sel)))


def bonds(sym, x, scale=1.3):
    out = {}
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            d = float(np.linalg.norm(x[i] - x[j]))
            rc = (covalent_radii[atomic_numbers[sym[i]]]
                  + covalent_radii[atomic_numbers[sym[j]]])
            if d < scale * rc:
                out[(i, j)] = d
    return out


def bond_max_diff(s1, x1, s2, x2):
    """Largest bond-length difference, over the union of both bond lists.
    A bond present in one structure and absent in the other counts as the full
    difference against the actual distance, not as a missing entry."""
    b1, b2 = bonds(s1, x1), bonds(s2, x2)
    worst, where = 0.0, None
    for k in set(b1) | set(b2):
        i, j = k
        d1 = b1.get(k, float(np.linalg.norm(x1[i] - x1[j])))
        d2 = b2.get(k, float(np.linalg.norm(x2[i] - x2[j])))
        if abs(d1 - d2) > worst:
            worst, where = abs(d1 - d2), f'{s1[i]}{i}-{s1[j]}{j}'
    return worst, where


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

rows = []
for rx in sorted(grp, key=lambda r: -nf[r]):
    ref = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
    if not os.path.exists(ref):
        continue
    sr, xr = read_xyz(ref)
    heavy = np.array([i for i, s in enumerate(sr) if s != 'H'])
    for m, dname in MODELS.items():
        p = f'{H}/{dname}/{rx}/transition_state.xyz'
        if not os.path.exists(p):
            continue
        sm, xm = read_xyz(p)
        if sm != sr:
            continue
        db, where = bond_max_diff(sr, xr, sm, xm)
        rows.append({'rxn': rx, 'grp': grp[rx], 'model': m,
                     'all': kabsch_rmsd(xm, xr),
                     'heavy': kabsch_rmsd(xm, xr, heavy),
                     'dbond': db, 'where': where})

print(f'{len(rows)} Zeilen\n')
a = np.array([r['all'] for r in rows])
h = np.array([r['heavy'] for r in rows])
b = np.array([r['dbond'] for r in rows])

print(f"{'':<14}{'median':>9}{'mean':>9}{'max':>9}{'>0.3':>7}")
print(f"{'RMSD all':<14}{np.median(a):>9.4f}{a.mean():>9.4f}{a.max():>9.4f}"
      f"{int((a > THR).sum()):>7}")
print(f"{'RMSD heavy':<14}{np.median(h):>9.4f}{h.mean():>9.4f}{h.max():>9.4f}"
      f"{int((h > THR).sum()):>7}")
print(f"{'max d(bond)':<14}{np.median(b):>9.4f}{b.mean():>9.4f}{b.max():>9.4f}"
      f"{int((b > THR).sum()):>7}")

print(f'\nZeilen ueber der 0.3-A-Schwelle nach dem All-Atom-Mass: '
      f'{int((a > THR).sum())}')
flagged = [r for r in rows if r['all'] > THR]
rescued = [r for r in flagged if r['dbond'] < 0.1]
print(f'davon mit groesster Bindungsdifferenz < 0.10 A: {len(rescued)}')
print('  -> nach dem All-Atom-Mass Versager, chemisch aber deckungsgleich\n')
if rescued:
    print(f"{'rxn':<10}{'grp':<6}{'Modell':<12}{'RMSD all':>10}"
          f"{'RMSD heavy':>12}{'max d(bond)':>13}  Bindung")
    for r in sorted(rescued, key=lambda x: -x['all']):
        print(f"{r['rxn']:<10}{r['grp']:<6}{r['model']:<12}{r['all']:>10.4f}"
              f"{r['heavy']:>12.4f}{r['dbond']:>13.4f}  {r['where']}")

print('\n=== je Modell: Zeilen ueber 0.3 A ===')
print(f"{'Modell':<12}{'nach all':>10}{'nach heavy':>12}{'nach d(bond)':>14}")
for m in MODELS:
    s = [r for r in rows if r['model'] == m]
    if not s:
        continue
    print(f"{m:<12}{sum(1 for r in s if r['all'] > THR):>10}"
          f"{sum(1 for r in s if r['heavy'] > THR):>12}"
          f"{sum(1 for r in s if r['dbond'] > THR):>14}")

json.dump(rows, open(f'{H}/rmsd_decomposed.json', 'w'), indent=1)
print(f'\ngeschrieben: {H}/rmsd_decomposed.json')
