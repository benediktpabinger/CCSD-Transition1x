"""Each reaction against its own correct reference.

Externally stable reactions: the RKS reference TS is the ground-state saddle, so
that is the right yardstick.
Externally unstable reactions with a confirmed broken-symmetry TS: the BS
structure is the right yardstick, and the RKS reference is not.

Scoring both groups against their own correct reference answers the question the
earlier tables could not: are the models genuinely worse on multireference
reactions, or did they only look worse because they were measured against a
reference that is wrong there?

Reactive bonds for the stable group are derived the way the rest of the project
does it -- the two atom pairs whose distance changes most between reactant and
product -- since no TS optimisation ran there to record them.
"""
import glob
import json
import os

import numpy as np
from ase.data import atomic_numbers, covalent_radii

H = '/home/energy/s242862'
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results', 'MACE': 'mace_bare_neb_results',
          'MACE+delta': 'mace_delta_neb_results_fw2'}
CONFIRMED = ['rxn0346', 'rxn0894', 'rxn1147', 'rxn1320', 'rxn4518', 'rxn5691',
             'rxn7949', 'rxn8827', 'rxn8837', 'rxn3107', 'rxn7957', 'rxn8832',
             'rxn8885']
STABLE = ['rxn7945', 'rxn7937', 'rxn1150', 'rxn0896', 'rxn7936', 'rxn0101',
          'rxn10005', 'rxn10054', 'rxn1154', 'rxn4513', 'rxn7955', 'rxn4519',
          'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155', 'rxn9246', 'rxn4498',
          'rxn1061', 'rxn4003', 'rxn4004', 'rxn4063', 'rxn4114', 'rxn4060',
          'rxn1961', 'rxn1962']


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B):
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    d = (Ac @ (V @ D @ W)) - Bc
    return float(np.sqrt((d ** 2).sum() / len(A)))


def reactive_from_endpoints(rx, sym):
    """Top-2 atom pairs by |d_product - d_reactant|, as in _rxn_coord_full.py."""
    r = f'{H}/orca_neb_results/{rx}/reactant.xyz'
    p = f'{H}/orca_neb_results/{rx}/product.xyz'
    if not (os.path.exists(r) and os.path.exists(p)):
        return []
    _, xr = read_xyz(r)
    _, xp = read_xyz(p)
    cand = []
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            dr = float(np.linalg.norm(xr[i] - xr[j]))
            dp = float(np.linalg.norm(xp[i] - xp[j]))
            rc = 1.3 * (covalent_radii[atomic_numbers[sym[i]]]
                        + covalent_radii[atomic_numbers[sym[j]]])
            if min(dr, dp) < rc:          # bonded in at least one endpoint
                cand.append((abs(dp - dr), i, j))
    cand.sort(reverse=True)
    return [(i, j) for _, i, j in cand[:2]]


def reactive_recorded(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1]) for e in rb[:2]]
    return []


def bs_ts(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    return f
    return None


def rc_err(pairs, xa, xb):
    if not pairs:
        return None
    return max(abs(float(np.linalg.norm(xa[i] - xa[j]))
                   - float(np.linalg.norm(xb[i] - xb[j]))) for i, j in pairs)


groups = {}
for label, rxns in (('RKS stabil (Referenz = RKS-TS)', STABLE),
                    ('BS (Referenz = BS-TS)', CONFIRMED)):
    rows = []
    for rx in rxns:
        if label.startswith('BS'):
            tgt = bs_ts(rx)
            pairs = reactive_recorded(rx)
        else:
            tgt = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
            pairs = None
        if not (tgt and os.path.exists(tgt)):
            continue
        st, xt = read_xyz(tgt)
        if pairs is None:
            pairs = reactive_from_endpoints(rx, st)
        if not pairs:
            continue
        for m, dn in MODELS.items():
            p = f'{H}/{dn}/{rx}/transition_state.xyz'
            if not os.path.exists(p):
                continue
            sm, xm = read_xyz(p)
            if sm != st:
                continue
            rows.append({'rxn': rx, 'model': m,
                         'rc': rc_err(pairs, xm, xt),
                         'rmsd': kabsch(xm, xt)})
    groups[label] = rows

for label, rows in groups.items():
    print(f'\n=== {label} ===')
    print(f'{len(rows)} Zeilen, {len(set(r["rxn"] for r in rows))} Reaktionen')
    print(f"{'Modell':<12}{'RC median':>11}{'RC >0.1':>9}"
          f"{'RMSD median':>13}{'RMSD >0.3':>11}")
    for m in MODELS:
        s = [r for r in rows if r['model'] == m]
        if not s:
            continue
        rc = np.array([r['rc'] for r in s])
        rm = np.array([r['rmsd'] for r in s])
        print(f'{m:<12}{np.median(rc):>11.4f}'
              f'{f"{int((rc>0.1).sum())}/{len(s)}":>9}'
              f'{np.median(rm):>13.4f}'
              f'{f"{int((rm>0.3).sum())}/{len(s)}":>11}')

print('\n=== direkter Vergleich: Faktor zwischen den Gruppen ===')
a = groups['RKS stabil (Referenz = RKS-TS)']
b = groups['BS (Referenz = BS-TS)']
print(f"{'Modell':<12}{'RC stabil':>11}{'RC BS':>10}{'Faktor':>9}"
      f"{'RMSD stabil':>13}{'RMSD BS':>10}{'Faktor':>9}")
for m in MODELS:
    sa = [r for r in a if r['model'] == m]
    sb = [r for r in b if r['model'] == m]
    if not (sa and sb):
        continue
    ra, rb = np.median([r['rc'] for r in sa]), np.median([r['rc'] for r in sb])
    ma, mb = np.median([r['rmsd'] for r in sa]), np.median([r['rmsd'] for r in sb])
    print(f'{m:<12}{ra:>11.4f}{rb:>10.4f}{rb/ra:>9.1f}'
          f'{ma:>13.4f}{mb:>10.4f}{mb/ma:>9.1f}')
