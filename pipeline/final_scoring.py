"""Model transition states scored on two metrics at once, each reaction against
its own correct reference.

Reactive coordinate (max deviation over the two bonds that break and form,
threshold 0.10 A) answers "is this the same transition state".
All-atom Kabsch RMSD (threshold 0.30 A) answers "is this the same structure",
and is the metric the benchmark already uses.

Keeping both separates two failures that a single number conflates: missing the
reaction coordinate, and getting it right while placing the rest of the molecule
differently. The second is not nothing -- it is just not the same thing.

Reference: the RKS TS where the RKS solution is externally stable, the
frequency-confirmed broken-symmetry TS where it is not.
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
BS_GROUP = ['rxn0346', 'rxn0894', 'rxn1147', 'rxn1320', 'rxn4518', 'rxn5691',
            'rxn7949', 'rxn8827', 'rxn8837', 'rxn3107', 'rxn7957', 'rxn8832',
            'rxn8885']
STABLE = ['rxn7945', 'rxn7937', 'rxn1150', 'rxn0896', 'rxn7936', 'rxn0101',
          'rxn10005', 'rxn10054', 'rxn1154', 'rxn4513', 'rxn7955', 'rxn4519',
          'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155', 'rxn9246', 'rxn4498',
          'rxn1061', 'rxn4003', 'rxn4004', 'rxn4063', 'rxn4114', 'rxn4060',
          'rxn1961', 'rxn1962']
THR_RC, THR_RMSD = 0.10, 0.30


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
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def reactive_pairs(rx, sym):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1]) for e in rb[:2]]
    r, p = (f'{H}/orca_neb_results/{rx}/reactant.xyz',
            f'{H}/orca_neb_results/{rx}/product.xyz')
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
            if min(dr, dp) < rc:
                cand.append((abs(dp - dr), i, j))
    cand.sort(reverse=True)
    return [(i, j) for _, i, j in cand[:2]]


def bs_ts(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    return f
    return None


LABEL = {(True, True): 'korrekt',
         (True, False): 'RK ok, Konformation daneben',
         (False, True): 'RK daneben, Struktur nah',
         (False, False): 'falsch'}

rows = []
for group, rxns in (('stabil', STABLE), ('BS', BS_GROUP)):
    for rx in rxns:
        tgt = (bs_ts(rx) if group == 'BS'
               else f'{H}/orca_neb_results/{rx}/transition_state.xyz')
        if not (tgt and os.path.exists(tgt)):
            continue
        st, xt = read_xyz(tgt)
        pairs = reactive_pairs(rx, st)
        if not pairs:
            continue
        for m, dn in MODELS.items():
            p = f'{H}/{dn}/{rx}/transition_state.xyz'
            if not os.path.exists(p):
                continue
            sm, xm = read_xyz(p)
            if sm != st:
                continue
            rc = max(abs(float(np.linalg.norm(xm[i] - xm[j]))
                         - float(np.linalg.norm(xt[i] - xt[j])))
                     for i, j in pairs)
            rmsd = kabsch(xm, xt)
            rows.append({'group': group, 'rxn': rx, 'model': m,
                         'rc': rc, 'rmsd': rmsd,
                         'verdict': LABEL[(rc <= THR_RC, rmsd <= THR_RMSD)]})

for group in ('stabil', 'BS'):
    sub = [r for r in rows if r['group'] == group]
    n_rxn = len(set(r['rxn'] for r in sub))
    print(f'\n=== {group}  ({n_rxn} Reaktionen, {len(sub)} Zeilen) ===')
    print(f"{'Modell':<12}{'korrekt':>9}{'RK ok, Konf.':>14}"
          f"{'RK daneben':>12}{'falsch':>9}   RC med   RMSD med")
    for m in MODELS:
        s = [r for r in sub if r['model'] == m]
        if not s:
            continue
        c = {k: sum(1 for r in s if r['verdict'] == k) for k in LABEL.values()}
        print(f'{m:<12}{c["korrekt"]:>9}'
              f'{c["RK ok, Konformation daneben"]:>14}'
              f'{c["RK daneben, Struktur nah"]:>12}{c["falsch"]:>9}'
              f'{np.median([r["rc"] for r in s]):>10.4f}'
              f'{np.median([r["rmsd"] for r in s]):>11.4f}')

print('\n=== Zeilen, bei denen die beiden Masse widersprechen ===')
odd = [r for r in rows if r['verdict'].startswith('RK')]
print(f'{len(odd)} von {len(rows)}')
print(f"{'Gruppe':<8}{'rxn':<10}{'Modell':<12}{'RC':>8}{'RMSD':>9}  Befund")
for r in sorted(odd, key=lambda x: -x['rmsd'])[:15]:
    print(f"{r['group']:<8}{r['rxn']:<10}{r['model']:<12}{r['rc']:>8.3f}"
          f"{r['rmsd']:>9.3f}  {r['verdict']}")

json.dump(rows, open(f'{H}/final_scoring.json', 'w'), indent=1)
print(f'\ngeschrieben: {H}/final_scoring.json  ({len(rows)} Zeilen)')
