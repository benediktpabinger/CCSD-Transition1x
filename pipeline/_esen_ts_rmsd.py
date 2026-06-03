"""Compute Kabsch RMSD between eSEN TS and ORCA wB97M-V TS for all 30 reactions."""
import numpy as np
from ase.io import read

REACTIONS = [
    'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
    'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150',
    'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
    'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
    'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
    'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
]
TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
MIDDLE10 = {'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955','rxn4519','rxn4500','rxn2553','rxn8829','rxn1155'}

def kabsch_rmsd(P, Q):
    P = P - P.mean(axis=0); Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((P @ R.T) - Q)**2, axis=1))))

results = {}
print(f'{"Rxn":<12} {"Group":<7} {"RMSD (A)":>9}')
print('-' * 32)

for rxn in REACTIONS:
    grp = 'High' if rxn in TOP10 else ('Mid' if rxn in MIDDLE10 else 'Low')
    esen_ts  = f'/home/energy/s242862/esen_neb_results/{rxn}/transition_state.xyz'
    orca_ts  = f'/home/energy/s242862/orca_neb_results/{rxn}/transition_state.xyz'
    try:
        p1 = read(esen_ts).get_positions()
        p2 = read(orca_ts).get_positions()
        if p1.shape != p2.shape:
            print(f'{rxn:<12} {grp:<7} shape mismatch {p1.shape} vs {p2.shape}')
            continue
        rmsd = kabsch_rmsd(p1, p2)
        results[rxn] = (grp, rmsd)
        print(f'{rxn:<12} {grp:<7} {rmsd:>9.4f}')
    except Exception as e:
        print(f'{rxn:<12} {grp:<7} ERROR: {e}')

print()
print('Summary by group:')
for label, gset in [('Low MR', 'Low'), ('Mid MR', 'Mid'), ('High MR', 'High'), ('All 30', None)]:
    vals = [v[1] for v in results.values() if gset is None or v[0] == gset]
    if vals:
        print(f'  {label:<10}: n={len(vals):2d}  mean={np.mean(vals):.4f}  '
              f'median={np.median(vals):.4f}  max={np.max(vals):.4f} A')
