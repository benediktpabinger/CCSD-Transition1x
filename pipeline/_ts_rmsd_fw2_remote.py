import numpy as np, os
from ase.io import read

CONVERGED = [
    'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
    'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150',
    'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
    'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
    'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
    'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
]
BASE = '/home/energy/s242862'
TOP10 = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
BOT10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}
MR = {r: ('High' if r in TOP10 else ('Low' if r in BOT10 else 'Mid')) for r in CONVERGED}

def kabsch_rmsd(P, Q):
    P = P - P.mean(axis=0); Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum(((P @ R.T) - Q)**2, axis=1))))

def load(path):
    if not os.path.exists(path):
        return None
    return read(path).get_positions()

def rs(p1, p2):
    if p1 is None or p2 is None: return 'N/A'
    return f'{kabsch_rmsd(p1, p2):.3f}'

print(f'{"rxn":<10} {"MR":<5} {"ORCA-fw2":>10} {"fw1-fw2":>10} {"eSEN-fw2":>10} {"UMA-fw2":>10}')
print('-' * 56)

rmsds = []
for rxn in CONVERGED:
    orca = load(f'{BASE}/orca_neb_results/{rxn}/transition_state.xyz')
    fw1  = load(f'{BASE}/mace_delta_neb_results/{rxn}/transition_state.xyz')
    fw2  = load(f'{BASE}/mace_delta_neb_results_fw2/{rxn}/transition_state.xyz')
    esen = load(f'{BASE}/esen_neb_results/{rxn}/transition_state.xyz')
    uma  = load(f'{BASE}/uma_neb_results/{rxn}/transition_state.xyz')
    print(f'{rxn:<10} {MR[rxn]:<5} {rs(orca,fw2):>10} {rs(fw1,fw2):>10} {rs(esen,fw2):>10} {rs(uma,fw2):>10}')
    if orca is not None and fw2 is not None:
        rmsds.append(kabsch_rmsd(orca, fw2))

print('\nUnits: Angstrom. Kabsch RMSD on centroid-aligned TS geometries.')
if rmsds:
    print(f'Mean ORCA-fw2 RMSD: {np.mean(rmsds):.3f}  (n={len(rmsds)})')
