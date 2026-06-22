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

def conv(path_dir):
    return os.path.exists(os.path.join(path_dir, 'converged'))

def rs(p1, p2):
    if p1 is None or p2 is None: return None
    return kabsch_rmsd(p1, p2)

rows = []
for rxn in CONVERGED:
    orca   = load(f'{BASE}/orca_neb_results/{rxn}/transition_state.xyz')
    s_dir  = f'{BASE}/uma_neb_results/{rxn}'
    m_dir  = f'{BASE}/uma_m_neb_results/{rxn}'
    fw1_dir = f'{BASE}/mace_delta_neb_results/{rxn}'
    fw2_dir = f'{BASE}/mace_delta_neb_results_fw2/{rxn}'
    uma_s  = load(f'{s_dir}/transition_state.xyz')
    uma_m  = load(f'{m_dir}/transition_state.xyz')
    maced_fw1 = load(f'{fw1_dir}/transition_state.xyz')
    maced_fw2 = load(f'{fw2_dir}/transition_state.xyz')
    rows.append({
        'rxn': rxn, 'mr': MR[rxn],
        's_conv': conv(s_dir), 'm_conv': conv(m_dir), 'fw1_conv': conv(fw1_dir), 'fw2_conv': conv(fw2_dir),
        's_rmsd': rs(orca, uma_s), 'm_rmsd': rs(orca, uma_m), 'fw1_rmsd': rs(orca, maced_fw1), 'fw2_rmsd': rs(orca, maced_fw2),
    })

print(f'{"rxn":<10} {"MR":<5} {"UMA-S":>8} {"UMA-M":>8} {"MACEd-fw1":>10} {"MACEd-fw2":>10}')
print('-' * 62)
for r in rows:
    sr   = f'{r["s_rmsd"]:.3f}'   if r['s_rmsd']   is not None else 'N/A'
    mr_  = f'{r["m_rmsd"]:.3f}'   if r['m_rmsd']   is not None else 'N/A'
    fr1  = f'{r["fw1_rmsd"]:.3f}' if r['fw1_rmsd'] is not None else 'N/A'
    fr2  = f'{r["fw2_rmsd"]:.3f}' if r['fw2_rmsd'] is not None else 'N/A'
    print(f'{r["rxn"]:<10} {r["mr"]:<5} {sr:>8} {mr_:>8} {fr1:>10} {fr2:>10}')

print('\nMean RMSD vs ORCA wB97M-V NEB, by MR category (Angstrom):')
print(f'{"MR":<6} {"UMA-S":>10} {"UMA-M":>10} {"MACEd-fw1":>10} {"MACEd-fw2":>10} {"n":>4}')
for cat in ['High', 'Mid', 'Low']:
    s_vals   = [r['s_rmsd']   for r in rows if r['mr'] == cat and r['s_rmsd']   is not None]
    m_vals   = [r['m_rmsd']   for r in rows if r['mr'] == cat and r['m_rmsd']   is not None]
    fw1_vals = [r['fw1_rmsd'] for r in rows if r['mr'] == cat and r['fw1_rmsd'] is not None]
    fw2_vals = [r['fw2_rmsd'] for r in rows if r['mr'] == cat and r['fw2_rmsd'] is not None]
    s_mean   = f'{np.mean(s_vals):.3f}'   if s_vals   else 'N/A'
    m_mean   = f'{np.mean(m_vals):.3f}'   if m_vals   else 'N/A'
    fw1_mean = f'{np.mean(fw1_vals):.3f}' if fw1_vals else 'N/A'
    fw2_mean = f'{np.mean(fw2_vals):.3f}' if fw2_vals else 'N/A'
    print(f'{cat:<6} {s_mean:>10} {m_mean:>10} {fw1_mean:>10} {fw2_mean:>10} {len(s_vals):>4}')

s_all   = [r['s_rmsd']   for r in rows if r['s_rmsd']   is not None]
m_all   = [r['m_rmsd']   for r in rows if r['m_rmsd']   is not None]
fw1_all = [r['fw1_rmsd'] for r in rows if r['fw1_rmsd'] is not None]
fw2_all = [r['fw2_rmsd'] for r in rows if r['fw2_rmsd'] is not None]
print(f'{"All":<6} {np.mean(s_all):>10.3f} {np.mean(m_all):>10.3f} {np.mean(fw1_all):>10.3f} {np.mean(fw2_all):>10.3f} {len(s_all):>4}')
