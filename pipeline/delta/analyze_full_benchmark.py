"""
Full benchmark analysis for 30 reactions.

Computes and saves to JSON:
  1. T1x wB97X-D3 barrier (from T1x HDF5 optimised R/TS)
  2. NEB wB97M-V barrier (last 10 images of neb.db: max - first)
  3. wB97X-D3 barrier on NEB geometries (orca_engrad last 10 images)
  4. CCSD(T) barrier (mr_benchmark/results/{rxn}_ccsdt.json)
  5. NEVPT2 barrier (nevpt2_results/{rxn}_pyscf_avas/, only for reliable reactions)
  6. MACE barrier and MACE+delta barrier (from eval_benchmark_sp.json)
  7. RMSD of TS: T1x geometry vs NEB wB97M-V geometry

Groups: High MR (top 10), Mid MR (middle 10), Low MR (bottom 10)

Output: ~/delta_head/full_benchmark_results.json

Usage:
    python analyze_full_benchmark.py
"""
import json
import os
import re

import ase.db
import h5py
import numpy as np

HOME       = '/home/energy/s242862'
T1X_H5     = f'{HOME}/data/Transition1x.h5'
NEB_DIR    = f'{HOME}/orca_neb_results'
ENGRAD_DIR = f'{HOME}/mr_benchmark/orca_engrad'
CCSDT_DIR  = f'{HOME}/mr_benchmark/results'
NEV_DIR    = f'{HOME}/nevpt2_results'
EVAL_SP    = f'{HOME}/delta_head/eval_benchmark_sp.json'
OUT_PATH   = f'{HOME}/delta_head/full_benchmark_results.json'

EH_BOHR_TO_EV_ANG = 51.42208619
ENERGY_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')

TOP10    = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
            'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150']
BOTTOM10 = ['rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
            'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962']
MIDDLE10 = ['rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
            'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155']
ALL30 = TOP10 + BOTTOM10 + MIDDLE10

FRAC_LO, FRAC_HI = 0.05, 1.95
NEV_RELIABLE = set()  # will be populated


def kabsch_rmsd(P, Q):
    """RMSD after optimal Kabsch rotation. P, Q: (N, 3) arrays."""
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_rot = P @ R.T
    return float(np.sqrt(np.mean(np.sum((P_rot - Q) ** 2, axis=1))))


def find_t1x(h5, rxn):
    """Return (ts_pos, r_pos, barrier_fwd_meV) from T1x HDF5."""
    for split in ('test', 'val', 'train'):
        if split not in h5:
            continue
        for formula in h5[split].keys():
            if rxn in h5[split][formula]:
                grp = h5[split][formula][rxn]
                ts_e = float(grp['transition_state']['wB97x_6-31G(d).energy'][0])
                r_e  = float(grp['reactant']['wB97x_6-31G(d).energy'][0])
                ts_pos = grp['transition_state']['positions'][0]   # (N, 3)
                return np.array(ts_pos), (ts_e - r_e) * 1000
    return None, None


def load_neb_images(rxn, n=10):
    """Last n images from neb.db. Returns (atoms_list, energies_eV)."""
    db_path = os.path.join(NEB_DIR, rxn, 'neb.db')
    with ase.db.connect(db_path) as db:
        rows = list(db.select())
    rows = rows[-n:]
    atoms_list = [row.toatoms() for row in rows]
    energies = np.array([a.get_potential_energy() for a in atoms_list])
    return atoms_list, energies


def parse_engrad_energy(sp_dir):
    """Parse energy (eV) from ORCA EnGrad sp.out."""
    out_path = os.path.join(sp_dir, 'sp.out')
    if not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        content = f.read()
    m = ENERGY_RE.findall(content)
    return float(m[-1]) * 27.2114 if m else None


def nev_plausible(rxn):
    """Return (barrier_fwd_meV, barrier_rev_meV) if plausible, else (None, None)."""
    path = os.path.join(NEV_DIR, f'{rxn}_pyscf_avas', 'nevpt2_results.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        nev = json.load(f)

    def count_frac(occ):
        return sum(1 for n in occ if FRAC_LO < n < FRAC_HI)

    r_frac  = count_frac(nev.get('reactant', {}).get('nat_occ', []))
    ts_frac = count_frac(nev.get('ts', {}).get('nat_occ', []))
    p_frac  = count_frac(nev.get('product', {}).get('nat_occ', []))

    fwd = nev.get('barrier_forward_meV')
    rev = nev.get('barrier_reverse_meV')

    if r_frac == 0 or ts_frac == 0 or p_frac == 0:
        return None, None
    if rev is not None and rev < 0:
        return None, None

    return fwd, rev


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


def main():
    # Load eval_benchmark_sp.json for MACE+delta energies
    with open(EVAL_SP) as f:
        eval_sp = {r['rxn']: r for r in json.load(f)['reactions']}

    results = []

    with h5py.File(T1X_H5, 'r') as h5:
        for rxn in ALL30:
            group = ('high' if rxn in TOP10 else
                     'low'  if rxn in BOTTOM10 else 'mid')

            row = {'rxn': rxn, 'group': group}

            # --- T1x wB97X-D3 barrier (optimised TS) ---
            try:
                ts_pos_t1x, t1x_barrier = find_t1x(h5, rxn)
                row['t1x_wb97x_fwd_meV'] = round(t1x_barrier, 1) if t1x_barrier is not None else None
            except Exception as e:
                ts_pos_t1x = None
                row['t1x_wb97x_fwd_meV'] = None
                print(f'{rxn}: T1x error: {e}')

            # --- NEB wB97M-V barrier + TS geometry ---
            try:
                atoms_list, e_wb97m = load_neb_images(rxn)
                rel_wb97m = (e_wb97m - e_wb97m[0]) * 1000
                row['neb_wb97m_fwd_meV'] = round(float(rel_wb97m.max()), 1)
                # TS image = max energy
                ts_idx = int(rel_wb97m.argmax())
                ts_pos_neb = atoms_list[ts_idx].get_positions()
            except Exception as e:
                atoms_list = None; ts_pos_neb = None
                row['neb_wb97m_fwd_meV'] = None
                print(f'{rxn}: NEB wB97M-V error: {e}')

            # --- RMSD of TS geometries ---
            if ts_pos_t1x is not None and ts_pos_neb is not None:
                if ts_pos_t1x.shape == ts_pos_neb.shape:
                    row['ts_rmsd_ang'] = round(kabsch_rmsd(ts_pos_t1x, ts_pos_neb), 4)
                else:
                    row['ts_rmsd_ang'] = None
                    print(f'{rxn}: TS shape mismatch T1x={ts_pos_t1x.shape} NEB={ts_pos_neb.shape}')
            else:
                row['ts_rmsd_ang'] = None

            # --- wB97X-D3 barrier on NEB geometries (orca_engrad last 10 images) ---
            try:
                e_wb97x_list = []
                for i in range(10):
                    sp_dir = os.path.join(ENGRAD_DIR, rxn, f'geom_{i:04d}')
                    e = parse_engrad_energy(sp_dir)
                    e_wb97x_list.append(e)
                if any(e is None for e in e_wb97x_list):
                    row['neb_wb97x_fwd_meV'] = None
                else:
                    e_arr = np.array(e_wb97x_list)
                    row['neb_wb97x_fwd_meV'] = round(float((e_arr.max() - e_arr[0]) * 1000), 1)
            except Exception as e:
                row['neb_wb97x_fwd_meV'] = None
                print(f'{rxn}: EnGrad error: {e}')

            # --- CCSD(T) barrier ---
            ccsdt_path = os.path.join(CCSDT_DIR, f'{rxn}_ccsdt.json')
            if os.path.exists(ccsdt_path):
                with open(ccsdt_path) as f:
                    cct = json.load(f)
                row['ccsdt_fwd_meV'] = cct.get('barrier_fwd_meV')
                row['ccsdt_rev_meV'] = cct.get('barrier_rev_meV')
            else:
                row['ccsdt_fwd_meV'] = None
                row['ccsdt_rev_meV'] = None

            # --- NEVPT2 barrier (only reliable, only top-10) ---
            if rxn in TOP10:
                nev_fwd, nev_rev = nev_plausible(rxn)
                row['nevpt2_fwd_meV'] = round(nev_fwd, 1) if nev_fwd is not None else None
                row['nevpt2_rev_meV'] = round(nev_rev, 1) if nev_rev is not None else None
                row['nevpt2_reliable'] = nev_fwd is not None
            else:
                row['nevpt2_fwd_meV'] = None
                row['nevpt2_rev_meV'] = None
                row['nevpt2_reliable'] = False

            # --- MACE + delta barriers (from eval_benchmark_sp.json) ---
            sp = eval_sp.get(rxn)
            if sp is not None:
                def rel_max(key):
                    e = np.array(sp[key])
                    return round(float(((e - e[0]) * 1000).max()), 1)
                row['mace_fwd_meV']  = rel_max('e_mace_eV')
                row['delta_fwd_meV'] = rel_max('e_delta_eV')
            else:
                row['mace_fwd_meV']  = None
                row['delta_fwd_meV'] = None

            results.append(row)
            print(f'{rxn} ({group}): T1x={row.get("t1x_wb97x_fwd_meV","?"):>7}  '
                  f'wb97m={row.get("neb_wb97m_fwd_meV","?"):>7}  '
                  f'wb97x_neb={row.get("neb_wb97x_fwd_meV","?"):>7}  '
                  f'ccsdt={row.get("ccsdt_fwd_meV","?"):>7}  '
                  f'rmsd={row.get("ts_rmsd_ang","?"):>6}  '
                  f'mace={row.get("mace_fwd_meV","?"):>7}  '
                  f'delta={row.get("delta_fwd_meV","?"):>7}', flush=True)

    # --- Print summary tables ---
    print()
    print('=' * 130)
    print('FULL BENCHMARK ANALYSIS — 30 REACTIONS')
    print('=' * 130)

    hdr = (f'{"Rxn":12s} {"Grp":5s} {"RMSD":>6} '
           f'{"T1x_wB97X":>9} {"NEB_wB97M":>9} {"NEB_wB97X":>9} '
           f'{"CCSD(T)":>8} {"NEVPT2":>7} {"MACE":>7} {"delta":>7}')
    print(hdr)
    print('-' * 130)

    for group_label, rxns in [('HIGH MR (top 10)', TOP10),
                               ('LOW MR (bottom 10)', BOTTOM10),
                               ('MID MR (middle 10)', MIDDLE10)]:
        print(f'\n--- {group_label} ---')
        for r in results:
            if r['rxn'] not in rxns:
                continue
            nev_str = f'{r["nevpt2_fwd_meV"]:>7.0f}' if r['nevpt2_fwd_meV'] is not None else '    N/A'
            rmsd_str = f'{r["ts_rmsd_ang"]:>6.3f}' if r['ts_rmsd_ang'] is not None else '   N/A'

            def fmt(v, w=7):
                return f'{v:>{w}.0f}' if v is not None else f'{"N/A":>{w}}'

            print(f'{r["rxn"]:12s} {r["group"]:5s} {rmsd_str} '
                  f'{fmt(r["t1x_wb97x_fwd_meV"],9)} {fmt(r["neb_wb97m_fwd_meV"],9)} '
                  f'{fmt(r["neb_wb97x_fwd_meV"],9)} {fmt(r["ccsdt_fwd_meV"],8)} '
                  f'{nev_str} {fmt(r["mace_fwd_meV"],7)} {fmt(r["delta_fwd_meV"],7)}')

    # --- Method comparison stats (vs CCSD(T) as reference for top-10) ---
    top10_results = [r for r in results if r['rxn'] in TOP10 and r['ccsdt_fwd_meV'] is not None]
    print(f'\n\n--- BARRIER COMPARISON vs CCSD(T) (top-10, n={len(top10_results)}) ---')
    print(f'{"Method":20s} {"MAE":>8} {"RMSE":>8} {"Bias":>9}')
    print('-' * 50)

    for label, key in [
        ('wB97M-V (NEB)',    'neb_wb97m_fwd_meV'),
        ('wB97X-D3 (NEB)',   'neb_wb97x_fwd_meV'),
        ('wB97X-D3 (T1x)',   't1x_wb97x_fwd_meV'),
        ('MACE',             'mace_fwd_meV'),
        ('MACE+delta',       'delta_fwd_meV'),
        ('NEVPT2',           'nevpt2_fwd_meV'),
    ]:
        valid = [(r[key], r['ccsdt_fwd_meV']) for r in top10_results
                 if r[key] is not None and r['ccsdt_fwd_meV'] is not None]
        if not valid:
            continue
        pred, ref = zip(*valid)
        err = np.array(pred) - np.array(ref)
        print(f'{label:20s} {np.abs(err).mean():8.1f} {np.sqrt((err**2).mean()):8.1f} {err.mean():+9.1f}')

    # NEVPT2 reliable count
    n_reliable = sum(1 for r in results if r['nevpt2_reliable'])
    print(f'\nNEVPT2 reliable: {n_reliable} / 10 top-MR reactions')

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump({'reactions': results}, f, indent=2, cls=NumpyEncoder)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
