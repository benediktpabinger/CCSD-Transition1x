"""
Collect MR benchmark Step 1+2 results:
  - wB97X-D3/6-31G(d): parsed from ORCA sp.out files
  - wB97M-V/def2-TZVP: read directly from neb.db

Writes ~/mr_benchmark/results/{rxn}_sp.json per reaction and
       ~/mr_benchmark/sp_summary.json with barrier comparison.

Usage:
    python mr_benchmark_collect_sp.py
"""
import json
import os
import re

import ase.db
import numpy as np

TOP10 = [
    'rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885',
    'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150',
]

BASE    = '/home/energy/s242862/mr_benchmark'
NEB_DIR = '/home/energy/s242862/orca_neb_results'
SP_DIR  = f'{BASE}/orca_sp'
OUT_DIR = f'{BASE}/results'
os.makedirs(OUT_DIR, exist_ok=True)

ENERGY_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')


def parse_orca_energy(out_path):
    with open(out_path) as f:
        content = f.read()
    matches = ENERGY_RE.findall(content)
    if not matches:
        return None
    return float(matches[-1]) * 27.2114  # Ha → eV


def barrier(energies):
    ts = int(np.argmax(energies))
    fwd = energies[ts] - energies[0]
    rev = energies[ts] - energies[-1]
    return ts, round(fwd * 1000, 2), round(rev * 1000, 2)  # meV


summary = []

for rxn in TOP10:
    db_path = f'{NEB_DIR}/{rxn}/neb.db'
    sp_rxn  = f'{SP_DIR}/{rxn}'

    with ase.db.connect(db_path) as db:
        rows = list(db.select())

    n = len(rows)
    geometries = []
    errors = 0

    for i, row in enumerate(rows):
        e_wB97M = row.toatoms().get_potential_energy()  # eV, wB97M-V

        sp_out = f'{sp_rxn}/geom_{i:04d}/sp.out'
        e_wB97X = parse_orca_energy(sp_out)
        if e_wB97X is None:
            errors += 1
            continue

        geometries.append({
            'geom_idx':    i,
            'e_wb97x_eV':  round(e_wB97X, 6),
            'e_wb97m_eV':  round(e_wB97M, 6),
            'delta_eV':    round(e_wB97M - e_wB97X, 6),
        })

    # Last 10 images = converged NEB path
    last10 = [g for g in geometries if g['geom_idx'] >= n - 10]
    e_x = [g['e_wb97x_eV'] for g in last10]
    e_m = [g['e_wb97m_eV'] for g in last10]

    ts_x, fwd_x, rev_x = barrier(e_x)
    ts_m, fwd_m, rev_m = barrier(e_m)

    result = {
        'rxn':              rxn,
        'n_geometries':     len(geometries),
        'n_errors':         errors,
        'barrier_wb97x_fwd_meV': fwd_x,
        'barrier_wb97x_rev_meV': rev_x,
        'barrier_wb97m_fwd_meV': fwd_m,
        'barrier_wb97m_rev_meV': rev_m,
        'functional_gap_fwd_meV': round(fwd_m - fwd_x, 1),
        'geometries':       geometries,
    }

    out_path = f'{OUT_DIR}/{rxn}_sp.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    summary.append({k: v for k, v in result.items() if k != 'geometries'})
    print(f'{rxn}:  wB97X-D3={fwd_x:.0f}  wB97M-V={fwd_m:.0f}  '
          f'gap={fwd_m - fwd_x:+.0f} meV  errors={errors}')

with open(f'{BASE}/sp_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nSaved: {BASE}/sp_summary.json')
print(f'       {OUT_DIR}/{{rxn}}_sp.json')
