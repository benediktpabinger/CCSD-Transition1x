"""
Collect ORCA SP results (wB97X-D3/6-31G(d)) and pair with:
  - Silver standard energies from neb.db (wB97M-V/def2-TZVP)
  - T1x reference barrier from Transition1x.h5

Outputs per reaction:
  ~/delta_experiment/results/rxnXXXX.json

Usage:
    python collect_sp_results.py
"""
import json
import os
import re

import h5py
import numpy as np
import ase.db

REACTIONS  = ['rxn0103', 'rxn1319', 'rxn7952']
SP_BASE    = '/home/energy/s242862/delta_experiment/orca_sp'
NEB_DIR    = '/home/energy/s242862/orca_neb_results'
T1X_H5     = '/home/energy/s242862/data/Transition1x.h5'
OUT_BASE   = '/home/energy/s242862/delta_experiment/results'

os.makedirs(OUT_BASE, exist_ok=True)

def parse_orca_energy(out_path):
    """Extract final single point energy (Eh) from ORCA output."""
    energy = None
    with open(out_path) as f:
        for line in f:
            if 'FINAL SINGLE POINT ENERGY' in line:
                energy = float(line.split()[-1])
    return energy  # Hartree

def silver_energies_from_db(db_path):
    """Return list of wB97M-V energies (eV) in db row order."""
    with ase.db.connect(db_path) as db:
        return [row.toatoms().get_potential_energy() for row in db.select()]

def find_t1x_barrier(h5, rxn_id):
    for split in ('test', 'val', 'train', 'data'):
        if split not in h5:
            continue
        for formula in h5[split].keys():
            if rxn_id in h5[split][formula]:
                grp = h5[split][formula][rxn_id]
                ts_e = float(grp['transition_state']['wB97x_6-31G(d).energy'][0])
                r_e  = float(grp['reactant']['wB97x_6-31G(d).energy'][0])
                return (ts_e - r_e) * 1000  # meV
    return None

HARTREE_TO_EV = 27.2114

with h5py.File(T1X_H5, 'r') as h5:
    for rxn in REACTIONS:
        sp_dir    = os.path.join(SP_BASE, rxn)
        db_path   = os.path.join(NEB_DIR, rxn, 'neb.db')
        geom_dirs = sorted(d for d in os.listdir(sp_dir)
                           if os.path.isdir(os.path.join(sp_dir, d)))

        silver = silver_energies_from_db(db_path)
        t1x_barrier = find_t1x_barrier(h5, rxn)

        records = []
        failed  = []
        for i, name in enumerate(geom_dirs):
            out_path = os.path.join(sp_dir, name, 'sp.out')
            if not os.path.exists(out_path):
                failed.append(name)
                continue
            e_t1x_ha = parse_orca_energy(out_path)
            if e_t1x_ha is None:
                failed.append(name)
                continue
            e_t1x_ev     = e_t1x_ha * HARTREE_TO_EV
            e_silver_ev  = silver[i] if i < len(silver) else None
            delta_ev     = (e_silver_ev - e_t1x_ev) if e_silver_ev is not None else None
            records.append({
                'geom_idx':      i,
                'name':          name,
                'e_t1x_eV':      e_t1x_ev,
                'e_silver_eV':   e_silver_ev,
                'delta_eV':      delta_ev,
            })

        # Barrier from SP energies
        if records:
            t1x_e   = [r['e_t1x_eV']   for r in records]
            silver_e = [r['e_silver_eV'] for r in records if r['e_silver_eV'] is not None]
            sp_barrier    = (max(t1x_e)    - t1x_e[0])    * 1000
            silver_barrier = (max(silver_e) - silver_e[0]) * 1000 if silver_e else None
        else:
            sp_barrier = silver_barrier = None

        result = {
            'rxn':              rxn,
            'n_geometries':     len(geom_dirs),
            'n_completed':      len(records),
            'n_failed':         len(failed),
            'failed':           failed,
            't1x_barrier_h5_meV':    t1x_barrier,
            'sp_barrier_meV':        sp_barrier,
            'silver_barrier_meV':    silver_barrier,
            'geometries':       records,
        }

        out_path = os.path.join(OUT_BASE, f'{rxn}.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f'{rxn}: {len(records)}/{len(geom_dirs)} completed, '
              f'SP barrier={sp_barrier:.0f} meV, silver barrier={silver_barrier:.0f} meV')
        if failed:
            print(f'  Failed: {failed[:5]}{"..." if len(failed)>5 else ""}')
