"""
Prepare delta model training and validation data.

Reads results.json files from the SP runs, loads atomic positions from
Transition1x.h5 (train, val Group B) or neb.db (val Group A), and saves:

    ~/delta_cache/train.pt   — 500 reactions x 10 geoms (energy + forces)
    ~/delta_cache/val.pt     — Group A (energy only) + Group B (energy + forces)

Each entry is a dict:
    {
        'rxn':            str,
        'geom_idx':       int,
        'positions':      np.array (N, 3) Angstrom,
        'atomic_numbers': np.array (N,),
        'delta_eV':       float,
        'delta_forces':   np.array (N, 3) eV/Ang, or None,
    }

Usage:
    python prepare_delta_data.py
"""
import json
import os

import ase.db
import h5py
import numpy as np
import torch

HOME      = '/home/energy/s242862'
H5FILE    = f'{HOME}/data/Transition1x.h5'
CACHE_DIR = f'{HOME}/delta_cache'

TRAIN_SP  = f'{HOME}/train_delta_sp'
VAL_A_SP  = f'{HOME}/val_delta_sp'
VAL_B_SP  = f'{HOME}/val_delta_sp_flip'
NEB_BASE  = f'{HOME}/orca_neb_val_results'

TRAIN_LIST = f'{HOME}/ccsd_dataset/train_delta_rxns.txt'
VAL_A_LIST = f'{HOME}/ccsd_dataset/val_converged.txt'
VAL_B_LIST = f'{HOME}/ccsd_dataset/val_failed.txt'


def load_rxn_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_h5_positions(rxn, split):
    with h5py.File(H5FILE, 'r') as f:
        for formula in f[split]:
            if rxn in f[split][formula]:
                g = f[split][formula][rxn]
                return g['positions'][:], g['atomic_numbers'][:]
    raise KeyError(f'{rxn} not found in {split} split')


def load_neb_positions(rxn):
    db_path = f'{NEB_BASE}/{rxn}/neb.db'
    with ase.db.connect(db_path) as db:
        rows = list(db.select())
    positions      = np.array([row.toatoms().get_positions() for row in rows])
    atomic_numbers = rows[0].toatoms().get_atomic_numbers()
    return positions, atomic_numbers


def load_results(json_path, need_forces_key=None):
    if not os.path.exists(json_path):
        return None
    data = json.load(open(json_path))
    if not data.get('geometries'):
        return None
    if need_forces_key and need_forces_key not in data['geometries'][0]:
        return None
    return data


def process_train(rxns):
    records, missing = [], []
    for rxn in rxns:
        data = load_results(f'{TRAIN_SP}/{rxn}/results.json', need_forces_key='forces_wb97m_eV_per_ang')
        if data is None:
            missing.append(rxn)
            continue
        try:
            all_pos, atomic_numbers = load_h5_positions(rxn, 'train')
        except KeyError:
            missing.append(rxn)
            continue
        for g in data['geometries']:
            idx = g['geom_idx']
            records.append({
                'rxn':            rxn,
                'geom_idx':       idx,
                'positions':      all_pos[idx].astype(np.float32),
                'atomic_numbers': atomic_numbers,
                'delta_eV':       float(g['delta_eV']),
                'delta_forces':   np.array(g['delta_forces_eV_per_ang'], dtype=np.float32),
            })
    print(f'Train: {len(records)} geoms from {len(rxns)-len(missing)}/{len(rxns)} reactions'
          + (f' | missing: {missing[:3]}{"..." if len(missing)>3 else ""}' if missing else ''))
    return records


def process_val_a(rxns):
    records, missing = [], []
    for rxn in rxns:
        data = load_results(f'{VAL_A_SP}/{rxn}/results.json')
        if data is None:
            missing.append(rxn)
            continue
        try:
            all_pos, atomic_numbers = load_neb_positions(rxn)
        except Exception:
            missing.append(rxn)
            continue
        for g in data['geometries']:
            idx = g['geom_idx']
            records.append({
                'rxn':            rxn,
                'geom_idx':       idx,
                'positions':      all_pos[idx].astype(np.float32),
                'atomic_numbers': atomic_numbers,
                'delta_eV':       float(g['delta_eV']),
                'delta_forces':   None,  # wB97M-V forces not in neb.db
            })
    print(f'Val A: {len(records)} geoms from {len(rxns)-len(missing)}/{len(rxns)} reactions')
    return records


def process_val_b(rxns):
    records, missing = [], []
    for rxn in rxns:
        data = load_results(f'{VAL_B_SP}/{rxn}/results.json')
        if data is None:
            missing.append(rxn)
            continue
        try:
            all_pos, atomic_numbers = load_h5_positions(rxn, 'val')
        except KeyError:
            missing.append(rxn)
            continue
        has_forces = 'delta_forces_eV_per_ang' in data['geometries'][0]
        for g in data['geometries']:
            idx = g['geom_idx']
            records.append({
                'rxn':            rxn,
                'geom_idx':       idx,
                'positions':      all_pos[idx].astype(np.float32),
                'atomic_numbers': atomic_numbers,
                'delta_eV':       float(g['delta_eV']),
                'delta_forces':   np.array(g['delta_forces_eV_per_ang'], dtype=np.float32) if has_forces else None,
            })
    print(f'Val B: {len(records)} geoms from {len(rxns)-len(missing)}/{len(rxns)} reactions')
    return records


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print('=== Training data ===')
    train_records = process_train(load_rxn_list(TRAIN_LIST))
    torch.save(train_records, f'{CACHE_DIR}/train.pt')
    print(f'Saved {CACHE_DIR}/train.pt\n')

    print('=== Validation data ===')
    val_records = process_val_a(load_rxn_list(VAL_A_LIST)) + process_val_b(load_rxn_list(VAL_B_LIST))
    torch.save(val_records, f'{CACHE_DIR}/val.pt')
    print(f'Saved {CACHE_DIR}/val.pt\n')

    n_train_f = sum(1 for r in train_records if r['delta_forces'] is not None)
    n_val_f   = sum(1 for r in val_records   if r['delta_forces'] is not None)
    print(f'Summary:')
    print(f'  Train: {len(train_records)} geoms, {n_train_f} with forces')
    print(f'  Val:   {len(val_records)} geoms, {n_val_f} with forces')


if __name__ == '__main__':
    main()
