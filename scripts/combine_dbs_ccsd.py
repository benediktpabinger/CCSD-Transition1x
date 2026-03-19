"""
Same pipeline as combine_dbs.py but for CCSD NEB results.
Reads neb.db + fmaxs.json from each reaction output directory
and writes a Transition1x-compatible H5 with ccsd.energy / ccsd.forces.

Usage:
    python combine_dbs_ccsd.py --h5file ccsd_t1x.h5 --rxns rxns_ccsd.json

rxns_ccsd.json is a list of paths to ccsd_neb output directories, e.g.:
    ["/home/s242862/ccsd_neb_results/rxn0103", ...]
"""
import hashlib
import itertools
import json
import os
import re
from argparse import ArgumentParser

import ase.db
import h5py
import numpy as np
from tqdm import tqdm

# Test set molecular formulas (same split as Transition1x)
test_mols = ["C3H5NO2", "C2H5NO2", "C2H3N3O2", "C2HNO3", "C3H8N2O", "C3HNO2", "C5H5NO", "C2HNO"]


def main(args):
    rxns = json.load(open(args.rxns))
    h5file = h5py.File(args.h5file, 'w')

    test = h5file.create_group('test')
    for mol in test_mols:
        if mol in [get_formula(p) for p in rxns]:
            test[mol] = h5py.SoftLink(f'/data/{mol}')

    data = h5file.create_group('data')
    indexfile = open(args.h5file + '.index.json', 'w')
    index = {}

    for i, path in tqdm(enumerate(rxns)):
        rxn = re.search(r'rxn\d+', path).group()
        fmaxs_path = os.path.join(path, 'fmaxs.json')
        db_path = os.path.join(path, 'neb.db')

        if not os.path.exists(db_path):
            print(f"WARNING: {db_path} not found, skipping")
            continue
        if not os.path.exists(fmaxs_path):
            print(f"WARNING: {fmaxs_path} not found, skipping")
            continue

        new_rxn_name = f'rxn{str(i).zfill(4)}'
        write_rxn(data, fmaxs_path, db_path, new_rxn_name)
        index[new_rxn_name] = rxn

    json.dump(index, indexfile)
    h5file.close()
    print(f"Written: {args.h5file}")


def get_formula(path):
    """Try to infer formula from neb.db."""
    db_path = os.path.join(path, 'neb.db')
    if not os.path.exists(db_path):
        return None
    with ase.db.connect(db_path) as db:
        for row in db.select('', limit=1):
            return row.formula
    return None


def get_hash(row):
    s = str(row.positions) + row.formula
    return int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16) % (10 ** 8)


def write_rxn(h5file, fmaxs_path, db_path, rxn):
    fmaxs = json.load(open(fmaxs_path))

    skip_next = False
    first = True
    cum_fmax = 0

    with ase.db.connect(db_path) as db:
        for i, (fmax, path) in enumerate(zip(fmaxs, sliced_it(10, db.select('')))):
            cum_fmax += fmax
            skip_this = skip_next
            skip_next = False
            last = i == len(fmaxs) - 1

            if last:
                skip_this = False

            if cum_fmax < 0.1:
                skip_next = True
            else:
                cum_fmax = 0

            if skip_this:
                continue

            if not first:
                path = path[1:-1]

            forces_path = np.array([row.forces for row in path])
            positions_path = np.array([row.positions for row in path])
            energy_path = np.array([row.energy for row in path])

            if first:
                forces = forces_path
                positions = positions_path
                energy = energy_path
                reactant = path[0]
                product = path[-1]
            else:
                forces = np.concatenate((forces, forces_path), axis=0)
                positions = np.concatenate((positions, positions_path), axis=0)
                energy = np.concatenate((energy, energy_path), axis=0)

            first = False

    transition_state = path[np.argmax(energy_path)]

    formula = reactant.formula
    atomic_numbers = reactant.numbers

    if formula in h5file:
        grp = h5file[formula]
    else:
        grp = h5file.create_group(formula)

    subgrp = grp.create_group(rxn)
    single_molecule(reactant, subgrp.create_group('reactant'))
    single_molecule(transition_state, subgrp.create_group('transition_state'))
    single_molecule(product, subgrp.create_group('product'))

    write_group({
        'forces': forces,
        'positions': positions,
        'energy': energy,
        'atomic_numbers': atomic_numbers,
    }, subgrp)


def single_molecule(molecule, subgrp):
    write_group({
        'forces': np.expand_dims(molecule.forces, 0),
        'positions': np.expand_dims(molecule.positions, 0),
        'energy': np.expand_dims(molecule.energy, 0),
        'atomic_numbers': molecule.numbers,
        'hash': get_hash(molecule),
    }, subgrp)


def write_group(dict_, grp):
    grp.create_dataset('atomic_numbers', data=dict_['atomic_numbers'])
    grp.create_dataset('ccsd.forces', data=dict_['forces'])
    grp.create_dataset('ccsd.energy', data=dict_['energy'])
    grp.create_dataset('positions', data=dict_['positions'])
    if 'hash' in dict_:
        grp.create_dataset('hash', data=dict_['hash'])


def sliced_it(n, iterable):
    it = iter(iterable)
    while True:
        chunk = itertools.islice(it, n)
        yield list(chunk)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--h5file', required=True, help='Output H5 file path')
    parser.add_argument('--rxns', required=True, help='JSON list of ccsd_neb output directories')
    args = parser.parse_args()
    main(args)
