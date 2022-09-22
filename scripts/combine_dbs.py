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

# fmt: off
train_mols = ["C6H11N", "C5H4O2", "C3H8O2", "C5H4N2", "C4H5N", "C2H4N2O", "C4H4N2O", "C3H5NO3", "C4H8O2", "C4H3N3", "C6H6O", "C5H11NO", "C5H8", "C6H12", "C6H14", "C4H6", "C6H5N", "C5H10O", "C5H11N", "C6H4O", "C6H10O", "C2H6O", "C2H5NO", "C3H4N2O", "C7H8", "C3H8", "C3H6O2", "C7H10", "C3H2N2", "C4H9N", "C5H8O2", "C4H4O2", "C4H5NO2", "C5H8N2", "C3H7N3", "C4H4O", "C3H2N2O2", "CH3N5", "C3H7N3O", "C3H4O3", "C6H14O", "C3H4O2", "C5H7N", "C4H8O3", "C2H3NO2", "C3H6O3", "CH2N4O", "C6H12O", "C5H6N2", "C3H3N3O", "C5H9N", "C4H5N3", "C4H3NO", "C5H4", "C2H4O2", "C4H6O", "C3H3NO2", "C4H8N2", "C3H4O", "C3H3NO", "C4H7NO", "C2H4N4", "C2H3N", "C3H5N", "C6H10", "C5H8O", "C5H3NO", "C6H8", "C4H9NO", "C5H10N2", "C2H3N3O", "C2H4N2O2", "C5H6", "C6H7N", "C2H2N4", "C3H2N2O", "C3H5N3", "C3HN3O", "C5H6O2", "C4H7N3", "C4H4N2", "C6H8O", "C5H12O", "C2H4O", "C7H12", "C4H3N", "C5H7NO", "C4H8O", "C3H2N4", "C2H3N5", "C2H3N3", "C4H4O3", "C5H12O2", "C4H10O2", "C3H4", "C2N2", "C4H8N2O", "C3H4N4", "C4H5NO", "C5H5N", "C3H2O3", "C3H7NO2", "C5H9NO", "C2H4N4O", "C3H7N", "C3H5NO", "C4N2", "C3HN", "C3H8O", "C4H6N2O", "C3H8N2O2", "C3H3NO3", "CH4N2O", "C7H16", "C3H2O", "C6H6", "C5H2O", "C4H6O3", "C3H6O", "C3H5N3O", "C2H6O2", "C5H6O", "C5H4O", "C7H14", "C4HNO", "C2H2O2", "C4H6O2", "C4H10", "C5H12", "C4H8", "C4H2O2", "C5H10O2", "C2H3NO", "C3H4N2O2", "C4H2N2O", "C5H10", "C4H10O3", "C6H9N", "C3H8O3", "C4H10O", "C3H7NO", "C3H6N2", "C3H3N3", "C4H7N", "C2H2N2O2", "C2H5N3O", "C3H6N2O2", "C3H4N2", "C4H7NO2", "C2H4N2", "C4H3NO2", "C4H6N2", "CHN3O2", "C6H13N", "C4H2"]
test_mols = ["C3H5NO2", "C2H5NO2", "C2H3N3O2", "C2HNO3", "C3H8N2O", "C3HNO2", "C5H5NO", "C2HNO"]
val_mols = ["C3N2O", "C2H2N2O", "C4H10N2O", "C3H6N2O", "C4H9NO2", "CHN3O", "C2H6N2O", "CN2O3"]
# fmt: on


def main(args):  # pylint: disable=redefined-outer-name
    rxns = json.load(open(args.rxns))
    h5file = h5py.File(args.h5file, "w")

    train = h5file.create_group("train")
    test = h5file.create_group("test")
    val = h5file.create_group("val")

    for mol in train_mols:
        train[mol] = h5py.SoftLink(f"/data/{mol}")
    for mol in test_mols:
        test[mol] = h5py.SoftLink(f"/data/{mol}")
    for mol in val_mols:
        val[mol] = h5py.SoftLink(f"/data/{mol}")

    data = h5file.create_group("data")
    indexfile = open(args.h5file + ".index.json", "w")
    index = {}

    for i, path in tqdm(enumerate(rxns)):
        rxn = re.search(
            r"rxn\d+", path
        ).group()
        fmaxs_path = os.path.join(path, "fmaxs.json")
        db_path = os.path.join(path, "neb.db")

        new_rxn_name = f"rxn{str(i).zfill(4)}"
        write_rxn(data, fmaxs_path, db_path, new_rxn_name)
        index[new_rxn_name] = rxn

    json.dump(index, indexfile)


def get_hash(row):
    s = str(row.positions) + row.formula
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def write_rxn(h5file, fmaxs_path, db_path, rxn):
    fmaxs = json.load(open(fmaxs_path))

    skip_next = False
    first = True
    cum_fmax = 0

    with ase.db.connect(db_path) as db:
        for i, (fmax, path) in enumerate(zip(fmaxs, sliced_it(10, db.select("")))):
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
                reactant = path[0]  # pylint: disable=undefined-loop-variable
                product = path[-1]  # pylint: disable=undefined-loop-variable

            else:
                forces = np.concatenate((forces, forces_path), axis=0)
                positions = np.concatenate((positions, positions_path), axis=0)
                energy = np.concatenate((energy, energy_path), axis=0)

            first = False

    transition_state = path[  # pylint: disable=undefined-loop-variable
        np.argmax(energy_path)
    ]

    formula = reactant.formula
    atomic_numbers = reactant.numbers

    if formula in h5file:
        grp = h5file[formula]
    else:
        grp = h5file.create_group(formula)

    subgrp = grp.create_group(rxn)
    single_molecule(reactant, subgrp.create_group("reactant"))
    single_molecule(transition_state, subgrp.create_group("transition_state"))
    single_molecule(product, subgrp.create_group("product"))

    dict_ = {
        "forces": forces,
        "positions": positions,
        "energy": energy,
        "atomic_numbers": atomic_numbers,
    }
    write_group(dict_, subgrp)


def single_molecule(molecule, subgrp):
    dict_ = {
        "forces": np.expand_dims(molecule.forces, 0),
        "positions": np.expand_dims(molecule.positions, 0),
        "energy": np.expand_dims(molecule.energy, 0),
        "atomic_numbers": molecule.numbers,
        "hash": get_hash(molecule),
    }
    write_group(dict_, subgrp)


def write_group(dict_, grp):
    grp.create_dataset("atomic_numbers", data=dict_["atomic_numbers"])
    grp.create_dataset("wB97x_6-31G(d).forces", data=dict_["forces"])
    grp.create_dataset("wB97x_6-31G(d).energy", data=dict_["energy"])
    grp.create_dataset("positions", data=dict_["positions"])

    if "hash" in dict_:
        grp.create_dataset("hash", data=dict_["hash"])


def sliced_it(n, iterable):
    it = iter(iterable)
    while True:
        chunk = itertools.islice(it, n)
        yield list(chunk)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("h5file", help="Path to the h5 file to write to")
    parser.add_argument("rxns", help="Path to rxns.json, contains all reactions that should be included in the dataset ")
    args = parser.parse_args()

    main(args)
