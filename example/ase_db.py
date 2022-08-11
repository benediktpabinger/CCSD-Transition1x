import json
import os
from argparse import ArgumentParser

import ase.db
from ase import Atoms
from tqdm import tqdm
from transition1x import Dataloader


def main(args):  # pylint: disable=redefined-outer-name
    assert not os.path.exists(args.db), f"File {args.db} db already exists"
    dataloaders = {
        "train": Dataloader(args.h5file, "train"),
        "test": Dataloader(args.h5file, "test"),
        "val": Dataloader(args.h5file, "val"),
    }

    with ase.db.connect(args.db) as db:
        for split, dataloader in dataloaders.items():
            split_idx = []

            for configuration in tqdm(dataloader):
                atoms = Atoms(configuration["atomic_numbers"])
                atoms.set_positions(configuration["positions"])

                data = {
                    "energy": configuration["wB97x_6-31G(d).atomization_energy"],
                    "forces": configuration["wB97x_6-31G(d).forces"],
                }

                idx = db.write(atoms, data=data)
                split_idx.append(
                    idx - 1
                )  # idx are 0-indexed in the dataset but 1-indexed in the database

            json.dump(
                split_idx,
                open(os.path.join(os.path.dirname(args.db), split + "_idx.json"), "w"),
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("h5file", nargs="?", default="data/transition1x.h5")
    parser.add_argument("db", nargs="?", default="data/transition1x.db")
    args = parser.parse_args()

    main(args)
