from argparse import ArgumentParser

import ase.db
from ase import Atoms
from t1x import Dataloader
from tqdm import tqdm

def main(args):  # pylint: disable=redefined-outer-name
    dataloader = Dataloader(args.h5file)

    with ase.db.connect(args.db) as db:
        for configuration in tqdm(dataloader):
            atoms = Atoms(configuration["atomic_numbers"])
            atoms.set_positions(configuration["positions"])

            data = (
                {
                    "wB97x/6-31G(d).energy": configuration["wB97x_6-31G(d).energy"],
                    "wB97x/6-31G(d).atomization_energy": configuration["wB97x_6-31G(d).atomization_energy"],
                    "wB97x/6-31G(d).forces": configuration["wB97x_6-31G(d).forces"],
                },
            )
            db.write(atoms, data=data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("h5file")
    parser.add_argument("db")
    args = parser.parse_args()

    main(args)
