from argparse import ArgumentParser

import ase.db
from ase import Atoms
from dataloader import Dataloader


def main(args):  # pylint: disable=redefined-outer-name
    dataloader = Dataloader(args.h5_in)

    with ase.db.connect(args.db_out) as db:
        for mol in dataloader:
            atoms = Atoms(mol["atomic_numbers"])
            atoms.set_positions(mol["positions"])

            data = (
                {
                    "wB97x/6-31G(d).energy": mol["wB97x/6-31G(d).energy"].__float__(),
                    "wB97x/6-31G(d).atomization_energy": mol["wB97x/6-31G(d).atomization_energy"].__float__(),
                    "wB97x/6-31G(d).forces": mol["wB97x/6-31G(d).forces"].tolist(),
                },
            )
            db.write(atoms, data=data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("h5_in")
    parser.add_argument("db_out")
    args = parser.parse_args()
