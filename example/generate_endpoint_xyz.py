from argparse import ArgumentParser
import os
from ase import Atoms
from ase.io import write

from t1x import Dataloader


def main(args):  # pylint: disable=redefined-outer-name
    dataloader = Dataloader(args.h5file, 'test', only_final=True)
    for configurations in dataloader:
        rxn = configurations['rxn']
        reactant = configurations['reactant']
        product = configurations['product']
        transition_state = configurations['transition_state']

        rxn_path = os.path.join(args.output, rxn)
        os.makedirs(rxn_path, exist_ok=True)
        reactant = Atoms(positions=reactant['positions'], numbers = reactant['atomic_numbers'])
        product = Atoms(positions=product['positions'], numbers = product['atomic_numbers'])
        transition_state = Atoms(positions=transition_state['positions'], numbers = transition_state['atomic_numbers'])

        write(os.path.join(rxn_path, 'r.xyz'), reactant)
        write(os.path.join(rxn_path, 'p.xyz'), product)
        write(os.path.join(rxn_path, 'ts.xyz'), transition_state)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("h5file")
    parser.add_argument("output")
    args = parser.parse_args()

    main(args)
