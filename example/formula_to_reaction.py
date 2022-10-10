import json
import h5py
import os
from argparse import ArgumentParser

from tqdm import tqdm


def main(args):  # pylint: disable=redefined-outer-name

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rxn_formula_index = {}
    with h5py.File(args.h5file, "r") as f:
        for formula, grp in tqdm(f['data'].items()):
            for rxn in grp:
                if formula not in rxn_formula_index:
                    rxn_formula_index[formula] = []
                rxn_formula_index[formula].append(rxn)

    json.dump(rxn_formula_index, open(args.output, "w")) # pylint: disable=consider-using-with


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("h5file", nargs="?", default="data/transition1x.h5")
    parser.add_argument("output", nargs="?", default="data/formula_reaction_index.json")
    args = parser.parse_args()

    main(args)
