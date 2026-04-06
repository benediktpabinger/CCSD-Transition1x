"""
Convert Transition1x.h5 to extended XYZ format for MACE training.

Reads wB97x/6-31G(d) energies and forces and writes extxyz files
that MACE can read directly.

Usage:
    python convert_h5_to_extxyz.py --h5file ~/data/Transition1x.h5 --output ~/data/transition1x_train.xyz --split train
    python convert_h5_to_extxyz.py --h5file ~/data/Transition1x.h5 --output ~/data/transition1x_val.xyz --split val
    python convert_h5_to_extxyz.py --h5file ~/data/Transition1x.h5 --output ~/data/transition1x_test.xyz --split test
"""
import argparse
import os

import h5py
import numpy as np
from ase import Atoms
from ase.io import write


def main(args):
    print(f"Converting split='{args.split}' from {args.h5file}")
    print(f"Output: {args.output}")

    if os.path.exists(args.output) and not args.overwrite:
        print("Output already exists. Use --overwrite to replace.")
        return

    if os.path.exists(args.output) and args.overwrite:
        os.remove(args.output)

    n_reactions = 0
    n_configs = 0

    with h5py.File(args.h5file, 'r') as f:
        split = f[args.split]

        for formula in split:
            for rxn in split[formula]:
                if args.max_reactions and n_reactions >= args.max_reactions:
                    break

                grp = split[formula][rxn]
                positions      = grp['positions'][:]
                atomic_numbers = grp['atomic_numbers'][:]
                energies       = grp['wB97x_6-31G(d).energy'][:]
                forces         = grp['wB97x_6-31G(d).forces'][:]

                for i in range(len(positions)):
                    atoms = Atoms(numbers=atomic_numbers, positions=positions[i])
                    atoms.info['energy'] = float(energies[i])
                    atoms.arrays['forces'] = forces[i].astype(np.float64)
                    write(args.output, atoms, format='extxyz', append=True)
                    n_configs += 1

                n_reactions += 1
                if n_reactions % 500 == 0:
                    print(f"  {n_reactions} reactions, {n_configs} configs...")

            if args.max_reactions and n_reactions >= args.max_reactions:
                break

    print(f"\nDone. {n_reactions} reactions, {n_configs} total configs -> {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file',        required=True)
    parser.add_argument('--output',        required=True)
    parser.add_argument('--split',         default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--max-reactions', type=int, default=None)
    parser.add_argument('--overwrite',     action='store_true')
    args = parser.parse_args()
    main(args)
