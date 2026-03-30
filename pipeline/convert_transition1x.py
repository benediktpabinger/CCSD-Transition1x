"""
Convert Transition1x.h5 to SchNetPack database format (.db).

Reads wB97x/6-31G(d) energies and forces from H5 and writes
to a SchNetPack ASE database for PaiNN training.

Usage:
    python convert_transition1x.py \
        --h5file ~/data/Transition1x.h5 \
        --output ~/data/transition1x_train.db \
        --split train

    # Quick smoke test (first 50 reactions only):
    python convert_transition1x.py \
        --h5file ~/data/Transition1x.h5 \
        --output ~/data/transition1x_test50.db \
        --split train \
        --max-reactions 50
"""
import argparse
import os

import h5py
import numpy as np
from ase import Atoms
import ase.db

ENERGY_KEY = 'energy'
FORCES_KEY = 'forces'

# SchNetPack 2.x metadata schema
SPK_METADATA = {
    '_property_unit_dict': {ENERGY_KEY: 'eV', FORCES_KEY: 'eV/Angstrom'},
    '_distance_unit': 'Ang',
}


def main(args):
    print(f"Converting split='{args.split}' from {args.h5file}")
    print(f"Output: {args.output}")

    if os.path.exists(args.output):
        if args.overwrite:
            os.remove(args.output)
        else:
            print("Output already exists. Use --overwrite to replace.")
            return

    n_reactions = 0
    n_configs = 0

    with ase.db.connect(args.output) as conn:
        conn.metadata = SPK_METADATA

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
                        conn.write(atoms, data={
                            ENERGY_KEY: np.array([energies[i]], dtype=np.float64),
                            FORCES_KEY: forces[i].astype(np.float32),
                        })
                    n_reactions += 1
                    n_configs   += len(positions)

                    if n_reactions % 100 == 0:
                        print(f"  {n_reactions} reactions, {n_configs} configs...")

                if args.max_reactions and n_reactions >= args.max_reactions:
                    break

    print(f"\nDone. {n_reactions} reactions, {n_configs} total configs written to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file',        required=True)
    parser.add_argument('--output',        required=True)
    parser.add_argument('--split',         default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--max-reactions', type=int, default=None,
                        help='Limit number of reactions (for smoke tests)')
    parser.add_argument('--overwrite',     action='store_true')
    args = parser.parse_args()
    main(args)
