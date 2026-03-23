"""
Combine CCSD NEB results into a single H5 file matching Transition1x format.

Reads per-reaction output from ccsd_neb_pyscf.py (neb.db + xyz files) and
writes a unified H5 dataset:

    /{split}/{formula}/{reaction}/
        positions        — (N_configs, N_atoms, 3) Angstrom
        atomic_numbers   — (N_atoms,)
        ccsd.energy      — (N_configs,) eV
        ccsd.forces      — (N_configs, N_atoms, 3) eV/Angstrom
        reactant/positions    — (1, N_atoms, 3)
        product/positions     — (1, N_atoms, 3)
        transition_state/positions — (1, N_atoms, 3)

Only converged reactions (output_dir/converged file exists) are included.

Usage:
    python combine_results.py \
        --results-dir ~/ccsd_pyscf_results \
        --reaction-list ~/test_reactions.txt \
        --ref-h5 ~/data/Transition1x.h5 \
        --output ccsd_transition1x_test.h5 \
        --split test
"""
import argparse
import os

import h5py
import numpy as np
from ase.db import connect
from ase.io import read


def get_formula(ref_h5, split, reaction):
    """Look up molecular formula for a reaction from the reference H5."""
    with h5py.File(ref_h5, 'r') as f:
        for formula in f[split]:
            if reaction in f[split][formula]:
                return formula
    return None


def read_neb_db(db_path, n_images=10):
    """Read final NEB iteration from neb.db.

    Returns positions (N, n_atoms, 3), energies (N,), forces (N, n_atoms, 3).
    """
    db = connect(db_path)
    rows = list(db.select())
    final = rows[-n_images:]

    positions = np.array([r.positions for r in final])   # (10, N_atoms, 3)
    energies  = np.array([r.energy for r in final])      # (10,)
    forces    = np.array([r.data['forces'] for r in final])  # (10, N_atoms, 3)
    atomic_numbers = final[0].numbers                    # (N_atoms,)

    return positions, energies, forces, atomic_numbers


def main(args):
    with open(args.reaction_list) as f:
        reactions = [line.strip() for line in f if line.strip()]

    converged, skipped = [], []
    for rxn in reactions:
        out_dir = os.path.join(args.results_dir, rxn)
        if os.path.exists(os.path.join(out_dir, 'converged')):
            converged.append(rxn)
        else:
            skipped.append(rxn)

    print(f"Converged: {len(converged)} / {len(reactions)}")
    print(f"Skipped (not converged): {len(skipped)}")
    if skipped:
        print(f"  Missing: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    with h5py.File(args.output, 'w') as out:
        split_grp = out.create_group(args.split)

        for rxn in converged:
            out_dir = os.path.join(args.results_dir, rxn)
            db_path = os.path.join(out_dir, 'neb.db')
            r_xyz   = os.path.join(out_dir, 'reactant.xyz')
            p_xyz   = os.path.join(out_dir, 'product.xyz')
            ts_xyz  = os.path.join(out_dir, 'transition_state.xyz')

            try:
                positions, energies, forces, atomic_numbers = read_neb_db(db_path)
            except Exception as e:
                print(f"  WARNING: could not read {db_path}: {e}")
                continue

            formula = get_formula(args.ref_h5, args.split, rxn)
            if formula is None:
                print(f"  WARNING: formula not found for {rxn}, skipping")
                continue

            if formula not in split_grp:
                split_grp.create_group(formula)
            rxn_grp = split_grp[formula].create_group(rxn)

            rxn_grp.create_dataset('positions',      data=positions)
            rxn_grp.create_dataset('atomic_numbers', data=atomic_numbers)
            rxn_grp.create_dataset('ccsd.energy',    data=energies)
            rxn_grp.create_dataset('ccsd.forces',    data=forces)

            # Endpoint and TS geometries
            for name, xyz_path in [('reactant', r_xyz), ('product', p_xyz),
                                   ('transition_state', ts_xyz)]:
                if os.path.exists(xyz_path):
                    atoms = read(xyz_path)
                    grp = rxn_grp.create_group(name)
                    grp.create_dataset('positions',
                                       data=atoms.get_positions()[np.newaxis])
                else:
                    print(f"  WARNING: {name}.xyz missing for {rxn}")

            print(f"  Written {rxn} ({len(energies)} images, formula={formula})")

    print(f"\nDone. Output: {args.output}")
    print(f"  {len(converged)} reactions written.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir',   required=True,
                        help='Directory containing per-reaction output folders')
    parser.add_argument('--reaction-list', required=True,
                        help='Path to test_reactions.txt')
    parser.add_argument('--ref-h5',        required=True,
                        help='Original Transition1x.h5 (for formula lookup)')
    parser.add_argument('--output',        default='ccsd_transition1x_test.h5')
    parser.add_argument('--split',         default='test')
    parser.add_argument('--n-images',      type=int, default=10)
    args = parser.parse_args()
    main(args)
