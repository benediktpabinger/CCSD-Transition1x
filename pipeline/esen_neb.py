"""
NEB at ωB97M-V/def2-TZVPD using eSEN (OMol25) via fairchem ASE calculator.

Drop-in analogue of orca_neb.py — identical NEB logic, only the calculator
changes. Outputs are compatible with the existing analysis pipeline.

Pipeline:
  1. Load final wB97x NEB images from Transition1x.h5 as starting band
  2. Relax endpoints with eSEN / BFGS
  3. Run NEB → CI-NEB using NEBOptimizer
  4. Save neb.db + fmaxs.json (same structure as orca_neb.py)

Results stored in ~/esen_neb_results/{reaction}/

Usage:
    python esen_neb.py \
        --h5file     ~/data/Transition1x.h5 \
        --reaction   rxn7949 \
        --output     ~/esen_neb_results/rxn7949 \
        --checkpoint ~/checkpoints/esen_sm_conserving_all.pt
"""
import argparse
import json
import os
import sys

import ase.db
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.mep import NEB, NEBTools
from ase.mep.neb import NEBOptimizer
from ase.optimize.bfgs import BFGS


def make_esen_calc(checkpoint_path):
    from fairchem.core import OCPCalculator
    return OCPCalculator(
        checkpoint_path=checkpoint_path,
        cpu=False,
    )


def load_wB97x_images(h5file, reaction, split):
    """Load final wB97x NEB images from H5 as starting band.
    Identical to orca_neb.py — reuses same initial geometry.
    """
    with h5py.File(h5file, 'r') as f:
        split_group = f[split]
        for formula in split_group:
            if reaction in split_group[formula]:
                rxn_group = split_group[formula][reaction]
                positions = rxn_group['positions'][:]
                atomic_numbers = rxn_group['atomic_numbers'][:]
                total = positions.shape[0]

                # R + last 8 interior + P
                final_positions = [positions[0]] + list(positions[-8:]) + [positions[9]]
                images = [
                    Atoms(numbers=atomic_numbers, positions=pos)
                    for pos in final_positions
                ]
                print(f'Loaded 10 wB97x images from H5 ({total} total configs)')
                return images

    raise ValueError(f"Reaction '{reaction}' not found in split '{split}' of {h5file}")


def plot_mep(images, output):
    neb_tools = NEBTools(images)
    fit = neb_tools.get_fit()
    fig, ax = plt.subplots()
    ax.plot(fit.fit_path, fit.fit_energies, 'b-o', markersize=4,
            label=f'eSEN (OMol25) barrier: {max(fit.fit_energies):.3f} eV')
    ax.set_ylabel('Energy [eV]')
    ax.set_xlabel('Reaction Coordinate [Å]')
    ax.legend()
    fig.savefig(os.path.join(output, 'mep.png'))
    plt.close(fig)


class DBWriter:
    def __init__(self, db_path, images):
        self.images = images
        self.db_path = db_path

    def write(self):
        with ase.db.connect(self.db_path) as db:
            for atoms in self.images:
                if atoms.calc and atoms.calc.results:
                    db.write(atoms, data=atoms.calc.results)


def assign_calcs(images, checkpoint_path):
    """Assign eSEN calculator to all images.
    Unlike ORCA, one calculator instance can serve all images — eSEN
    is stateless and GPU-batched, so sharing is safe and efficient.
    """
    calc = make_esen_calc(checkpoint_path)
    for atoms in images:
        atoms.calc = calc


def main(args):
    if not os.path.exists(args.h5file):
        print(f'ERROR: HDF5 file not found: {args.h5file}')
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f'ERROR: eSEN checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f'Loading wB97x NEB images for {args.reaction} ...')
    images = load_wB97x_images(args.h5file, args.reaction, args.split)

    print(f'Attaching eSEN calculator (checkpoint: {args.checkpoint}) ...')
    assign_calcs(images, args.checkpoint)

    # Relax endpoints
    r_xyz = os.path.join(args.output, 'reactant.xyz')
    p_xyz = os.path.join(args.output, 'product.xyz')

    if args.skip_relax and os.path.exists(r_xyz) and os.path.exists(p_xyz):
        print('Skipping endpoint relaxation (loading existing xyz) ...')
        images[0].set_positions(read(r_xyz).get_positions())
        images[-1].set_positions(read(p_xyz).get_positions())
    else:
        print('Relaxing reactant ...')
        BFGS(images[0], logfile=os.path.join(args.output, 'relax_r.log')).run(fmax=0.05)
        write(r_xyz, images[0])

        print('Relaxing product ...')
        BFGS(images[-1], logfile=os.path.join(args.output, 'relax_p.log')).run(fmax=0.05)
        write(p_xyz, images[-1])

    # NEB
    print('Running NEB (eSEN / OMol25) ...')
    neb = NEB(images, climb=False, parallel=False)
    neb_tools = NEBTools(images)
    relax_neb = NEBOptimizer(neb, logfile=os.path.join(args.output, 'neb.log'))

    db_path = os.path.join(args.output, 'neb.db')
    db_writer = DBWriter(db_path, images)
    fmaxs = []

    relax_neb.attach(db_writer.write)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))
    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

    # CI-NEB
    print('Running CI-NEB ...')
    neb.climb = True
    converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)

    if converged:
        open(os.path.join(args.output, 'converged'), 'w').close()
        print('CI-NEB converged!')
    else:
        print('WARNING: CI-NEB did not converge within step limit')

    json.dump(fmaxs, open(os.path.join(args.output, 'fmaxs.json'), 'w'))

    ts_out = max(images, key=lambda x: x.get_potential_energy())
    write(os.path.join(args.output, 'transition_state.xyz'), ts_out)
    write(r_xyz, images[0])
    write(p_xyz, images[-1])

    plot_mep(images, args.output)

    print(f'\nDone. Results in {args.output}/')
    print(f'  neb.db            — eSEN energies + forces')
    print(f'  transition_state.xyz')
    print(f'  fmaxs.json        — convergence history')
    print(f'  mep.png           — minimum energy path')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file',      required=True)
    parser.add_argument('--reaction',    required=True)
    parser.add_argument('--split',       default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output',      required=True)
    parser.add_argument('--checkpoint',  required=True, help='Path to eSEN .pt checkpoint')
    parser.add_argument('--neb-fmax',    type=float, default=0.15)
    parser.add_argument('--cineb-fmax',  type=float, default=0.05)
    parser.add_argument('--steps',       type=int,   default=500)
    parser.add_argument('--skip-relax',  action='store_true')
    args = parser.parse_args()
    main(args)
