"""
NEB using UMA-m (medium) via fairchem ASE calculator.

Drop-in analogue of uma_neb.py — identical NEB logic, uses UMA-m checkpoint.
Results stored in ~/uma_m_neb_results/{reaction}/

Usage:
    python uma_m_neb.py \
        --h5file     ~/data/Transition1x.h5 \
        --reaction   rxn7949 \
        --output     ~/uma_m_neb_results/rxn7949 \
        --checkpoint ~/checkpoints/uma-m-1p2.pt
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


def load_wB97x_images(h5file, reaction, split):
    with h5py.File(h5file, 'r') as f:
        split_group = f[split]
        for formula in split_group:
            if reaction in split_group[formula]:
                rxn_group = split_group[formula][reaction]
                positions = rxn_group['positions'][:]
                atomic_numbers = rxn_group['atomic_numbers'][:]
                total = positions.shape[0]
                final_positions = [positions[0]] + list(positions[-8:]) + [positions[9]]
                images = [Atoms(numbers=atomic_numbers, positions=pos) for pos in final_positions]
                print(f'Loaded 10 wB97x images from H5 ({total} total configs)')
                return images
    raise ValueError(f"Reaction '{reaction}' not found in split '{split}' of {h5file}")


def plot_mep(images, output):
    neb_tools = NEBTools(images)
    fit = neb_tools.get_fit()
    fig, ax = plt.subplots()
    ax.plot(fit.fit_path, fit.fit_energies, 'b-o', markersize=4,
            label=f'UMA-m barrier: {max(fit.fit_energies):.3f} eV')
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
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    predict_unit = pretrained_mlip.load_predict_unit(checkpoint_path, device='cuda')
    for atoms in images:
        atoms.calc = FAIRChemCalculator(predict_unit, task_name='omol')


def main(args):
    if not os.path.exists(args.h5file):
        print(f'ERROR: HDF5 file not found: {args.h5file}')
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f'ERROR: UMA-m checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f'Loading wB97x NEB images for {args.reaction} ...')
    images = load_wB97x_images(args.h5file, args.reaction, args.split)

    print(f'Attaching UMA-m calculator (checkpoint: {args.checkpoint}) ...')
    assign_calcs(images, args.checkpoint)

    r_xyz = os.path.join(args.output, 'reactant.xyz')
    p_xyz = os.path.join(args.output, 'product.xyz')

    if args.skip_relax and os.path.exists(r_xyz) and os.path.exists(p_xyz):
        print('Skipping endpoint relaxation ...')
        images[0].set_positions(read(r_xyz).get_positions())
        images[-1].set_positions(read(p_xyz).get_positions())
    else:
        print('Relaxing reactant ...')
        BFGS(images[0], logfile=os.path.join(args.output, 'relax_r.log')).run(fmax=0.05)
        write(r_xyz, images[0])
        print('Relaxing product ...')
        BFGS(images[-1], logfile=os.path.join(args.output, 'relax_p.log')).run(fmax=0.05)
        write(p_xyz, images[-1])

    print('Running NEB (UMA-m) ...')
    neb = NEB(images, climb=False, parallel=False, method='improvedtangent')
    neb_tools = NEBTools(images)
    relax_neb = NEBOptimizer(neb, logfile=os.path.join(args.output, 'neb.log'))

    db_path = os.path.join(args.output, 'neb.db')
    db_writer = DBWriter(db_path, images)
    fmaxs = []

    relax_neb.attach(db_writer.write)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))
    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file',      default='/home/energy/s242862/data/Transition1x.h5')
    parser.add_argument('--reaction',    required=True)
    parser.add_argument('--split',       default='test')
    parser.add_argument('--output',      required=True)
    parser.add_argument('--checkpoint',  default='/home/energy/s242862/checkpoints/uma-m-1p1.pt')
    parser.add_argument('--neb-fmax',    type=float, default=0.15)
    parser.add_argument('--cineb-fmax',  type=float, default=0.05)
    parser.add_argument('--steps',       type=int,   default=500)
    parser.add_argument('--skip-relax',  action='store_true')
    args = parser.parse_args()
    main(args)
