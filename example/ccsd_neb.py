"""
CCSD/cc-pVDZ NEB calculation — same pipeline as Transition1x generation but with CCSD accuracy.

Reads reactant, transition state, and product from Transition1x.h5 as starting geometries.
Relaxes endpoints with CCSD/BFGS, then runs NEB → CI-NEB.
Output: neb.db + fmaxs.json (identical structure to original wB97x pipeline).

Serial mode (default):
    python ccsd_neb.py --h5file ~/data/Transition1x.h5 --reaction rxn0103 --output ~/ccsd_neb/rxn0103

Parallel mode (one MPI rank per interior image — faster for full test set):
    mpirun -n <n_images-2> python ccsd_neb.py ... --parallel
"""
import argparse
import json
import os
import shutil
import sys

import ase.db
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import write
from ase.mep import NEB, NEBTools
from ase.optimize import LBFGS
from ase.optimize.bfgs import BFGS
from transition1x import Dataloader


def load_endpoints(h5file, reaction, split):
    """Load reactant, transition state, and product from Transition1x H5."""
    dl = Dataloader(h5file, split, only_final=True)
    for mol in dl:
        if mol['rxn'] == reaction:
            def to_atoms(d):
                return Atoms(numbers=d['atomic_numbers'], positions=d['positions'])
            return (
                to_atoms(mol['reactant']),
                to_atoms(mol['transition_state']),
                to_atoms(mol['product']),
            )
    raise ValueError(f"Reaction '{reaction}' not found in split '{split}' of {h5file}")


def make_calculator(label, directory, basis, nprocs):
    orca_path = shutil.which('orca')
    if orca_path is None:
        raise RuntimeError("ORCA not found in PATH. Load with: module load ORCA/5.0.4-gompi-2023a")
    profile = OrcaProfile(command=orca_path)
    return ORCA(
        profile=profile,
        label=label,
        directory=directory,
        orcasimpleinput=f'MP2 {basis} TightSCF EnGrad',
        orcablocks=f'%maxcore 4000\n%pal nprocs {nprocs} end',
    )


def interpolate_band(images, ts):
    """IDPP interpolation using wB97x TS as midpoint — same as original neb.py."""
    middle_idx = len(images) // 2
    images[middle_idx].set_positions(ts.get_positions())
    first_band = NEB(images[:middle_idx + 1])
    second_band = NEB(images[middle_idx:])
    first_band.interpolate('idpp')
    second_band.interpolate('idpp')


def plot_mep(neb_tools, output):
    fit = neb_tools.get_fit()
    fig, ax = plt.subplots()
    ax.plot(fit.fit_path, fit.fit_energies, label=f"Barrier: {max(fit.fit_energies):.2f} eV")
    ax.set_ylabel("Energy [eV]")
    ax.set_xlabel("Reaction Coordinate [Å]")
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


class CalculationChecker:
    def __init__(self, neb):
        self.neb = neb

    def check(self):
        missing = [
            i for i, img in enumerate(self.neb.images[1:-1])
            if {'forces', 'energy'} - img.calc.results.keys()
        ]
        if missing:
            raise ValueError(f"Missing calculation for image(s): {missing}")


def main(args):
    if not os.path.exists(args.h5file):
        print(f"ERROR: HDF5 file not found: {args.h5file}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Load wB97x endpoints from H5
    print(f"Loading endpoints for {args.reaction} from H5 ...")
    reactant, ts, product = load_endpoints(args.h5file, args.reaction, args.split)

    # Build image list: n_images copies, endpoints fixed
    images = [reactant.copy() for _ in range(args.n_images - 1)] + [product.copy()]

    # Assign CCSD calculators to all images
    for i, atoms in enumerate(images):
        calc_dir = os.path.join(args.output, f'orca/img{i}')
        os.makedirs(calc_dir, exist_ok=True)
        atoms.calc = make_calculator('orca', calc_dir, args.basis, args.nprocs)

    # Relax endpoints with MP2 (or load existing if --skip-relax)
    r_xyz = os.path.join(args.output, 'reactant.xyz')
    p_xyz = os.path.join(args.output, 'product.xyz')

    if args.skip_relax and os.path.exists(r_xyz) and os.path.exists(p_xyz):
        print("Loading previously relaxed endpoints ...")
        from ase.io import read as ase_read
        images[0] = ase_read(r_xyz)
        images[-1] = ase_read(p_xyz)
        for i, idx in enumerate([0, len(images) - 1]):
            calc_dir = os.path.join(args.output, f'orca/img{idx}')
            images[idx].calc = make_calculator('orca', calc_dir, args.basis, args.nprocs)
    else:
        print("Relaxing reactant with MP2 ...")
        BFGS(images[0], logfile=os.path.join(args.output, 'relax_r.log')).run(fmax=0.05)

        print("Relaxing product with MP2 ...")
        BFGS(images[-1], logfile=os.path.join(args.output, 'relax_p.log')).run(fmax=0.05)

        write(r_xyz, images[0])
        write(p_xyz, images[-1])

    # Interpolate intermediate images using wB97x TS as midpoint
    print("Interpolating band with IDPP (wB97x TS as midpoint) ...")
    interpolate_band(images, ts)

    # Run NEB
    print("Running NEB ...")
    neb = NEB(images, climb=False, parallel=args.parallel)
    neb_tools = NEBTools(images)
    relax_neb = LBFGS(neb, logfile=os.path.join(args.output, 'neb.log'), memory=10)

    db_writer = DBWriter(os.path.join(args.output, 'neb.db'), images)
    checker = CalculationChecker(neb)
    fmaxs = []

    relax_neb.attach(checker.check)
    relax_neb.attach(db_writer.write)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))
    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

    # CI-NEB
    print("Running CI-NEB ...")
    neb.climb = True
    converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)

    if converged:
        open(os.path.join(args.output, 'converged'), 'w').close()
        print("CI-NEB converged!")
    else:
        print("WARNING: CI-NEB did not converge within step limit")

    # Save outputs — identical structure to original pipeline
    json.dump(fmaxs, open(os.path.join(args.output, 'fmaxs.json'), 'w'))

    ts_out = max(images, key=lambda x: x.get_potential_energy())
    write(os.path.join(args.output, 'transition_state.xyz'), ts_out)
    write(os.path.join(args.output, 'reactant.xyz'), images[0])
    write(os.path.join(args.output, 'product.xyz'), images[-1])

    plot_mep(neb_tools, args.output)
    print(f"\nDone. Results in {args.output}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCSD NEB — same pipeline as Transition1x')
    parser.add_argument('--h5file', required=True, help='Path to Transition1x.h5')
    parser.add_argument('--reaction', required=True, help='Reaction name (e.g. rxn0103)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n-images', type=int, default=10, help='Number of NEB images (same as T1x: 10)')
    parser.add_argument('--basis', default='cc-pVDZ')
    parser.add_argument('--nprocs', type=int, default=1, help='ORCA MPI processes per image')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--neb-fmax', type=float, default=0.10)
    parser.add_argument('--cineb-fmax', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--skip-relax', action='store_true',
                        help='Skip endpoint relaxation and load existing reactant.xyz/product.xyz')
    parser.add_argument('--parallel', action='store_true',
                        help='Parallel NEB via MPI — run with: mpirun -n <n_images-2> python ccsd_neb.py ... --parallel')
    args = parser.parse_args()
    main(args)
