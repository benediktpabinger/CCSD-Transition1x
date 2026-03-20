"""
MP2/cc-pVDZ NEB warm-started from wB97x NEB images in Transition1x.h5.

Differences from mp2_neb.py (IDPP version):
  - Step 3 uses the final wB97x NEB iteration images directly as starting band
    instead of IDPP interpolation from R/TS/P.
  - This gives a better starting point: physically reasonable geometries already
    close to the MP2 path, so fewer NEB steps needed.

Pipeline:
  1. Load final wB97x NEB images from Transition1x.h5 (last n_images configs)
  2. Relax endpoints with MP2/BFGS (analytic gradients)
  3. Run NEB → CI-NEB with MP2 using wB97x images as starting band
  4. Compute CCSD single-point energies on the final converged images
  5. Save neb.db + fmaxs.json (MP2 forces) + ccsd_singlepoints.json (CCSD energies)

Usage:
    python mp2_neb_warmstart.py --h5file ~/data/Transition1x.h5 --reaction rxn0103 --output ~/neb_results/rxn0103
"""
import argparse
import json
import os
import shutil
import sys

import ase.db
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import write
from ase.mep import NEB, NEBTools
from ase.mep.neb import NEBOptimizer
from ase.optimize.bfgs import BFGS
from transition1x import Dataloader


def load_wB97x_images(h5file, reaction, split, n_images):
    """Load the last n_images configs from the wB97x NEB path in H5.

    The H5 stores all subsampled NEB iterations concatenated along axis 0.
    The last n_images entries correspond to the final converged NEB iteration.
    """
    dl = Dataloader(h5file, split, only_final=False)
    for mol in dl:
        if mol['rxn'] == reaction:
            positions = mol['positions']    # shape: (N, n_atoms, 3)
            atomic_numbers = mol['atomic_numbers']  # shape: (n_atoms,)

            total = positions.shape[0]
            if total < n_images:
                raise ValueError(
                    f"Only {total} configs in H5 for {reaction}, need {n_images}"
                )

            # Take last n_images — these are the final NEB iteration
            final_positions = positions[-n_images:]
            images = [
                Atoms(numbers=atomic_numbers, positions=pos)
                for pos in final_positions
            ]
            print(f"Loaded {n_images} wB97x images from H5 "
                  f"(last {n_images} of {total} total configs)")
            return images

    raise ValueError(f"Reaction '{reaction}' not found in split '{split}' of {h5file}")


def load_endpoints(h5file, reaction, split):
    """Load reactant and product from Transition1x H5 (for verification)."""
    dl = Dataloader(h5file, split, only_final=True)
    for mol in dl:
        if mol['rxn'] == reaction:
            def to_atoms(d):
                return Atoms(numbers=d['atomic_numbers'], positions=d['positions'])
            return to_atoms(mol['reactant']), to_atoms(mol['product'])
    raise ValueError(f"Reaction '{reaction}' not found in split '{split}' of {h5file}")


def get_orca_path():
    orca_path = shutil.which('orca')
    if orca_path is None:
        raise RuntimeError("ORCA not found in PATH. Load with: module load ORCA/5.0.4-gompi-2023a")
    return orca_path


def make_mp2_calculator(label, directory, basis, nprocs):
    profile = OrcaProfile(command=get_orca_path())
    return ORCA(
        profile=profile,
        label=label,
        directory=directory,
        orcasimpleinput=f'MP2 {basis} TightSCF EnGrad',
        orcablocks=f'%maxcore 4000\n%pal nprocs {nprocs} end',
    )


def make_ccsd_calculator(label, directory, basis, nprocs):
    profile = OrcaProfile(command=get_orca_path())
    return ORCA(
        profile=profile,
        label=label,
        directory=directory,
        orcasimpleinput=f'CCSD {basis} TightSCF',
        orcablocks=f'%maxcore 4000\n%pal nprocs {nprocs} end',
    )


def run_ccsd_singlepoints(images, output, basis, nprocs):
    print("\nRunning CCSD single-point energies on final images ...")
    results = []
    for i, atoms in enumerate(images):
        print(f"  Image {i}/{len(images)-1} ...")
        calc_dir = os.path.join(output, f'orca_ccsd/img{i}')
        os.makedirs(calc_dir, exist_ok=True)
        atoms_copy = atoms.copy()
        atoms_copy.calc = make_ccsd_calculator('orca', calc_dir, basis, nprocs)
        try:
            energy = atoms_copy.get_potential_energy()
            results.append({'image': i, 'ccsd_energy': energy, 'success': True})
            print(f"    CCSD energy: {energy:.6f} eV")
        except Exception as e:
            results.append({'image': i, 'ccsd_energy': None, 'success': False, 'error': str(e)})
            print(f"    ERROR: {e}")

    out_path = os.path.join(output, 'ccsd_singlepoints.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    successful = sum(r['success'] for r in results)
    print(f"CCSD single-points: {successful}/{len(results)} successful → {out_path}")
    return results


def plot_mep(neb_tools, output, ccsd_results=None):
    fit = neb_tools.get_fit()
    fig, ax = plt.subplots()
    ax.plot(fit.fit_path, fit.fit_energies, 'b-o', markersize=4,
            label=f"MP2 (barrier: {max(fit.fit_energies):.2f} eV)")

    if ccsd_results and all(r['success'] for r in ccsd_results):
        ccsd_energies = np.array([r['ccsd_energy'] for r in ccsd_results])
        ccsd_rel = ccsd_energies - ccsd_energies.min()
        x = np.linspace(fit.fit_path[0], fit.fit_path[-1], len(ccsd_rel))
        ax.plot(x, ccsd_rel, 'r-s', markersize=4,
                label=f"CCSD (barrier: {ccsd_rel.max():.2f} eV)")

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

    # Load final wB97x NEB images as starting band
    print(f"Loading wB97x NEB images for {args.reaction} from H5 ...")
    images = load_wB97x_images(args.h5file, args.reaction, args.split, args.n_images)

    # Assign MP2 calculators to all images
    for i, atoms in enumerate(images):
        calc_dir = os.path.join(args.output, f'orca/img{i}')
        os.makedirs(calc_dir, exist_ok=True)
        atoms.calc = make_mp2_calculator('orca', calc_dir, args.basis, args.nprocs)

    # Relax endpoints with MP2
    print("Relaxing reactant with MP2 ...")
    BFGS(images[0], logfile=os.path.join(args.output, 'relax_r.log')).run(fmax=0.05)

    print("Relaxing product with MP2 ...")
    BFGS(images[-1], logfile=os.path.join(args.output, 'relax_p.log')).run(fmax=0.05)

    write(os.path.join(args.output, 'reactant.xyz'), images[0])
    write(os.path.join(args.output, 'product.xyz'), images[-1])

    # NEB with NEBOptimizer — same as original Transition1x pipeline
    print("Running NEB (MP2) ...")
    neb = NEB(images, climb=True, parallel=args.parallel)
    neb_tools = NEBTools(images)
    relax_neb = NEBOptimizer(neb, logfile=os.path.join(args.output, 'neb.log'))

    db_writer = DBWriter(os.path.join(args.output, 'neb.db'), images)
    checker = CalculationChecker(neb)
    fmaxs = []

    relax_neb.attach(checker.check)
    relax_neb.attach(db_writer.write)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))
    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

    # CI-NEB — same optimizer object, preserves history
    print("Running CI-NEB (MP2) ...")
    neb.climb = True
    converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)

    if converged:
        open(os.path.join(args.output, 'converged'), 'w').close()
        print("CI-NEB converged!")
    else:
        print("WARNING: CI-NEB did not converge within step limit")

    json.dump(fmaxs, open(os.path.join(args.output, 'fmaxs.json'), 'w'))

    ts_out = max(images, key=lambda x: x.get_potential_energy())
    write(os.path.join(args.output, 'transition_state.xyz'), ts_out)
    write(os.path.join(args.output, 'reactant.xyz'), images[0])
    write(os.path.join(args.output, 'product.xyz'), images[-1])

    # CCSD single-points on final converged images
    ccsd_results = run_ccsd_singlepoints(images, args.output, args.basis, args.nprocs)

    plot_mep(neb_tools, args.output, ccsd_results)

    print(f"\nDone. Results in {args.output}/")
    print(f"  neb.db                 — MP2 energies + forces (all NEB iterations)")
    print(f"  fmaxs.json             — convergence history")
    print(f"  ccsd_singlepoints.json — CCSD energies on final path")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MP2 NEB warm-started from wB97x images')
    parser.add_argument('--h5file', required=True, help='Path to Transition1x.h5')
    parser.add_argument('--reaction', required=True, help='Reaction name (e.g. rxn0103)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n-images', type=int, default=10, help='Number of NEB images')
    parser.add_argument('--basis', default='cc-pVDZ')
    parser.add_argument('--nprocs', type=int, default=1, help='ORCA processes per image')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--neb-fmax', type=float, default=0.10)
    parser.add_argument('--cineb-fmax', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()
    main(args)
