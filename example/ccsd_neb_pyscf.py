"""
True CCSD/cc-pVDZ NEB using PySCF analytic gradients.

PySCF has analytic CCSD gradients (unlike ORCA 5 MDCI), so this script
runs genuine CCSD geometry optimization along the NEB path — not just
single-points after MP2.

Pipeline:
  1. Load final wB97x NEB images from Transition1x.h5 as starting band
  2. Relax endpoints with CCSD/BFGS (analytic gradients via PySCF)
  3. Run NEB → CI-NEB with CCSD using NEBOptimizer (same as Transition1x)
  4. Save neb.db + fmaxs.json (identical structure to original wB97x pipeline)

Usage:
    python ccsd_neb_pyscf.py --h5file ~/data/Transition1x.h5 --reaction rxn0103 --output ~/ccsd_pyscf/rxn0103
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
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.mep import NEB, NEBTools
from ase.mep.neb import NEBOptimizer
from ase.optimize.bfgs import BFGS
from pyscf import gto, scf, cc
from pyscf.grad import ccsd as ccsd_grad


class PySCFCCSD(Calculator):
    """ASE calculator wrapping PySCF CCSD with analytic gradients."""

    implemented_properties = ['energy', 'forces']

    def __init__(self, basis='cc-pVDZ', n_threads=1, verbose=3, **kwargs):
        super().__init__(**kwargs)
        self.basis = basis
        self.n_threads = n_threads
        self.verbose = verbose

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        import os
        os.environ['OMP_NUM_THREADS'] = str(self.n_threads)

        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()  # Angstrom

        mol = gto.M(
            atom=[(s, tuple(p)) for s, p in zip(symbols, positions)],
            basis=self.basis,
            unit='Angstrom',
            verbose=self.verbose,
        )

        mf = scf.RHF(mol)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF did not converge")

        mycc = cc.RCCSD(mf)
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError("CCSD did not converge")

        energy_ha = mycc.e_tot  # Hartree

        gradients = ccsd_grad.Gradients(mycc)
        grad_ha_bohr = gradients.kernel()  # Hartree/Bohr, shape (N_atoms, 3)

        # Convert to ASE units: eV and eV/Å
        self.results['energy'] = energy_ha * units.Hartree
        self.results['forces'] = -grad_ha_bohr * units.Hartree / units.Bohr


def load_wB97x_images(h5file, reaction, split, n_images):
    """Load the final wB97x NEB iteration from H5 as starting band.

    H5 structure (from combine_dbs.py):
      /{split}/{formula}/{rxn}/positions      — (N_configs, N_atoms, 3)
      /{split}/{formula}/{rxn}/atomic_numbers — (N_atoms,)

    First iteration: 10 images (R + 8 interior + P)
    Later iterations: 8 interior images only

    Final band: R = positions[0], P = positions[9], interior = positions[-8:]
    """
    with h5py.File(h5file, 'r') as f:
        split_group = f[split]
        for formula in split_group:
            if reaction in split_group[formula]:
                rxn_group = split_group[formula][reaction]
                positions = rxn_group['positions'][:]
                atomic_numbers = rxn_group['atomic_numbers'][:]
                total = positions.shape[0]

                final_positions = [positions[0]] + list(positions[-8:]) + [positions[9]]
                images = [
                    Atoms(numbers=atomic_numbers, positions=pos)
                    for pos in final_positions
                ]
                print(f"Loaded 10 wB97x images from H5 "
                      f"(R + last 8 interior + P, from {total} total configs)")
                return images

    raise ValueError(f"Reaction '{reaction}' not found in split '{split}' of {h5file}")


def plot_mep(neb_tools, output):
    fit = neb_tools.get_fit()
    fig, ax = plt.subplots()
    ax.plot(fit.fit_path, fit.fit_energies, 'b-o', markersize=4,
            label=f"CCSD (barrier: {max(fit.fit_energies):.2f} eV)")
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

    # Assign CCSD calculators to all images
    for i, atoms in enumerate(images):
        atoms.calc = PySCFCCSD(basis=args.basis, n_threads=args.n_threads)

    # Relax endpoints with CCSD — skip if already done
    r_xyz = os.path.join(args.output, 'reactant.xyz')
    p_xyz = os.path.join(args.output, 'product.xyz')

    if args.skip_relax and os.path.exists(r_xyz) and os.path.exists(p_xyz):
        print("Skipping endpoint relaxation — loading existing reactant.xyz and product.xyz ...")
        images[0].set_positions(read(r_xyz).get_positions())
        images[-1].set_positions(read(p_xyz).get_positions())
    else:
        print("Relaxing reactant with CCSD ...")
        BFGS(images[0], logfile=os.path.join(args.output, 'relax_r.log')).run(fmax=0.05)

        print("Relaxing product with CCSD ...")
        BFGS(images[-1], logfile=os.path.join(args.output, 'relax_p.log')).run(fmax=0.05)

        write(r_xyz, images[0])
        write(p_xyz, images[-1])

    # NEB with NEBOptimizer — same as original Transition1x pipeline
    print("Running NEB (CCSD) ...")
    neb = NEB(images, climb=False, parallel=False)
    neb_tools = NEBTools(images)
    relax_neb = NEBOptimizer(neb, logfile=os.path.join(args.output, 'neb.log'))

    db_writer = DBWriter(os.path.join(args.output, 'neb.db'), images)
    checker = CalculationChecker(neb)
    fmaxs = []

    relax_neb.attach(checker.check)
    relax_neb.attach(db_writer.write)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))
    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

    # CI-NEB
    print("Running CI-NEB (CCSD) ...")
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

    plot_mep(neb_tools, args.output)

    print(f"\nDone. Results in {args.output}/")
    print(f"  neb.db      — CCSD energies + forces (all NEB iterations)")
    print(f"  fmaxs.json  — convergence history")
    print(f"  mep.png     — minimum energy path")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='True CCSD NEB using PySCF analytic gradients')
    parser.add_argument('--h5file', required=True, help='Path to Transition1x.h5')
    parser.add_argument('--reaction', required=True, help='Reaction name (e.g. rxn0103)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n-images', type=int, default=10, help='Number of NEB images')
    parser.add_argument('--basis', default='cc-pVDZ')
    parser.add_argument('--n-threads', type=int, default=4,
                        help='OpenMP threads for PySCF integrals (set = --cpus-per-task)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--neb-fmax', type=float, default=0.10)
    parser.add_argument('--cineb-fmax', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--skip-relax', action='store_true',
                        help='Skip endpoint relaxation if reactant.xyz/product.xyz already exist')
    args = parser.parse_args()
    main(args)
