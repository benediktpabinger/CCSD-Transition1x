"""
NEB at ωB97M-V/def2-TZVP using ORCA via ASE.

Pipeline:
  1. Load final wB97x NEB images from Transition1x.h5 as starting band
  2. Relax endpoints with ωB97M-V/def2-TZVP / BFGS
  3. Run NEB → CI-NEB using NEBOptimizer
  4. Save neb.db + fmaxs.json (identical structure to ccsd_neb_pyscf.py)

Results stored in ~/orca_neb_results/{reaction}/

Usage:
    python orca_neb.py \
        --h5file   ~/data/Transition1x.h5 \
        --reaction rxn0103 \
        --output   ~/orca_neb_results/rxn0103
"""
import argparse
import json
import os
import sys
import shutil
import tempfile

import ase.db
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, units
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read, write
from ase.mep import NEB, NEBTools
from ase.mep.neb import NEBOptimizer
from ase.optimize.bfgs import BFGS


def make_orca_calc(orca_cmd, n_threads, scratchdir):
    """Build an ASE ORCA calculator for ωB97M-V/def2-TZVP.

    ORCA parallel: nprocs > 1 requires OpenMPI launched via mpirun.
    On DTU cluster we use nprocs=1 and rely on OpenMP threads for
    the integral engine (set via OMP_NUM_THREADS in job script).
    """
    profile = OrcaProfile(command=orca_cmd)
    return ORCA(
        profile=profile,
        charge=0,
        mult=1,
        orcasimpleinput='wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad',
        orcablocks='%pal nprocs 1 end\n%maxcore 4000\n%scf maxiter 200 end',
        directory=scratchdir,
    )


def load_wB97x_images(h5file, reaction, split):
    """Load the final wB97x NEB images from H5 as starting band."""
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
                print(f"Loaded 10 wB97x images from H5 ({total} total configs)")
                return images

    raise ValueError(f"Reaction '{reaction}' not found in split '{split}' of {h5file}")


def plot_mep(images, output, functional):
    neb_tools = NEBTools(images)
    fit = neb_tools.get_fit()
    fig, ax = plt.subplots()
    ax.plot(fit.fit_path, fit.fit_energies, 'b-o', markersize=4,
            label=f"{functional} (barrier: {max(fit.fit_energies):.3f} eV)")
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


def assign_calcs(images, orca_cmd, n_threads, scratch_base):
    """Assign a fresh ORCA calculator (with own scratch dir) to each image."""
    for i, atoms in enumerate(images):
        scratch = os.path.join(scratch_base, f'img{i:02d}')
        os.makedirs(scratch, exist_ok=True)
        atoms.calc = make_orca_calc(orca_cmd, n_threads, scratch)


def main(args):
    if not os.path.exists(args.h5file):
        print(f"ERROR: HDF5 file not found: {args.h5file}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Scratch dir for ORCA temp files (use local /tmp for speed)
    scratch_base = os.path.join('/tmp', f'orca_{args.reaction}_{os.getpid()}')
    os.makedirs(scratch_base, exist_ok=True)

    print(f"Loading wB97x NEB images for {args.reaction} ...")
    images = load_wB97x_images(args.h5file, args.reaction, args.split)

    assign_calcs(images, args.orca_cmd, args.n_threads, scratch_base)

    # Relax endpoints
    r_xyz = os.path.join(args.output, 'reactant.xyz')
    p_xyz = os.path.join(args.output, 'product.xyz')

    if args.skip_relax and os.path.exists(r_xyz) and os.path.exists(p_xyz):
        print("Skipping endpoint relaxation (loading existing xyz) ...")
        images[0].set_positions(read(r_xyz).get_positions())
        images[-1].set_positions(read(p_xyz).get_positions())
    else:
        print("Relaxing reactant ...")
        BFGS(images[0], logfile=os.path.join(args.output, 'relax_r.log')).run(fmax=0.05)
        write(r_xyz, images[0])

        print("Relaxing product ...")
        BFGS(images[-1], logfile=os.path.join(args.output, 'relax_p.log')).run(fmax=0.05)
        write(p_xyz, images[-1])

    # NEB
    print("Running NEB (wB97M-V/def2-TZVP) ...")
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
    print("Running CI-NEB ...")
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
    write(r_xyz, images[0])
    write(p_xyz, images[-1])

    plot_mep(images, args.output, 'wB97M-V/def2-TZVP')

    # Cleanup scratch
    shutil.rmtree(scratch_base, ignore_errors=True)

    print(f"\nDone. Results in {args.output}/")
    print(f"  neb.db      — wB97M-V/def2-TZVP energies + forces")
    print(f"  fmaxs.json  — convergence history")
    print(f"  mep.png     — minimum energy path")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file',    required=True)
    parser.add_argument('--reaction',  required=True)
    parser.add_argument('--split',     default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output',    required=True)
    parser.add_argument('--orca-cmd',  default='orca', help='Path to ORCA executable')
    parser.add_argument('--n-threads', type=int, default=8)
    parser.add_argument('--neb-fmax',  type=float, default=0.15)
    parser.add_argument('--cineb-fmax',type=float, default=0.05)
    parser.add_argument('--steps',     type=int, default=500)
    parser.add_argument('--skip-relax',action='store_true')
    args = parser.parse_args()
    main(args)
