"""
NEB at the OMol25 level of theory using ORCA via ASE.

This is a NEW script; pipeline/orca_neb.py and ~/orca_neb_results/ are left
untouched so the existing benchmark stays reproducible.

Level of theory follows OMol25 (arXiv:2505.08762, Sec. 2.7 + App. A):
    wB97M-V / def2-TZVPD, RI-J + COSX, TightSCF, DEFGRID3,
    integral threshold 1e-12, primitive batch threshold 1e-13
Deviation: ORCA 5.0.4 instead of 6.0.0 (6.0.0 is not installed on this
cluster). The paper notes these thresholds became ORCA defaults in later
versions; here they are set explicitly, which reproduces that behaviour.

Differences from pipeline/orca_neb.py -- ONLY these two:
  1. level of theory (def2-TZVP -> def2-TZVPD, + DEFGRID3 + thresholds)
  2. ORCA runs MPI-parallel (nprocs > 1) instead of serial
The NEB algorithm itself is byte-for-byte the same protocol:
    10 images from H5 (R + last 8 interior + P)
    BFGS endpoint relaxation to fmax 0.05
    NEB (climb=False) to fmax 0.15, then CI-NEB (climb=True) to fmax 0.05
    NEBOptimizer, max 500 steps, TS = highest-energy image

Endpoints are re-relaxed at this level by default: the def2-TZVP geometries
are not minima on the def2-TZVPD/DEFGRID3 surface, and keeping them would
make every barrier a mixture of two levels of theory. Use --skip-relax only
for restarts.

Usage:
    python orca_neb_omol25.py \
        --h5file   ~/data/Transition1x.h5 \
        --reaction rxn0103 \
        --output   ~/orca_neb_omol25/rxn0103 \
        --nprocs   12
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
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read, write
from ase.mep import NEB, NEBTools
from ase.mep.neb import NEBOptimizer
from ase.optimize.bfgs import BFGS

LEVEL = 'wB97M-V/def2-TZVPD (OMol25 settings)'

SIMPLEINPUT = ('wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3 EnGrad')


def resolve_orca(orca_cmd):
    """ORCA's MPI startup requires the ABSOLUTE path to the binary.
    Passing a bare 'orca' works serially but breaks with nprocs > 1."""
    if os.path.isabs(orca_cmd):
        return orca_cmd
    full = shutil.which(orca_cmd)
    if not full:
        print(f'ERROR: ORCA executable not found on PATH: {orca_cmd}')
        sys.exit(1)
    return full


def make_orca_calc(orca_cmd, nprocs, maxcore, scratchdir):
    """ASE ORCA calculator at OMol25 settings, MPI-parallel."""
    blocks = (
        f'%pal nprocs {nprocs} end\n'
        f'%maxcore {maxcore}\n'
        '%scf\n'
        '  maxiter 200\n'
        '  Thresh 1e-12\n'
        '  TCut   1e-13\n'
        'end'
    )
    return ORCA(
        profile=OrcaProfile(command=orca_cmd),
        charge=0,
        mult=1,
        orcasimpleinput=SIMPLEINPUT,
        orcablocks=blocks,
        directory=scratchdir,
    )


def load_wB97x_images(h5file, reaction, split):
    """Load the final wB97x NEB images from H5 as starting band.
    Identical selection to pipeline/orca_neb.py: R + last 8 interior + P."""
    with h5py.File(h5file, 'r') as f:
        if split not in f:
            raise ValueError(f"split '{split}' not in {h5file}")
        split_group = f[split]
        for formula in split_group:
            if reaction in split_group[formula]:
                rxn_group = split_group[formula][reaction]
                positions = rxn_group['positions'][:]
                atomic_numbers = rxn_group['atomic_numbers'][:]
                total = positions.shape[0]
                final_positions = [positions[0]] + list(positions[-8:]) + [positions[9]]
                images = [Atoms(numbers=atomic_numbers, positions=pos)
                          for pos in final_positions]
                print(f'Loaded 10 wB97x images from H5 ({total} total configs)')
                return images
    raise ValueError(f"Reaction '{reaction}' not found in split '{split}' of {h5file}")


def load_images_from_db(db_path, n_images=10):
    """Das zuletzt geschriebene Band aus neb.db.

    DBWriter haengt nach jedem Optimierungsschritt alle n_images Bilder an,
    die letzten n_images Zeilen sind also das Band, wie es beim Abbruch stand.
    """
    with ase.db.connect(db_path) as db:
        rows = list(db.select())
    if len(rows) < n_images:
        raise ValueError('neb.db hat nur %d Eintraege, gebraucht werden %d'
                         % (len(rows), n_images))
    images = [r.toatoms() for r in rows[-n_images:]]
    print('Warmstart: %d Bilder aus neb.db (%d Eintraege, also %d Schritte)'
          % (n_images, len(rows), len(rows) // n_images))
    return images


def plot_mep(images, output, label):
    neb_tools = NEBTools(images)
    fit = neb_tools.get_fit()
    fig, ax = plt.subplots()
    ax.plot(fit.fit_path, fit.fit_energies, 'b-o', markersize=4,
            label=f'{label} (barrier: {max(fit.fit_energies):.3f} eV)')
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


def assign_calcs(images, orca_cmd, nprocs, maxcore, scratch_base):
    for i, atoms in enumerate(images):
        scratch = os.path.join(scratch_base, f'img{i:02d}')
        os.makedirs(scratch, exist_ok=True)
        atoms.calc = make_orca_calc(orca_cmd, nprocs, maxcore, scratch)


def main(args):
    if not os.path.exists(args.h5file):
        print(f'ERROR: HDF5 file not found: {args.h5file}')
        sys.exit(1)

    orca_cmd = resolve_orca(args.orca_cmd)
    os.makedirs(args.output, exist_ok=True)

    print(f'Level: {LEVEL}')
    print(f'ORCA:  {orca_cmd}  (nprocs={args.nprocs})')

    # provenance, so the output directory is self-describing
    json.dump({'level': LEVEL,
               'orcasimpleinput': SIMPLEINPUT,
               'thresh': '1e-12', 'tcut': '1e-13',
               'orca_version': '5.0.4 (OMol25 used 6.0.0)',
               'nprocs': args.nprocs,
               'neb_fmax': args.neb_fmax, 'cineb_fmax': args.cineb_fmax,
               'endpoints_relaxed_at_this_level': not (args.skip_relax
                                                       or args.resume),
               'resumed_from_neb_db': bool(args.resume)},
              open(os.path.join(args.output, 'level.json'), 'w'), indent=1)

    scratch_base = os.path.join('/tmp', f'orca25_{args.reaction}_{os.getpid()}')
    os.makedirs(scratch_base, exist_ok=True)

    db_path = os.path.join(args.output, 'neb.db')
    if args.resume:
        images = load_images_from_db(db_path)
    else:
        print(f'Loading wB97x NEB images for {args.reaction} ...')
        images = load_wB97x_images(args.h5file, args.reaction, args.split)
    assign_calcs(images, orca_cmd, args.nprocs, args.maxcore, scratch_base)

    r_xyz = os.path.join(args.output, 'reactant.xyz')
    p_xyz = os.path.join(args.output, 'product.xyz')

    if (args.skip_relax or args.resume) and os.path.exists(r_xyz) \
            and os.path.exists(p_xyz):
        print('Skipping endpoint relaxation (loading existing xyz) ...')
        images[0].set_positions(read(r_xyz).get_positions())
        images[-1].set_positions(read(p_xyz).get_positions())
    else:
        print(f'Relaxing reactant at {LEVEL} ...')
        BFGS(images[0], logfile=os.path.join(args.output, 'relax_r.log')).run(fmax=0.05)
        write(r_xyz, images[0])
        print(f'Relaxing product at {LEVEL} ...')
        BFGS(images[-1], logfile=os.path.join(args.output, 'relax_p.log')).run(fmax=0.05)
        write(p_xyz, images[-1])

    print(f'Running NEB ({LEVEL}) ...')
    neb = NEB(images, climb=False, parallel=False)
    neb_tools = NEBTools(images)
    relax_neb = NEBOptimizer(neb, logfile=os.path.join(args.output, 'neb.log'))

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

    plot_mep(images, args.output, LEVEL)
    shutil.rmtree(scratch_base, ignore_errors=True)

    print(f'\nDone. Results in {args.output}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file',     required=True)
    parser.add_argument('--reaction',   required=True)
    parser.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output',     required=True)
    parser.add_argument('--orca-cmd',   default='orca')
    parser.add_argument('--nprocs',     type=int, default=12,
                        help='ORCA MPI ranks; SLURM must allocate --ntasks=nprocs')
    parser.add_argument('--maxcore',    type=int, default=3000)
    parser.add_argument('--neb-fmax',   type=float, default=0.15)
    parser.add_argument('--cineb-fmax', type=float, default=0.05)
    parser.add_argument('--steps',      type=int, default=500)
    parser.add_argument('--resume', action='store_true',
                        help='Warmstart aus <output>/neb.db statt aus dem H5; '
                             'die Endpunkte werden dabei nicht neu relaxiert')
    parser.add_argument('--skip-relax', action='store_true',
                        help='restart only; endpoints must already be at THIS level')
    args = parser.parse_args()
    main(args)
