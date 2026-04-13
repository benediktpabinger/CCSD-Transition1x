"""
NEB rerun at ωB97M-V/def2-TZVP with two strategies based on prior step count:

  < 70 steps: resume from last images in neb.db (just needed more time)
  ≥ 70 steps: restart from relaxed endpoints (reactant.xyz / product.xyz)
              with ASE linear interpolation for interior images — the T1x
              starting band was likely poor for these reactions

Endpoint relaxation is always skipped (already done in original run).
Reactions without relaxed endpoints are skipped (failed before NEB started).

Usage:
    python orca_neb_rerun.py \
        --reaction rxn0103 \
        --output   ~/orca_neb_results/rxn0103
"""
import argparse
import json
import os
import sys
import shutil

import ase.db
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read, write
from ase.mep import NEB, NEBTools
from ase.mep.neb import NEBOptimizer


def make_orca_calc(orca_cmd, n_threads, scratchdir):
    profile = OrcaProfile(command=orca_cmd)
    return ORCA(
        profile=profile,
        charge=0,
        mult=1,
        orcasimpleinput='wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad',
        orcablocks='%pal nprocs 1 end\n%maxcore 4000\n%scf maxiter 200 end',
        directory=scratchdir,
    )


def count_neb_steps(neb_log):
    """Count completed NEB steps from neb.log."""
    if not os.path.exists(neb_log):
        return 0
    count = 0
    with open(neb_log) as f:
        for line in f:
            if 'NEBOptimizer' in line:
                count += 1
    return count


def load_images_from_db(db_path, n_images=10):
    """Load the last complete band from neb.db."""
    with ase.db.connect(db_path) as db:
        all_rows = list(db.select())
    if len(all_rows) < n_images:
        return None
    last_band = all_rows[-n_images:]
    images = [row.toatoms() for row in last_band]
    print(f"Resumed {n_images} images from neb.db ({len(all_rows)} total entries)")
    return images


def load_images_from_endpoints(r_xyz, p_xyz, n_images=10):
    """Load relaxed endpoints and linearly interpolate interior images."""
    reactant = read(r_xyz)
    product = read(p_xyz)
    images = [reactant.copy() for _ in range(n_images)]
    images[-1] = product.copy()
    neb = NEB(images)
    neb.interpolate()
    print(f"Initialized {n_images} images via linear interpolation from relaxed endpoints")
    return images


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
    for i, atoms in enumerate(images):
        scratch = os.path.join(scratch_base, f'img{i:02d}')
        os.makedirs(scratch, exist_ok=True)
        atoms.calc = make_orca_calc(orca_cmd, n_threads, scratch)


def main(args):
    os.makedirs(args.output, exist_ok=True)

    r_xyz = os.path.join(args.output, 'reactant.xyz')
    p_xyz = os.path.join(args.output, 'product.xyz')
    neb_log = os.path.join(args.output, 'neb.log')
    db_path = os.path.join(args.output, 'neb.db')

    # Skip if endpoints missing (failed before NEB — not recoverable here)
    if not os.path.exists(r_xyz) or not os.path.exists(p_xyz):
        print(f"Skipping {args.reaction}: relaxed endpoints not found (endpoint relaxation likely failed)")
        sys.exit(0)

    scratch_base = os.path.join('/tmp', f'orca_{args.reaction}_{os.getpid()}')
    os.makedirs(scratch_base, exist_ok=True)

    # Decide strategy based on prior step count
    prior_steps = count_neb_steps(neb_log)
    print(f"Prior NEB steps: {prior_steps}")

    if prior_steps < args.step_threshold:
        print(f"< {args.step_threshold} steps — resuming from last neb.db images")
        images = load_images_from_db(db_path, n_images=10)
        if images is None:
            print("WARNING: neb.db missing or incomplete, falling back to endpoint interpolation")
            images = load_images_from_endpoints(r_xyz, p_xyz)
    else:
        print(f">= {args.step_threshold} steps — restarting with linear interpolation from relaxed endpoints")
        images = load_images_from_endpoints(r_xyz, p_xyz)

    assign_calcs(images, args.orca_cmd, args.n_threads, scratch_base)

    # NEB — append to existing log
    print("Running NEB (wB97M-V/def2-TZVP) ...")
    neb = NEB(images, climb=False, parallel=False)
    neb_tools = NEBTools(images)
    relax_neb = NEBOptimizer(neb, logfile=open(neb_log, 'a'))

    db_writer = DBWriter(db_path, images)
    fmaxs = []

    relax_neb.attach(db_writer.write)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))
    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

    # CI-NEB
    print(f"Running CI-NEB (fmax < {args.cineb_fmax}) ...")
    neb.climb = True
    converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)

    if converged:
        open(os.path.join(args.output, 'converged'), 'w').close()
        print("CI-NEB converged!")
    else:
        print("WARNING: CI-NEB did not converge within step limit")

    # Append to existing fmaxs.json
    existing = []
    fmaxs_path = os.path.join(args.output, 'fmaxs.json')
    if os.path.exists(fmaxs_path):
        existing = json.load(open(fmaxs_path))
    json.dump(existing + fmaxs, open(fmaxs_path, 'w'))

    ts_out = max(images, key=lambda x: x.get_potential_energy())
    write(os.path.join(args.output, 'transition_state.xyz'), ts_out)
    write(r_xyz, images[0])
    write(p_xyz, images[-1])

    plot_mep(images, args.output, 'wB97M-V/def2-TZVP')
    shutil.rmtree(scratch_base, ignore_errors=True)

    print(f"\nDone. Results in {args.output}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reaction',       required=True)
    parser.add_argument('--output',         required=True)
    parser.add_argument('--orca-cmd',       default='orca')
    parser.add_argument('--n-threads',      type=int, default=8)
    parser.add_argument('--neb-fmax',       type=float, default=0.15)
    parser.add_argument('--cineb-fmax',     type=float, default=0.05)
    parser.add_argument('--steps',          type=int, default=500)
    parser.add_argument('--step-threshold', type=int, default=70)
    args = parser.parse_args()
    main(args)
