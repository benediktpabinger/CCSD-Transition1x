"""
NEB resume at ωB97M-V/def2-TZVP — always warm-starts from last neb.db images.

Strategy:
  - Load the last 10 images from neb.db (the band as it was when the previous
    run stopped). No cold restart, no linear interpolation.
  - Fall back to endpoint interpolation only if neb.db is missing or has
    fewer than 10 images (i.e. NEB never started in the previous run).

This avoids throwing away the already-optimised Phase 1 band, which was the
problem with the previous rerun script's >=70 step cold-restart logic.

Usage:
    python orca_neb_resume.py \
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


def count_first_run_steps(neb_log):
    """
    Count the number of NEB steps in the first run section of neb.log.
    Each run starts with a header line containing 'Step     Time     fmax'.
    Returns 0 if log is missing or empty.
    """
    if not os.path.exists(neb_log):
        return 0
    steps = 0
    in_first_run = False
    with open(neb_log) as f:
        for line in f:
            if 'Step' in line and 'Time' in line and 'fmax' in line:
                if in_first_run:
                    # Second header = start of run 2, stop counting
                    break
                in_first_run = True
            elif in_first_run and 'NEBOptimizer' in line:
                steps += 1
    return steps


def load_images_from_db(db_path, n_images=10, max_entries=None):
    """
    Load the last complete band from neb.db.

    If max_entries is given, only consider the first max_entries rows —
    this lets us load the band from the end of the first run rather than
    the end of the most recent (cold-restarted) run.
    """
    if not os.path.exists(db_path):
        return None
    with ase.db.connect(db_path) as db:
        all_rows = list(db.select())
    if max_entries is not None:
        all_rows = all_rows[:max_entries]
    if len(all_rows) < n_images:
        return None
    last_band = all_rows[-n_images:]
    images = [row.toatoms() for row in last_band]
    label = f"first-run band" if max_entries is not None else "latest band"
    print(f"Loaded {n_images} images from neb.db ({label}, used {len(all_rows)} entries)")
    return images


def load_images_from_endpoints(r_xyz, p_xyz, n_images=10):
    """Load relaxed endpoints and linearly interpolate interior images."""
    reactant = read(r_xyz)
    product  = read(p_xyz)
    images   = [reactant.copy() for _ in range(n_images)]
    images[-1] = product.copy()
    neb = NEB(images)
    neb.interpolate()
    print(f"Initialised {n_images} images via linear interpolation (neb.db unavailable)")
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
        self.images   = images
        self.db_path  = db_path

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

    r_xyz    = os.path.join(args.output, 'reactant.xyz')
    p_xyz    = os.path.join(args.output, 'product.xyz')
    neb_log  = os.path.join(args.output, 'neb.log')
    db_path  = os.path.join(args.output, 'neb.db')

    # Safety: never overwrite a reaction that already converged
    if os.path.exists(os.path.join(args.output, 'converged')):
        print(f"Skipping {args.reaction}: already converged, not touching it")
        sys.exit(0)

    if not os.path.exists(r_xyz) or not os.path.exists(p_xyz):
        print(f"Skipping {args.reaction}: relaxed endpoints not found")
        sys.exit(0)

    scratch_base = os.path.join('/tmp', f'orca_{args.reaction}_{os.getpid()}')
    os.makedirs(scratch_base, exist_ok=True)

    # Load the band from the END OF THE FIRST RUN — this is the best-quality
    # band we have. The first run passed Phase 1 NEB (fmax → ~0.15 eV/Å), so
    # its final band is well-optimised. Later cold-restarts threw this away
    # and started from linear interpolation, which is worse.
    #
    # Strategy:
    #   1. Count steps in section 1 of neb.log → n1
    #   2. Load neb.db[:n1*10][-10:] = last band from run 1
    #   3. Fall back to latest neb.db band if run-1 slice is too small
    #   4. Fall back to endpoint interpolation if neb.db is missing/empty

    n1 = count_first_run_steps(neb_log)
    print(f"First-run steps in neb.log: {n1}")

    images = None
    if n1 > 0:
        images = load_images_from_db(db_path, n_images=10, max_entries=n1 * 10)
    if images is None:
        print("First-run band unavailable — trying latest neb.db band")
        images = load_images_from_db(db_path, n_images=10)
    if images is None:
        print("neb.db missing or incomplete — falling back to endpoint interpolation")
        images = load_images_from_endpoints(r_xyz, p_xyz)

    assign_calcs(images, args.orca_cmd, args.n_threads, scratch_base)

    print("Running NEB (wB97M-V/def2-TZVP) ...")
    neb = NEB(images, climb=False, parallel=False)
    neb_tools = NEBTools(images)
    relax_neb = NEBOptimizer(neb, logfile=open(neb_log, 'a'))

    db_writer = DBWriter(db_path, images)
    fmaxs = []

    relax_neb.attach(db_writer.write)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))
    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

    print(f"Running CI-NEB (fmax < {args.cineb_fmax}) ...")
    neb.climb = True
    converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)

    if converged:
        open(os.path.join(args.output, 'converged'), 'w').close()
        print("CI-NEB converged!")
    else:
        print("WARNING: CI-NEB did not converge within step limit")

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
    parser.add_argument('--reaction',   required=True)
    parser.add_argument('--output',     required=True)
    parser.add_argument('--orca-cmd',   default='orca')
    parser.add_argument('--n-threads',  type=int,   default=8)
    parser.add_argument('--neb-fmax',   type=float, default=0.15)
    parser.add_argument('--cineb-fmax', type=float, default=0.05)
    parser.add_argument('--steps',      type=int,   default=500)
    args = parser.parse_args()
    main(args)
