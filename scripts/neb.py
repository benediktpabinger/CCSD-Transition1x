import argparse
import json
import os

import matplotlib.pyplot as plt

import ase.db
from ase.calculators.orca import ORCA
from ase.io import read, write
from ase.neb import NEB, NEBOptimizer, NEBTools
from ase.optimize.bfgs import BFGS

parser = argparse.ArgumentParser()

parser.add_argument("--reactant", type=str)
parser.add_argument("--product", type=str)
parser.add_argument("--orcabinary", type=str)
parser.add_argument("--transition_state", type=str, default=None)
parser.add_argument("--n_images", type=int, default=10)
parser.add_argument("--output", type=str, default="output")
parser.add_argument("--neb_fmax", type=float, default=0.1)
parser.add_argument("--cineb_fmax", type=float, default=0.05)
parser.add_argument("--steps", type=int, default=500)

args = parser.parse_args()


def main(args):  # pylint: disable=redefined-outer-name
    product = read(args.product)
    reactant = read(args.reactant)

    os.makedirs(args.output, exist_ok=True)
    os.environ["ASE_ORCA_COMMAND"] = f"{args.orcabinary} PREFIX.inp > PREFIX.out"

    atom_configs = [reactant.copy() for i in range(args.n_images - 1)] + [product]

    for i, atom_config in enumerate(atom_configs):
        atom_config.calc = ORCA(
            label=os.path.join(args.output, f"orca/img{i}"),
            orcasimpleinput="wB97X 6-31G(d)",
        )

    print("Relaxing endpoints ... ")
    BFGS(atom_configs[0]).run()
    BFGS(atom_configs[-1]).run()

    print("Interpolating band ... ")
    interpolate_band(atom_configs, args.transition_state)

    print("Running NEB ... ")
    neb = NEB(atom_configs, climb=True, parallel=True)
    calculation_checker = CalculationChecker(neb)
    neb_tools = NEBTools(neb.images)

    relax_neb = NEBOptimizer(neb)
    db_writer = DBWriter(os.path.join(args.output, "neb.db"), atom_configs)
    fmaxs = []

    relax_neb.attach(calculation_checker.check_calculations)
    relax_neb.attach(db_writer.write)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))

    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

    print("NEB has converged, turn on CI-NEB ...")
    neb.climb = True
    converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)

    if converged:
        open(os.path.join(args.output, "converged"), "w")
        print("Reaction converged ... ")

    fig = plot_mep(neb_tools)
    fig.savefig(os.path.join(args.output, "mep.png"))
    json.dump(fmaxs, open(os.path.join(args.output, "fmaxs.json"), "w"))

    transition_state = max(atom_configs, key=lambda x: x.get_potential_energy())
    write(os.path.join(args.output, "transition_state.xyz"), transition_state)
    write(os.path.join(args.output, "transition_state.png"), transition_state)
    write(os.path.join(args.output, "reactant.xyz"), atom_configs[0])
    write(os.path.join(args.output, "reactant.png"), atom_configs[0])
    write(os.path.join(args.output, "product.xyz"), atom_configs[-1])
    write(os.path.join(args.output, "product.png"), atom_configs[-1])


def plot_mep(neb_tools):
    fit = neb_tools.get_fit()

    fig, ax = plt.subplots()
    ax.plot(
        fit.fit_path, fit.fit_energies, label=f"Barrier: {max(fit.fit_energies):.2f} eV"
    )

    ax.patch.set_facecolor("#E8E8E8")
    ax.grid(color="w")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Energy [eV]")
    ax.set_xlabel("Reaction Coordinate [AA]")
    ax.legend()

    return fig


class CalculationChecker:
    def __init__(self, neb):
        self.neb = neb

    def check_calculations(self):
        missing_calculations = []
        for i, image in enumerate(self.neb.images[1:-1]):
            if {"forces", "energy"} - image.calc.results.keys():
                missing_calculations.append(i)

        if missing_calculations:
            raise ValueError(f"missing calculation for image(s) {missing_calculations}")


class DBWriter:
    def __init__(self, db_path, atomss):
        self.atomss = atomss
        self.db_path = db_path

    def write(self):
        with ase.db.connect(self.db_path) as db:
            for atoms in self.atomss:
                if atoms.calc.results:
                    db.write(atoms, data=atoms.calc.results)


def interpolate_band(atom_configs, transition_state=None):
    if transition_state:
        transition_state = read(transition_state)
        ts_positions = transition_state.get_positions()
        middle_idx = len(atom_configs) // 2
        atom_configs[middle_idx].set_positions(ts_positions)
        first_band = NEB(atom_configs[: middle_idx + 1])
        second_band = NEB(atom_configs[middle_idx:])
        first_band.interpolate("idpp")
        second_band.interpolate("idpp")
    else:
        band = NEB(atom_configs)
        band.interpolate("idpp")
    return atom_configs


if __name__ == "__main__":
    main(args)
