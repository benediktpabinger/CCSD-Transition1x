import h5py

REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}


def get_molecular_reference_energy(atomic_numbers):
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


class Dataloader:
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            for grp in f.values():
                formula = grp.name.lstrip("/")
                energies = grp["wB97x/6-31G(d).energy"]
                forces = grp["wB97x/6-31G(d).forces"]
                atomic_numbers = list(grp["atomic_numbers"])
                positions = grp["positions"]
                molecular_reference_energy = get_molecular_reference_energy(
                    atomic_numbers
                )

                for energy, force, positions in zip(energies, forces, positions):
                    d = {
                        "wB97x/6-31G(d).energy": energy.__float__(),
                        "wB97x/6-31G(d).atomization_energy": energy - molecular_reference_energy.__float__(),
                        "wB97x/6-31G(d).forces": force.tolist(),
                        "positions": positions,
                        "formula": formula,
                        "atomic_numbers": atomic_numbers,
                    }

                    yield d
