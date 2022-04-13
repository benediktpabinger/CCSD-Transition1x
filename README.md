#### Installation
To install, run:
```
$ git clone {address}
$ cd T1x
$ pip install .
```
if you want to run the ase_db.py example instead to install dependencies

```
$ pip install '.[example]'
```

#### Usage
In python run

```
import t1x.Dataloader

dataloader = t1x.Dataloader(path_to_h5_file)
for molecule in dataloader:
    energy = molecule["wB97x/6-31G(d).energy"]
    ...
```

The elements in the data loader each represent a single molecule. It is a dictionary that has the following keys available:
*    formula:                           chemical formula for the molecule.
*    positions:                         list of x, y, z coordinates of all atoms in the molecule in √Ö.
*    atomic_numbers:                    list of atomic numbers ordered in the same way as positions.
*    wB97x/6-31G(d).energy:             total energy of molecule
*    wB97x/6-31G(d).atomization_energy: atomization energy of molecule
*    wB97x/6-31G(d).forces:             list of x, y, z forces on each atom ordered in the same way as positions.


#### ase_db example
The ase\_db.py example we generate an ase.db database where each row has energy, forces and atomization\_energy in the data-field.


#### Reference Energies

The reference energies used to calculate atomization energy of the configurations are:
* H: -13.62eV
* C: -1029.41eV
* N: -1484.87eV
* O: -2041.83eV
* F: -2712.82eV
