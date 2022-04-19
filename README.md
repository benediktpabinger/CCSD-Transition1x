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

To get the hdf5 file, run:

```
$ python get_t1x.py {path}
```

if no path is specified data will be downloaded in the current folder

#### Usage
In python run

```
from t1x import Dataloader

dataloader = Dataloader(path_to_h5_file)
for molecule in dataloader:
    energy = molecule["wB97x_6-31G(d).energy"]
    ...
```

The elements in the data loader each represent a single molecule. It is a dictionary that has the following keys available:
*    formula:                           chemical formula for the molecule.
*    positions:                         list of x, y, z coordinates of all atoms in the molecule in √Ö.
*    atomic_numbers:                    list of atomic numbers ordered in the same way as positions.
*    wB97x/6-31G(d).energy:             total energy of molecule
*    wB97x/6-31G(d).atomization_energy: atomization energy of molecule
*    wB97x/6-31G(d).forces:             list of x, y, z forces on each atom ordered in the same way as positions.


It is also possible to go through the reactant, transition state and product only by setting 'only_final' kwarg to True when instantiating the data loader.
In this case the data loader will return dictionaries where the configurations can be accessed under 'product', 'transition_state' or 'reactant'.

```
dataloader = t1x.Dataloader(path_to_h5_file, only_final=True)
for molecule in dataloader:
    ts_energy = molecule["transition_state"]["wB97x_6-31G(d).energy"]
    r_energy = molecule["reactant"]["wB97x_6-31G(d).energy"]
    activation_energy = ts_energy - r_energy
    ...
```



#### ase_db example
The ase\_db.py example we generate an ase.db database where each row has energy, forces and atomization\_energy in the data-field.
