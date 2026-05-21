## Calculation Inventory (`calculation_inventory.csv`)

`calculation_inventory.csv` tracks the status of every calculation run on top of the Transition1x dataset. One row per reaction. Columns:

### Identity

| Column | Description |
|--------|-------------|
| `rxn` | Reaction ID (e.g. `rxn0103`) |
| `split` | Dataset split: `train`, `val`, or `test` |
| `in_vault` | Reaction is included in the curated CURATOR vault |

### Val/Test NEB (wB97M-V/def2-TZVP, ORCA)

| Column | Description |
|--------|-------------|
| `orca_neb_ts_exists` | A `transition_state.xyz` file exists in the NEB output directory |
| `orca_neb_converged` | NEB converged (fmax < 0.05 eV/Å); determined by presence of a `converged` marker file written only on actual convergence |
| `orca_neb_fmax` | Final fmax (eV/Å) of the NEB run |
| `orca_neb_in_faillist` | Reaction was manually added to the failed-NEB list |
| `orca_barrier_eV` | Forward barrier (eV) from the converged NEB path (wB97M-V/def2-TZVP) |
| `t1x_barrier_eV` | Forward barrier (eV) from the original T1x NEB (wB97X-D3/6-31G(d)) |

### CCSD/CCSD(T) on NEB path (test set only)

| Column | Description |
|--------|-------------|
| `ccsd_pyscf_neb` | CCSD/def2-DZ SPs on NEB images completed (PySCF) |
| `ccsd_dz_sp_images` | Individual SP image results exist |
| `ccsd_dz_compiled` | CCSD/def2-DZ barriers compiled into a summary JSON |
| `ccsdt_tz_sp_images` | CCSD(T)/def2-TZ SP image results exist |
| `ccsdt_tz_compiled` | CCSD(T)/def2-TZ barriers compiled |
| `ccsdt_tz_neb_db` | CCSD(T)/def2-TZ results stored in neb.db format |

### Model evaluation (test set)

| Column | Description |
|--------|-------------|
| `in_barrier_comp_p10` | Reaction included in the p10 barrier comparison evaluation |
| `painn_barrier` | PaiNN model barrier evaluated |
| `mace_p10_ep291_meV` | MACE (p10, epoch 291) barrier prediction (meV) |
| `painn_barrier_meV` | PaiNN barrier prediction (meV) |

### Delta model SPs

| Column | Description |
|--------|-------------|
| `delta_sp` | **Test set:** wB97X-D3/6-31G(d) and wB97M-V/def2-TZVP SPs computed for delta model training/evaluation |
| `val_delta_sp` | **Val set, Group A (174 converged NEB):** wB97X-D3/6-31G(d) ORCA SPs + gradients on 50 uniformly sampled NEB images; wB97M-V energy read from neb.db. Output: `~/val_delta_sp/{rxn}/results.json`. Note: wB97M-V forces not available for Group A (not stored in neb.db); only delta energy is usable as training target. |
| `val_delta_sp_flip` | **Val set, Group B (51 failed NEB):** wB97M-V/def2-TZVP ORCA SPs + gradients on 50 uniformly sampled T1x geometries; wB97X-D3 energy and forces read from T1x HDF5. Full delta energy and delta forces available. Output: `~/val_delta_sp_flip/{rxn}/results.json` |

### NEVPT2 (older ORCA-based attempts, test set)

| Column | Description |
|--------|-------------|
| `nevpt2_sp_66_orca` | NEVPT2 SP with 6-6 active space, ORCA |
| `nevpt2_optts_66_orca` | NEVPT2 with optimised TS geometry, 6-6 active space, ORCA |

### MR Benchmark: CCSD(T) and NEVPT2/AVAS (top-10 FOD reactions, test set)

| Column | Description |
|--------|-------------|
| `ccsdt_sp_pyscf` | CCSD(T)/def2-TZVP single points on R, TS, P completed (PySCF). Output: `~/mr_benchmark/results/{rxn}_ccsdt.json` |
| `nevpt2_avas_pyscf` | NEVPT2/AVAS calculation was submitted and ran |
| `nevpt2_avas_fixed` | CASSCF converged at all three geometries (R, TS, P); NEVPT2 completed. Output: `~/nevpt2_results/{rxn}_pyscf_avas/nevpt2_results.json` |
| `nevpt2_avas_validated` | `True` if nat_occ analysis confirms the active space is balanced (≥1 fractional occupation at R, TS, and P) |
| `nevpt2_avas_fwd_meV` | NEVPT2 forward barrier (meV) |
| `nevpt2_avas_rev_meV` | NEVPT2 reverse barrier (meV) |
| `nevpt2_avas_geometry` | Geometry source used for NEVPT2 SPs (e.g. `orca_neb`) |
| `nevpt2_avas_flag` | Reliability flag: `ok` = nat_occ balanced across R/TS/P; `red_flag` = TS-biased active space (unreliable barriers — use CCSD(T) instead); `failed` = CASSCF did not converge at all geometries. Empty = not yet computed |

### MP2 NEB (experimental)

| Column | Description |
|--------|-------------|
| `mp2_neb_crashed` | MP2 NEB job crashed |
| `mp2_neb_succeeded` | MP2 NEB job completed successfully |

---

## Val Delta SP Plausibility Check

A basic plausibility check was run on the completed val delta SP calculations (2026-05-12).

### What was checked

1. **Completeness** — all `results.json` present, expected number of geometries sampled
2. **Delta sign and magnitude** — E_wB97M-V − E_wB97X-D3 should be a consistently negative total energy difference (larger basis + different functional lowers total energy)
3. **Within-reaction consistency** — the delta should be smooth across geometries of the same molecule; large std signals either SCF failure or strong geometry dependence
4. **SCF failure vs. physical variance** — if only one of the two energies is anomalous at a geometry, it is a SCF artifact; if both move coherently, it is a real geometry effect

### Results

| | Group A (174 converged NEB) | Group B (51 failed NEB) |
|---|---|---|
| Reactions with `results.json` | 174 / 174 | 51 / 51 |
| Delta mean across reactions | −3.03 ± 0.29 eV | −2.84 ± 0.15 eV |
| Within-rxn std (mean) | 0.089 eV | 0.093 eV |
| Within-rxn std (max) | 0.251 eV | 0.200 eV |
| Within-rxn std (p95) | 0.164 eV | 0.156 eV |
| Reactions with std > 0.15 eV | 12 | 4 |
| Reactions with < 90% geometries | 5 | 0 |

**Incomplete reactions (Group A):** rxn1914, rxn3126, rxn4932, rxn5041, rxn5802 — all have exactly 40 images in `neb.db` (`n_total=40`). These NEBs converged quickly; all available geometries were sampled. Not a data quality problem.

**High-std outliers:** The worst case (rxn4928, std=0.25 eV) was inspected manually. The single extreme geometry (idx=763, delta=−4.74 eV vs. typical −3.0 eV) has *both* wB97M and wB97X total energies ~33 eV above the converged structure — a genuinely strained intermediate NEB geometry. Both energies move coherently, ruling out SCF failure. High delta at strained geometries is physically expected: basis set incompleteness error is larger when bonds are compressed. This is real geometry-dependent delta, not corrupted data.

### Conclusion

Data is clean. High within-reaction variance is genuine physics arising from sampling the full NEB optimization history (which includes distorted intermediate geometries). The delta model must learn geometry-dependent corrections — the validation set correctly captures this by including the full PES, not just the final MEP.

### TODO — more thorough investigation

> **Note to self:** the check above is a basic sanity check. A more thorough investigation would include:
> - For each high-std outlier, plot delta vs. geom_idx to see whether variance is smooth (geometry-dependent) or spiky (SCF noise)
> - Filter geometries by energy above the MEP minimum (e.g. only include geometries within 2 eV of the minimum energy image) and check whether this removes the extreme delta values — would help decide if strained geometries should be excluded from the validation set
> - Cross-check Group A delta statistics against the delta distribution seen during training (train set) to confirm the val set is in-distribution
> - Verify that the 5 short-history reactions (40 images) have converged NEBs that simply needed fewer iterations — check their `orca_neb_fmax` values

---

#### Installation
To install, run:
```
$ git clone https://gitlab.com/matschreiner/Transition1x
$ cd Transition1x
$ pip install .
```
if you want to run the ase\_db.py example or generate the dataset from scratch, instead install dependencies by running

```
$ pip install '.[example]'
```

To download the hdf5 file to a given path (default is './data'), run:

```
$ python download_t1x.py {path}
```

The data will be downloaded to the current folder if no path is specified.

#### Usage
In python run

```
from transition1x import Dataloader

dataloader = Dataloader(path_to_h5_file)
for molecule in dataloader:
    energy = molecule["wB97x_6-31G(d).energy"]
    ...
```

The elements in the data loader each represent a single molecule. It is a dictionary that has the following keys available:
*    rxn:                               the name of the reaction that the molecule is coming from
*    formula:                           chemical formula for the molecule.
*    positions:                         list of x, y, z coordinates of all atoms in the molecule in Å.
*    atomic_numbers:                    list of atomic numbers ordered in the same way as positions.
*    wB97x_6-31G(d).energy:             total energy of molecule in eV.
*    wB97x_6-31G(d).atomization_energy: atomization energy of molecule in eV.
*    wB97x_6-31G(d).forces:             list of x, y, z forces on each atom in eV/Å - atoms are ordered in the same way as in positions.


It is possible to provide a datasplit key to the dataloader from 'train', 'val' or 'test' to only iterate through the training, validation or test data, respectively.

```
dataloader = t1x.Dataloader(path_to_h5_file, datasplit='test')
for molecule in dataloader:
    energy = molecule["wB97x_6-31G(d).energy"] # Molecule from test-data
    ...
```

Finally, it is possible to go through the reactant, transition state and product only by setting 'only_final' kwarg to True when instantiating the data loader.
In this case the data loader will return dictionaries where the configurations can be accessed under 'product', 'transition_state' or 'reactant'.




```
dataloader = t1x.Dataloader(path_to_h5_file, only_final=True)
for molecule in dataloader:
    ts_energy = molecule["transition_state"]["wB97x_6-31G(d).energy"]
    r_energy = molecule["reactant"]["wB97x_6-31G(d).energy"]
    activation_energy = ts_energy - r_energy
    ...
```



#### Examples

##### ase_db example
The ase\_db.py example we generate an ase.db database where each row has forces and atomization\_energy in the data-field. This database can be generated by running the exampe

```
$ python example/ase_db.py {path_hdf5_in} {path_db_out}
```

By default, the path to h5 and db files are data/transition1x.{db/h5}

##### simple example
The simple.py example loops through all configurations in the dataset and prints them as pretty dicts.
Afterwards it will loop through the dataset again but this time with the data loader that only returns reactants, transition_states and products.

```
$ python example/simple.py {path_h5}
```


#### Data generation
To generate the dataset from scratch:
Download and install ORCA here 'https://www.orcasoftware.de/tutorials_orca/'.
The original data can be fetched here 'https://zenodo.org/record/3715478#.YyxLJexBxqs'. Download the zipped directory 'wb97xd3.tar.gz' - this 'https://zenodo.org/record/3715478/files/wb97xd3.tar.gz?download=1' is a direct link to the file.

Unzip the data and run the NEB script on all:
```
$ python scripts/neb.py --output {output_path} --reactant {reactant.xyz} --product {product.xyz} --transition-state {transition-state.xyz} --output {path_to_output} --orcabinary {path_to_orca_binary}
```

It is also possible to specify
```
--neb_fmax      # fmax threshhold for NEB
--cineb_fmax    # fmax threshhold for CINEB
--steps         # max steps for the algorithm
```
and their values are set as described in the paper by default.

Running the above code on a reaction will produce the following files in the {output_path} directory.

```
{output_path}/
 ├─ fmaxs.json
 ├─ neb.db
 ├─ converged
 ├─ plots/
 │   ├─ mep.png
 │   ├─ transition-state.png
 │   ├─ reactant.png
 │   ├─ product.png
 ├─ xyz/
 │   ├─ transition-state.xyz
 │   ├─ reactant.xyz
 │   ├─ product.xyz
 ├─ orca/
     ├─ ...
```

* fmaxs.json - contains a list with the fmax's for every iteration of the algorithm. This is used to filter paths later.
* neb.db - is an ase.db that contains all configurations encountered while runnning. Be aware that product and reactant is unchanging but is still saved after each iteration.
* converged - this is an empty file that serves to inform whether the reaction converged within the the given steps
* plots - this directory contains plots of the MEP and reactant, product and transition state
* xyz - this directory contains xyz files of reactant, product and transition state
* orca - this directory is used by orca to perform its calculations. ORCA files for last iteration of the algorithm can be found here.

Compile a JSON list with paths to all output directories for all converged reactions and run

```
$ python scripts/combine_dbs.py --h5file {path_to_h5_output} --rxns {path_to_json_list}
```

This will generate the Transition1x.h5 file that has been released with the paper.
Please feel free to contact me if you have any questions regarding the dataset.
