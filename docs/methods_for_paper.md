# Methods — source document for the workshop paper

Every statement below carries the repo path it comes from. The validated tables
under `results/` are authoritative for all numbers. Where a repo artefact and
the thesis chapter disagree, the repo wins and the disagreement is noted. Where
a fact could not be established from the repo, it says **NOT FOUND**.

Verified 2026-08-24 against the working tree and the cluster home
`/home/energy/s242862` (referred to below as `~`).

---

## 1. How the reactions were selected

The 45 reactions were not drawn at random and not chosen by outcome. They are
**stratified by N_FOD**, the fractional-occupation-number weighted density,
which is the established descriptor for multireference character. The candidate
pool is the 279-reaction FOD ranking in `~/fod_ranking.json`; the ranking is
reproduced from that file by
[`pipeline/paper_reactions.py`](../pipeline/paper_reactions.py), whose report
asserts the strata sizes and rank boundaries.

| Stratum | n | Ranks | N_FOD | RKS-unstable |
|---|---|---|---|---|
| top 26 by N_FOD | 26 | 1–26 | 0.684 – 1.146 | 17 of 26 |
| spread across the ranking | 9 | 40–269 | 0.017 – 0.566 | 1 of 9 |
| bottom 10 | 10 | 270–279 | 0.003 – 0.014 | 0 of 10 |
| **total** | **45** | 1–279 | 0.003 – 1.146 | **18 unstable / 27 stable** |

The last column is `group_rxn` from
[`results/paper_reactions.csv`](../results/paper_reactions.csv), aggregated per
stratum from that file and not re-derived from anything else. A reaction counts
as unstable when at least one of its three model transition states has
`unstable_ts = 1` in
[`results/omol25_model_geoms.csv`](../results/omol25_model_geoms.csv); the
derivation is performed inside `pipeline/paper_reactions.py`.

**Why the spread stratum contributes 9 and not 10.** The selection code draws
the middle group at the one-based ranks
`[11, 40, 68, 97, 126, 154, 183, 212, 240, 269]`. Rank 11 is `rxn0896`, which
already lies inside the top-26, so the union is 45 rather than 46 and the
spread stratum contributes 9 distinct reactions. Sources: the selection
expression in
[`pipeline/which_sheet_did_models_learn.py`](../pipeline/which_sheet_did_models_learn.py),
the comment in
[`pipeline/job_neb_omol25_45.sh`](../pipeline/job_neb_omol25_45.sh)
(*"union = 45; rxn0896 is rank 11 and belongs to both top-26 and mid, hence 45
not 46"*), and the check `genau eine Ueberschneidung high/spread: ['rxn0896']
(Rang 11)` in the `pipeline/paper_reactions.py` report.

**Purpose of the stratification.** The design is meant to show behaviour across
the whole range of multireference character rather than only at its extremes.
The middle stratum is thinly populated — 9 reactions for 230 ranks — but it
covers ranks 40 to 269 and closes the gap in N_FOD between 0.017 and 0.566
that a top-and-bottom design would leave empty.

### How N_FOD was computed

Source: [`pipeline/screen_fod.py`](../pipeline/screen_fod.py), driver
[`pipeline/job_fod_screen.sh`](../pipeline/job_fod_screen.sh). PySCF, restricted
Kohn–Sham, Fermi smearing:

```python
K_TO_HA = 3.16681e-6   # Boltzmann constant in Ha/K
T_EL    = 5000.0       # K  (Grimme standard)

def compute_fod(xyz_path, basis='def2-SVP', xc='PBE'):
    mol = gto.M(atom=atom_str, basis=basis, charge=0, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.max_cycle = 300
    sigma = T_EL * K_TO_HA
    mf = smearing_(mf, sigma=sigma, method='fermi')
```

with

```
N_FOD = sum_i |n_i - n0_i| = 2 * sum_{virtual} n_i
```

(the identity by electron conservation is stated in the module docstring).
The screening runs on **one geometry per reaction**, the transition state of the
reference NEB, `~/orca_neb_results/<rxn>/transition_state.xyz`. The 279
candidates are the test-split reactions that have a converged reference NEB and
a transition state file; the generating command is recorded as a comment in
`pipeline/job_fod_screen.sh`.

**Where the pool came from, and why it is 279 and not 287.** The Transition1x
test split holds **287** reactions (counted directly from
`data/Transition1x.h5`, group `test`, 8 molecular formulas; every one of them
has a `transition_state` group). `fod_ranking.json` carries **279** of them in
`results` and **zero** in `errors` — so the eight missing reactions did not fail
the FOD calculation, they were never candidates.

The reason is historical and is not derivable from the repo: the pool is the
set for which a converged NEB from an **earlier and unrelated stage of the
project** already existed on disk. That stage pursued a different question; the
result reported in this work came out of it as an incidental finding, after
which the earlier NEB no longer mattered. The restriction was therefore never a
design decision of this study.

This is worth stating rather than hiding, because it cuts the right way: the
pool boundary was fixed **before** the effect was known, so neither it nor the
ranking can have been shaped by the outcome.

The eight unscreened reactions are `rxn5688`, `rxn6190`, `rxn7063`, `rxn7944`,
`rxn7956`, `rxn8967`, `rxn9058`, `rxn9592`. They are 2.8 % of the test split and
they are **not drawn uniformly**: six are `C5H5NO` (6 of 97 in that formula)
against two `C3H5NO2` (2 of 162), and none of the other six formulas loses a
reaction. If the reason is indeed reference-NEB convergence, that NEB converged
less readily for `C5H5NO`. **NOT VERIFIED** — confirming it requires listing
`~/orca_neb_results/` on the cluster, which was unreachable at the time of
writing.

The independent methods audit in
[`paper_methods_info.md`](../paper_methods_info.md) §6 quotes the same lines of
the same script — no disagreement.

**Level difference.** N_FOD is a screening quantity at PBE/def2-SVP with
smearing and is not on the level used anywhere else in this work; it enters the
paper only as the axis along which the 45 reactions were stratified, never as a
measured result.

---

## 2. What type of reactions they are

The 45 are unimolecular CHNO reactions with 10 to 14 atoms from the
**Transition1x test split** (`--split test` in every run script; the pool was
restricted to test-split reactions when `fod_rxn_list.txt` was generated). All
carry 40 to 52 electrons, all even, so the neutral closed-shell singlet used
throughout is the correct assignment. Composition, counted from
`results/paper_reactions.csv`:

| Formula | n |
|---|---|
| C₃H₅NO₂ | 22 |
| C₅H₅NO | 16 |
| C₃H₈N₂O | 4 |
| C₂H₅NO₂ | 2 |
| C₂H₃N₃O₂ | 1 |

Fragment counting on the reference endpoints
(`~/orca_neb_results/<rxn>/{reactant,product}.xyz`, bond if
`d_ij < f · (r_i^cov + r_j^cov)`) gives **one connected reactant in all 45**, and
on the product side 34 with one fragment, 10 with two, and 1 with three — i.e.
**34 rearrangements and 11 fragmentations**. The result is stable against the
threshold: 45 / 34 is unchanged for `f` from 1.15 to 1.40.

> **Draft status.** These class labels are **not** in a validated table.
> `results/paper_reactions.csv` has the columns `rxn, nfod, stratum, formula,
> group_rxn` and no reaction-type column; the 34/11 split comes from an ad-hoc
> analysis run on 2026-08-24 and has not been frozen into a script with checks.
> Treat it as draft until it is.

Further reference: [`results/README.md`](../results/README.md),
[`results/paper_reactions.csv`](../results/paper_reactions.csv).

---

## 3. The NEB protocol for UMA-S, UMA-M and eSEN

Run scripts: [`pipeline/uma_neb.py`](../pipeline/uma_neb.py),
[`pipeline/uma_m_neb.py`](../pipeline/uma_m_neb.py),
[`pipeline/esen_neb.py`](../pipeline/esen_neb.py). Job wrappers:
`pipeline/job_uma_neb.sh`, `pipeline/job_uma_m_neb.sh`,
`pipeline/job_esen_neb.sh` (plus `*_next.sh` continuations with identical
settings).

### Models and environment

| | checkpoint (from the job script) | calculator construction |
|---|---|---|
| UMA-S | `~/checkpoints/uma-s-1p2.pt` | `FAIRChemCalculator(predict_unit, task_name='omol')` |
| UMA-M | `~/checkpoints/uma-m-1p1.pt` | `FAIRChemCalculator(predict_unit, task_name='omol')` |
| eSEN | `~/checkpoints/esen_sm_conserving_all.pt` | `FAIRChemCalculator(predict_unit)` — no `task_name`, resolved to `omol` by fairchem, see below |

> **Repo-internal disagreement.** The docstring of `pipeline/uma_m_neb.py`
> names `uma-m-1p2.pt`, while `pipeline/job_uma_m_neb.sh` line 36 sets
> `CHECKPOINT=/home/energy/s242862/checkpoints/uma-m-1p1.pt`. The job script is
> what ran, and only `uma-m-1p1.pt` exists in `~/checkpoints/`. **UMA-M is
> 1p1**; the docstring is stale.

Environment, read from the module the job scripts load
(`module load Python/3.13.5-GCCcore-14.3.0`): Python 3.13.5,
**fairchem-core 2.20.0**, ASE 3.28.0, PyTorch 2.8.0, NumPy 2.4.3. Hardware:
one NVIDIA RTX 3090 per task (`--gres=gpu:1`, partition `sm3090el8`).

### The eSEN `task_name` question

`esen_neb.py` constructs its calculator without `task_name`; the comment in
[`pipeline/model_sp_recheck.py`](../pipeline/model_sp_recheck.py) line 48 states
*"UMA needs the task; eSEN is single-task and rejects the argument."*

A verification run exists: job **10737787**, outputs in
`~/model_sp_recheck/{UMA-S,UMA-M,eSEN}.json` and the corresponding
`slurm_10737787_*.out`. It re-instantiates each calculator exactly as the NEB
scripts do, evaluates it on the stored transition-state geometry, and compares
the fresh forces with the forces stored in the extxyz. Result for eSEN over
n = 19 structures:

```
n = 19   median max|dF| = 2.10e-06   groesste = 1.32e-05 eV/A
Strukturen mit relevanter Abweichung: 0
```

UMA-S and UMA-M agree to the same order (median 3.28·10⁻⁶ and 1.53·10⁻⁶).

**What this establishes.** The stored forces belong to the stored geometries
and the calculator construction is reproducible, including eSEN's construction
without `task_name`. It covered 19 of the 45 reactions, not all of them. It does
not by itself say which head eSEN was driven on — that is settled below, from
the logs and from the installed fairchem.

### Charge, spin, and which head the models were driven on

None of the three scripts writes `charge` or `spin` into `atoms.info`, and none
passes either to the calculator, so whatever fairchem defaults to is what ran.
The runs recorded it. **Every converged NEB log of all three models** carries
these two lines, twenty times per run — ten images × two warnings, once per
image at its first evaluation:

```
WARNING:root:task_name='omol' detected, but charge is not set in atoms.info. Defaulting to charge=0. Ensure charge is an integer representing the total charge on the system and is within the range -100 to 100.
WARNING:root:task_name='omol' detected, but spin multiplicity is not set in atoms.info. Defaulting to spin=1. Ensure spin is an integer representing the spin multiplicity from 0 to 100.
```

| model | line found | example log (converged run, rxn7949) | occurrences |
|---|---|---|---|
| UMA-S | yes | `~/logs/uma_neb_10438881_0.log` | 20 |
| UMA-M | yes | `~/logs/uma_m_neb_10499578_0.log` | 20 |
| eSEN | yes | `~/logs/esen_neb_10438668_0.log` | 20 |

Across all model-NEB logs the counts are 450 (UMA-S), 450 (UMA-M) and 510
(eSEN) occurrences of each of the two lines.

The code that sets the values is
`fairchem/core/models/uma/escn_md.py` lines 973–987, installed at
`~/.local/lib/python3.13/site-packages/`:

```python
# Set charge defaults
if "charge" not in atoms.info:
    if task_name == UMATask.OMOL.value:
        logging.warning(
            "task_name='omol' detected, but charge is not set in atoms.info. ...")
    atoms.info["charge"] = DEFAULT_CHARGE

# Set spin defaults (OMOL uses spin=1, others use spin=0)
if "spin" not in atoms.info:
    if task_name == UMATask.OMOL.value:
        atoms.info["spin"] = DEFAULT_SPIN_OMOL
        logging.warning(
            "task_name='omol' detected, but spin multiplicity is not set in ...")
    else:
        atoms.info["spin"] = DEFAULT_SPIN
```

with the constants in
`fairchem/core/units/mlip_unit/api/inference.py` lines 28–30:

```python
DEFAULT_CHARGE = 0
DEFAULT_SPIN_OMOL = 1
DEFAULT_SPIN = 0
```

**Charge 0, spin multiplicity 1** — neutral singlet, matching the
`* xyzfile 0 1` used on the DFT side and correct for all 45 systems (even
electron count, §2).

### The eSEN `task_name` question — closed

The eSEN logs carry `task_name='omol' detected` even though `esen_neb.py`
passes no `task_name`. The resolution is in
`fairchem/core/calculate/ase_calculator.py`, lines 68–80:

```python
valid_datasets = list(predict_unit.dataset_to_tasks.keys())
if task_name is not None:
    if task_name not in valid_datasets:
        raise ValueError(
            f"Invalid task_name '{task_name}'. Valid options are {valid_datasets}"
        )
    self._task_name = task_name
elif len(valid_datasets) == 1:
    self._task_name = valid_datasets[0]
else:
    raise RuntimeError(
        f"A task name must be provided. Valid options are {valid_datasets}"
    )
```

Omitting `task_name` therefore does not mean "no task": if the checkpoint
declares exactly one dataset, fairchem uses it silently; otherwise it raises.
The eSEN checkpoint declares exactly one. Read directly from
`~/checkpoints/esen_sm_conserving_all.pt`, field `tasks_config`:

```
name energy   datasets ['omol']   property energy
name forces   datasets ['omol']   property forces
```

**All three models were driven on the `omol` head** — UMA-S and UMA-M through
the explicit `task_name='omol'`, eSEN through single-dataset resolution to the
same value. This closes the OPEN marker of the earlier draft, and no
with/without test was needed: both the log line and the code path are
unambiguous for eSEN.

> **Repo-internal inaccuracy.** The comment in
> [`pipeline/model_sp_recheck.py`](../pipeline/model_sp_recheck.py) line 48 —
> *"UMA needs the task; eSEN is single-task and rejects the argument"* — does
> not describe fairchem 2.20.0. `task_name='omol'` **would** be accepted for
> this checkpoint, because `'omol'` is its only entry in `valid_datasets`; only
> a different name would raise. The construction the comment leads to is
> harmless and equivalent, the reason it gives is wrong.

### Band setup

Identical in all three scripts:

```python
neb = NEB(images, climb=False, parallel=False, method='improvedtangent')
relax_neb = NEBOptimizer(neb, logfile=.../neb.log)
relax_neb.run(fmax=args.neb_fmax,   steps=args.steps)   # 0.15, 500
neb.climb = True
converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)   # 0.05, 500
```

- **Images: 10.** Not interpolated — the band is initialised from the final
  wB97x NEB images stored in `Transition1x.h5`, selected as
  `[positions[0]] + positions[-8:] + [positions[9]]`, i.e. reactant, the last
  eight interior images and the product (`load_wB97x_images`, identical in all
  three scripts).
- **Endpoints** are re-relaxed with the model itself, `BFGS(...).run(fmax=0.05)`,
  before the band is optimised.
- **Tangent:** improved tangent, set explicitly in all three model scripts.
  **Spring constant: k = 0.1 eV/Å per spring**, the ASE default, inherited —
  see *Spring constant and tangent method* below.
- **Charge and spin** are never written to `atoms.info` and never passed to the
  calculator in any of the three scripts. fairchem then defaults to charge 0 and
  spin multiplicity 1, logged by every run and set in
  `fairchem/core/units/mlip_unit/api/inference.py` — see *Charge, spin, and
  which head the models were driven on* above. That matches the `* xyzfile 0 1`
  of the DFT side and is correct for all 45 systems (even electron count, §2).
- **Optimizer:** `NEBOptimizer`, default method `ode` (`ode12r`).

### Spring constant and tangent method

Neither `k` nor any `spring` argument appears in
[`pipeline/uma_neb.py`](../pipeline/uma_neb.py),
[`pipeline/uma_m_neb.py`](../pipeline/uma_m_neb.py),
[`pipeline/esen_neb.py`](../pipeline/esen_neb.py) or
[`pipeline/orca_neb.py`](../pipeline/orca_neb.py) — zero matches in the repo
copies and in the cluster copies `~/pipeline/*.py` that actually ran. The ASE
default therefore applies everywhere.

Installed ASE, read from the environment the jobs load: `ase.__version__ =
3.28.0`, source file `~/.local/lib/python3.13/site-packages/ase/mep/neb.py`.

```
NEB.__init__      (self, images, k=0.1, climb=False, parallel=False,
                   remove_rotation_and_translation=False, world=None,
                   method=None, allow_shared_calculator=False, precon=None,
                   **kwargs)
BaseNEB.__init__  (self, images, k=0.1, climb=False, ...)
```

The default is declared in `BaseNEB.__init__` (lines 289–292) and passed through
unchanged by `DyNEB` (677–681) and `NEB` (802–806). Docstring, lines 833–834,
verbatim:

```
        k: float or list of floats
            Spring constant(s) in eV/Ang.  One number or one for each spring.
```

Expansion into the per-spring list, lines 352–354:

```python
        if isinstance(k, (float, int)):
            k = [k] * (self.nimages - 1)
        self.k = list(k)
```

**k = 0.1 eV/Å for every spring**, nine springs for ten images. The unit is
quoted as ASE writes it.

**Tangent method.** The three model scripts pass `method='improvedtangent'`
explicitly. The reference NEB (`pipeline/orca_neb.py` line 148) and the
OMol25-level NEB (`pipeline/orca_neb_omol25.py` line 212) pass no `method`. In
ASE 3.28.0 that resolves to the same value, with a warning — `neb.py` lines
328–340:

```python
        if method is None:
            warnings.warn(
              "The default method has changed from 'aseneb' to "
              "'improvedtangent'. ...", UserWarning)
            method = 'improvedtangent'
```

> **Open point.** The warning text states that the default *changed* from
> `'aseneb'`. Whether the original reference NEBs ran under an ASE old enough to
> still default to `'aseneb'` cannot be decided from the logs: the rerun logs
> (`~/logs/orca_neb_rerun_*.log`) do carry the warning, so those ran under a
> new-enough ASE, but the first-generation reference runs could not be
> attributed. This affects the tangent method of the reference band only, never
> `k` and never the model NEBs.

**How k enters the force** depends on the tangent method; recorded as the code
states it, without interpretation:

```python
# ImprovedTangentMethod.add_image_force, lines 133-137
imgforce -= tangential_force * tangent
imgforce += (spring2.nt * spring2.k - spring1.nt * spring1.k) * tangent

# ASENEBMethod.add_image_force, lines 157-166
imgforce -= np.vdot(spring1.t * spring1.k - spring2.t * spring2.k,
                    tangent) * factor
```

`NEBOptimizer` and `ode12r` contain no spring handling of their own — no
reference to `.k` anywhere in the optimizer.

> **Do not read the method warning as evidence about the band.** Every
> productive model-NEB log carries *"The default method has changed…"* although
> all three scripts set `method='improvedtangent'`. The source is not the
> optimised band but `NEBTools.get_fmax()`, `neb.py` lines 1179–1183:
>
> ```python
>     def get_fmax(self, **kwargs):
>         """Returns fmax, as used by optimizers with NEB."""
>         neb = NEB(self.images, **kwargs)
> ```
>
> The scripts attach this to every optimizer step
> (`relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))`) purely to
> write `fmaxs.json`. That throwaway NEB receives neither `method` nor `k` and
> inherits both defaults — the same `improvedtangent` and the same k = 0.1. The
> optimised band is unaffected.

### The stopping criterion is not any column of the audit table

`fmax = 0.05` acts on `NEBOptimizer.get_residual()`, which returns
`self.neb.get_residual()` — the **projected band force** in ASE convention, that
is the largest per-atom force norm over the band after projection. Three
distinct quantities must be kept apart:

| quantity | where | what it is |
|---|---|---|
| `f_band_final` | `results/neb_runs.csv` | projected band force, ASE norm convention, the criterion |
| `f_model_norm_max` | `results/omol25_model_geoms.csv` | raw MLIP force at the TS image, largest per-atom norm |
| `f_model_max` | `results/omol25_model_geoms.csv` | raw MLIP force at the TS image, largest Cartesian component |

The last two differ by a measured factor of 0.6055 to 0.9985 over the 135
structures (checked by `selfcheck()` in
[`pipeline/plot_omol25_figs.py`](../pipeline/plot_omol25_figs.py)). The first is
a band-wide, projected quantity and is not comparable to either: **41 of the
112 runs that met the band criterion have a raw TS-image force norm above
0.05 eV Å⁻¹.** That is ordinary NEB behaviour, not a defect, but it means the
criterion must never be drawn as a boundary on a raw-force axis.

### Run accounting

From [`results/neb_runs.csv`](../results/neb_runs.csv), built and checked by
[`pipeline/neb_runs.py`](../pipeline/neb_runs.py). Not recounted from logs.

- 135 searches, 45 reactions × 3 models.
- **The optimizer returned success for 133 of 135.**
- **112 met the band criterion** (`criterion_met = 1`, `f_band_final ≤ 0.05`
  with a 5·10⁻⁵ tolerance for the four-decimal log rounding).
- **21 carry the success marker although the criterion was not met**, from
  0.0699 to 0.3193 eV Å⁻¹, all with 441 to 638 logged steps. Marker mechanics:
  `ode12r` exhausts `for nit in range(1, steps + 1)` and falls off the end
  **without raising**, so `NEBOptimizer.run_ode` returns `True` and
  `pipeline/uma_neb.py` lines 159–163 write the `converged` file.
  `NEBOptimizer.run()` sets `self.max_steps = steps`, so the CI phase gets a
  fresh budget of 500 *attempts* while only *accepted* steps are logged. Full
  documentation in [`results/README.md`](../results/README.md); the 21 rows are
  frozen by name in `pipeline/neb_runs.py` and a check fails if the set changes.
- **2 raised an explicit failure:** `rxn0894/uma-s` (0.1191) and
  `rxn8837/esen` (0.1653).

Status against RKS stability of the resulting transition state:

| status | n | stable | unstable |
|---|---|---|---|
| marker **and** criterion | 112 | 70 | 42 |
| marker, criterion not met | 21 | 11 | 10 |
| no marker | 2 | 1 | 1 |
| all | 135 | 82 | 53 |

**Robustness.** Whether the reported separation depends on the 23 runs that did
not meet the criterion:

| subset | quantity | stable | unstable | ratio |
|---|---|---|---|---|
| all 135 | `f_model_max` | 0.0299 | 0.0344 | 1.15 |
| all 135 | `f_dft_max` | 0.0507 | 0.1293 | **2.55** |
| 112 with `criterion_met` | `f_model_max` | 0.0327 | 0.0338 | 1.03 |
| 112 with `criterion_met` | `f_dft_max` | 0.0507 | 0.1245 | **2.45** |

The DFT residual-force separation moves from 2.55 to 2.45 and the stable median
does not move at all. The result does not depend on the excluded runs.

---

## 4. The audit: how `results/omol25_model_geoms.csv` was made

Authoritative description:
[`results/omol25_model_geoms.md`](../results/omol25_model_geoms.md). Build
script: [`pipeline/omol25_model_geoms.py`](../pipeline/omol25_model_geoms.py).
135 rows, 26 columns, one per (reaction, model).

### Geometries

Always the three structures the MLIP itself produced,
`<modeldir>/<rxn>/{reactant,transition_state,product}.xyz`, **unrelaxed**. No
DFT optimisation, no reference geometry. Because both sides of every difference
are read at the same point, geometry cancels out of every error.

### The single points

One UKS single point per structure at the **OMol25 level of theory** — this
phrasing is deliberate, for two reasons. OMol25 used ORCA 6.0.0 while these runs
used 5.0.4 (recorded in the provenance written by
[`pipeline/orca_neb_omol25.py`](../pipeline/orca_neb_omol25.py):
`'orca_version': '5.0.4 (OMol25 used 6.0.0)'`), and no 6.0.0 calculation exists
anywhere in this work. And the route to the broken-symmetry solution differs:
`STABPerform` here, a 20° β-space rotation in OMol25. The second difference was
measured — see *Cross-check against the OMol25 symmetry-breaking protocol*
below — the first was not. Do not write "identical to OMol25".

Verbatim from
[`pipeline/job_orca_omol25_probe.sh`](../pipeline/job_orca_omol25_probe.sh):

```
! UKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
%pal nprocs 12 end
%maxcore 3500
%scf
  Thresh 1e-12
  TCut   1e-13
  MaxIter 300
  STABPerform true
  STABRestartUHFifUnstable true
end
* xyzfile 0 1 <tag>.xyz
```

`Thresh` and `TCut` are **integral neglect thresholds**, not convergence
criteria; 1e-12 is 25× tighter than the ORCA default of 2.5e-11. Both values
read from the ORCA headers themselves: `~/orca_om25/rxn0101_eSEN/ts_sp.out`
prints `Integral threshold ... 1.000000e-12`, while the reference-level run
`~/orca_freq/rxn0101_eSEN/bs_sp.out` prints `2.500000e-11`.

**The gradient** is a separate `EnGrad` run that reads the converged orbitals of
the preceding single point via `MORead`, so it is evaluated on whichever surface
`STABPerform` selected:

```
! UKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3 EnGrad MORead
%moinp "ts_start.gbw"
%pal nprocs 12 end
%maxcore 3500
%scf
  Thresh 1e-12
  TCut   1e-13
  MaxIter 300
end
* xyzfile 0 1 ts_sp.xyz
```

**The RKS single point at the transition state**
([`pipeline/job_rks_sheet.sh`](../pipeline/job_rks_sheet.sh), Slurm jobs
10767516 and 10767531) is run **deliberately without a stability analysis** —
the restricted solution is wanted there even where it is not the ground state,
because its distance to the broken solution *is* the breaking depth:

```
! RKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
%pal nprocs 12 end
%maxcore 3500
%scf
  Thresh 1e-12
  TCut   1e-13
  MaxIter 300
end
* xyzfile 0 1 ts_rks.xyz
```

It was run for the 53 rows with `unstable_ts = 1` plus one stable control. For
the remaining 81 rows `STABPerform` had already confirmed that no second
solution exists, so the depth is 0 by construction; `depth_src` records which
of the two applies per row (`rks_sp` in 54 rows, `stabperform_stable` in 81).

### Cross-check against the OMol25 symmetry-breaking protocol

OMol25 breaks spin symmetry by a 20° HOMO–LUMO rotation in the β space; the
audit runs use `STABPerform` with `STABRestartUHFifUnstable` instead. The two
routes were compared directly.

**Earlier comparison, reference geometries.** `pipeline/job_omol25_settings.sh`,
26 reactions (the top-26 N_FOD stratum), at
`~/orca_neb_results/<rxn>/transition_state.xyz`. Three calculations per
reaction: A = RKS, B = UKS + 20° rotation, C = UKS + stability restart, all at
def2-TZVPD with the OMol25 thresholds. B and C agree on whether a broken
solution exists in 26 of 26 (18 broken, 8 closed-shell) and, where broken, in
energy to 1.2·10⁻⁷ Ha. The same 18/8 split is produced by the PySCF stability
analysis at def2-TZVP. Recorded in
[`multireference_screening.md`](../multireference_screening.md), section
*Cross-validation*.

**This work, audit geometries.** The 26-reaction comparison ran at the reference
transition states; the audit table stands at the model geometries. The
comparison was repeated there, on every row.

*What was run.* 135 UKS single points, one per row of
`results/omol25_model_geoms.csv`, on the same
`<modeldir>/<rxn>/transition_state.xyz` the `ts_sp` runs used. Level identical
to `ts_sp` — wB97M-V/def2-TZVPD, def2/J, RIJCOSX, TightSCF, DEFGRID3,
Thresh 1e-12, TCut 1e-13, MaxIter 300, ORCA 5.0.4, `%pal nprocs 12`,
`%maxcore 3500`. The only difference is the `%scf` block: instead of
`STABPerform true` / `STABRestartUHFifUnstable true` it contains

```
  Rotate {HOMO, LUMO, 20, 1, 1} end
```

taken verbatim from `pipeline/job_omol25_settings.sh`. No stability analysis in
these runs. Orbital indices from `HOMO = n_elec/2 − 1`, `LUMO = n_elec/2` with
`n_elec` from `~/fod_ranking.json`; the formula was checked against the 26
hand-maintained index pairs in `job_omol25_settings.sh` (26/26 identical) and
every run additionally verified its indices against the electron count printed
by ORCA, aborting on a mismatch.

*Where.* Slurm job **10771382**, 135 array tasks, outputs in
`~/orca_rot_check/<rxn>_<Modell>/ts_rot.out`. Job script
[`pipeline/job_rot_check.sh`](../pipeline/job_rot_check.sh), task-file generator
[`pipeline/mk_rot_tasks.py`](../pipeline/mk_rot_tasks.py), collector
[`pipeline/rot_check.py`](../pipeline/rot_check.py), table
[`results/rotation_check.csv`](../results/rotation_check.csv). Nothing under
`orca_om25/` or `orca_rks_sheet/` was written or modified.

*What was compared.* Per row: ⟨S²⟩ at the transition state, total energy in
Hartree, SCF cycle count. `verdict_match` is 1 when both ⟨S²⟩ lie on the same
side of 0.05.

*Numbers.* All 135 runs terminated normally. **134 of 135 agree in verdict.**

| | n | result |
|---|---|---|
| stable rows | 82 | rotation returns to ⟨S²⟩ ≈ 0; largest value 0.000126 |
| unstable rows | 53 | 52 same verdict |
| all rows except one | 134 | \|ΔE\| ≤ 4.10·10⁻⁷ Ha = 0.011 meV |
| the exception | 1 | rxn4113/UMA-S, \|ΔE\| = 2.62·10⁻⁴ Ha = 7.14 meV |

The energy differences are bounded, not merely centred: 129 of 135 rows lie
below 10⁻⁷ Ha, 134 below 10⁻⁶ Ha, and the largest value among those 134 is
4.10·10⁻⁷ Ha (rxn10054/UMA-M). The single exception sits almost three orders of
magnitude above it, so a bound is quoted rather than a median.

SCF cycles: rotation median 25, max 303; STABPerform median 18, max 184.

*The one deviation.* **rxn4113 / UMA-S.**

| | ⟨S²⟩ | E [Ha] | SCF cycles |
|---|---|---|---|
| 20° rotation | 0.000000 | −322.370328867312 | 80 |
| STABPerform | 0.128374 | −322.370591167142 | 28 |

ΔE = +2.623·10⁻⁴ Ha = 7.14 meV.

**The deviation is conservative for this work.** The audit table stands on the
lower of the two solutions — it is the rotation route that misses one, not the
stability analysis. The reverse case does not occur: no row has a rotation
solution below the `STABPerform` one beyond the collector's 10⁻⁶ Ha tolerance,
and the most negative ΔE over all 135 rows is −8.85·10⁻⁸ Ha (rxn7957/UMA-M,
0.0024 meV), which is convergence noise. Both statements are asserted by
`pipeline/rot_check.py`; the single mismatching row is frozen there by name, so
the check fires if the set changes.

The breaking depth of that row in
`omol25_model_geoms.csv` is 7.1 meV. The other two models of the same reaction
agree (UMA-M ⟨S²⟩ 1.007267 / 1.007264, ΔE −1.5·10⁻⁸ Ha; eSEN 0.167953 /
0.167451, ΔE −8.0·10⁻¹² Ha). Six unstable rows with a smaller breaking depth —
down to 0.6 meV (rxn10054/eSEN) — agree.

The three hardest rows agree: rxn0894/UMA-M (⟨S²⟩ 1.038358 both, ΔE
−1.2·10⁻⁹ Ha), rxn8885/UMA-S (1.024370 / 1.024371, +1.7·10⁻¹¹ Ha),
rxn8837/UMA-S (1.009993 / 1.009994, +1.7·10⁻⁹ Ha).

*Scope of the test.* Both sides ran under ORCA 5.0.4. No calculation with ORCA
6.0.0 exists in this work, so the version difference to OMol25 is untouched by
this comparison. The test covers the symmetry-breaking route only.

### Definitions, kept apart on purpose

- **Residual force** — `max_i` \|F_i\| over all 3N Cartesian components, computed
  separately for each force field: `f_model_max` for the MLIP, `f_dft_max` for
  DFT. This is a property of one field, not a comparison.
- **Force error** — the difference of the two fields, component by component,
  reduced two ways: `f_err_max = max_i |F_i^MLIP − F_i^DFT|` and
  `f_err_mae = ⟨|F_i^MLIP − F_i^DFT|⟩_i`. Unlike a difference of two maxima this
  is positive by construction and is a genuine error.
- **Barriers** are zeroed at the model's **own reactant** of the same run,
  `barr_model = E_MLIP(TS) − E_MLIP(R)` and `barr_dft = E_DFT(TS) − E_DFT(R)`.
  That reactant is closed-shell in all 135 rows (`s2_r = 0`), so RKS and BS
  coincide there and the zero point favours neither surface.
- **Breaking depth** `depth_ts_mev = E_RKS(TS) − E_BS(TS)` at the model
  geometry, in meV.

Raw energies are stored in Hartree, derived quantities in eV (meV for the
depth).

### Self-checks enforced by the build script

`pipeline/omol25_model_geoms.py` aborts if any of these fails, and all pass:
135 rows · 53 unstable · the 0.05 threshold lies in an empty zone (smallest
non-zero ⟨S²⟩ is 0.057936) · depth present for every row · depth of the stable
rows below 1 meV (largest 0.0008) · depth of the unstable rows all > 0 · 54
depths measured, 81 set to 0 by `STABPerform` · **null control** `rxn7060/esen`
is stable yet has a measured RKS point, giving 0.0008 meV where 0 must
stand · the shared columns agree with the older `omol25_compare.csv` to
5·10⁻⁵.

---

## 5. How the groups were split

Two levels, both documented, both derived rather than assigned.

**Structure level — primary, used by every per-structure figure.** ⟨S²⟩ of the
audit single point at the model geometry, column `s2_ts`, thresholded at 0.05
into `unstable_ts`. The value is **exactly 0 in 82 of 135 rows and at least
0.0579 in the remaining 53, with nothing in between**, so the classes separate
without a chosen cut-off — the 0.05 sits in an empty zone, and the build script
checks that it still does. The classification comes from the same single point
that produced the forces and energies, at the same geometry; it is not borrowed
from the reference structure.

**Reaction level — derived, used only for per-reaction quantities** such as the
spread of the DFT barrier across the three model geometries. `group_rxn` in
`results/paper_reactions.csv`: unstable if at least one of the three model
transition states is unstable. 27 stable, 18 unstable.

The two levels are not interchangeable, and one reaction actually splits across
models: **`rxn8837`** is unstable for UMA-S (⟨S²⟩ 1.010, depth 3460 meV) and
UMA-M (1.007, 3212 meV) but stable for eSEN (⟨S²⟩ 0.000, depth 0) — three
different structures for the same reaction, on two different surfaces. It is the
only such case in the 45. Degree varies more widely than class: at `rxn8885` all
three are unstable, but UMA-M sits at ⟨S²⟩ 0.114 with a depth of 2.2 meV while
eSEN and UMA-S are fully broken at 1.028 / 2953 meV and 1.024 / 1971 meV. Both
observations are why the structure level is primary.

> Correction against an earlier draft: `rxn8885` was cited as a case where the
> class splits across models. It is not — all three rows have
> `unstable_ts = 1`. It is a case where the *depth* splits. The class splits at
> `rxn8837`.

---

## 6. Basis and level bookkeeping

| what | level | source |
|---|---|---|
| audit single points and gradients, all of `omol25_model_geoms.csv` | ωB97M-V/def2-TZVPD, def2/J, RIJCOSX, TightSCF, DEFGRID3, Thresh 1e-12, TCut 1e-13, ORCA 5.0.4 | `pipeline/job_orca_omol25_probe.sh`, `pipeline/job_rks_sheet.sh` |
| reference NEB (`~/orca_neb_results/`) | `wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad`, ORCA default grid, `%pal nprocs 1`, `%scf maxiter 200` | [`pipeline/orca_neb.py`](../pipeline/orca_neb.py) lines 49–52 |
| FOD screening (N_FOD) | PBE/def2-SVP, PySCF, Fermi smearing at T_el = 5000 K | `pipeline/screen_fod.py` |
| OMol25-level NEB (`~/orca_neb_omol25/`) | `wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3 EnGrad`, endpoints re-relaxed at the same level | [`pipeline/orca_neb_omol25.py`](../pipeline/orca_neb_omol25.py) |
| hinge measurements, both tables | ωB97M-V/def2-TZVPD, def2/J, RIJCOSX, TightSCF, DEFGRID3, Thresh 1e-12, TCut 1e-13, ORCA 5.0.4 — settings block byte-identical to the audit | [`pipeline/job_hinge_t1x.sh`](../pipeline/job_hinge_t1x.sh), [`pipeline/job_hinge_omol25.sh`](../pipeline/job_hinge_omol25.sh) |

Everything that enters `results/omol25_model_geoms.csv` sits at a **single
consistent level**, def2-TZVPD with the OMol25 thresholds. The reference NEB and
the FOD screening are at their own, cheaper levels and are declared as such
wherever they are used; neither contributes a number to the audit table.

**The hinge tables are entirely def2-TZVPD; the old def2-TZVP hinge numbers
are superseded.** An earlier version of this measurement stood at
wB97M-V/def2-TZVP with PySCF gradients evaluated at the def2-TZVP reference NEB
geometries, and produced the figure that is quoted as a median broken-symmetry
gradient of **1.697 eV Å⁻¹** over 19 multireference reactions. Neither its level
nor its geometries match anything else in this document. Its successor is the
median `f_bs` of the 15 locally unstable rows of
[`results/hinge_omol25.csv`](../results/hinge_omol25.csv), **1.8695 eV Å⁻¹**
(§7). No number in §7, and no number in either hinge table, comes from the old
analysis.

The legacy artefacts that still carry it are
[`results/hinge_rows.csv`](../results/hinge_rows.csv) (columns `F_rks`, `F_bs`),
the hard-coded pairs in [`pipeline/paper_rows.py`](../pipeline/paper_rows.py)
line 38 together with its assertion `abs(med_bs - 1.697) < 5e-4` on line 183,
and the `Median 1.697` annotation drawn by `fig4()` in
[`pipeline/plot_paper_figs.py`](../pipeline/plot_paper_figs.py) line 678. They
are retained as history. **They must not be cited, and no new analysis may read
them.**

**One file mixes levels and must not be used for new analyses.**
`results/omol25_compare.csv` carries def2-TZVP columns (`F_tzvp`, `barr_tzvp`,
`rxne_tzvp`) next to def2-TZVPD columns. It exists to document the shift between
the two levels and is retained only as a cross-check: `omol25_model_geoms.py`
compares its own shared columns against it and asserts agreement to 5·10⁻⁵. All
thirteen figures in `pipeline/plot_omol25_figs.py` read
`omol25_model_geoms.csv` and nothing else, except `fig9_4`, which additionally
reads `results/model_ts_rmsd.csv` for its geometric axis.

---

## 7. The hinge measurements (label-geometry consistency)

Two tables, one question, two places to ask it.
[`results/hinge_t1x.csv`](../results/hinge_t1x.csv) (45 rows) measures what the
training labels carry at the inherited Transition1x geometries;
[`results/hinge_omol25.csv`](../results/hinge_omol25.csv) (33 rows) isolates the
surface effect at the same saddles re-optimised at the training level. Both
evaluate two residual forces at **identical nuclear coordinates** — one on the
restricted surface, one on the surface that is actually the ground state at
that point — so nothing moves between the two numbers and only the electronic
solution differs.

Both tables are built and checked by one script,
[`pipeline/hinge_tables.py`](../pipeline/hinge_tables.py), which aborts rather
than writing a file when a check fails. It supersedes the earlier
`pipeline/hinge_t1x.py` and `pipeline/hinge_omol25.py`. Column-by-column
provenance is in [`results/README.md`](../results/README.md).

### Geometry sources

**Table 1 — the Transition1x label transition states, 45 of 45.**
`~/t1x_ts/<rxn>.xyz`, extracted by
[`pipeline/extract_t1x_ts.py`](../pipeline/extract_t1x_ts.py) from
`~/data/Transition1x.h5`, group `test/<formula>/<rxn>/transition_state`, level
ωB97x/6-31G(d). No NEB of our own, no re-optimisation — this is the structure
the models were trained on, taken as it stands.

The group matters: the reaction-level field `positions` is **not** the converged
band but the entire optimisation history (138 to 4274 images in the test split),
whose energy maximum is an early unrelaxed image lying electronvolts off. The
transition state has its own group, and that is what is read.

**Table 2 — the same saddles re-optimised at ωB97M-V/def2-TZVPD, 33 of 45.**
`~/orca_neb_omol25/<rxn>/transition_state.xyz`, written only on a successful
CI-NEB run. The protocol is
[`pipeline/orca_neb_omol25.py`](../pipeline/orca_neb_omol25.py):

| step | setting | source |
|---|---|---|
| starting band | 10 images — reactant, the last eight interior images of the Transition1x band, and `positions[9]` as product; identical selection to `pipeline/orca_neb.py` | `orca_neb_omol25.py` line 104 |
| endpoints | re-relaxed at the same level, `BFGS` to fmax 0.05 | lines 205, 208 |
| band | `NEB(images, climb=False)` to fmax **0.15** | lines 212, 220 |
| climbing image | `neb.climb = True`, then to fmax **0.05** | lines 223–224 |
| optimiser | `NEBOptimizer` (ASE ODE solver), `steps=500` per phase | lines 214, 257 |
| transition state | highest-energy image of the final band | line 235 |

`fmax` here is the **ASE projected band force** — the largest per-atom force
norm over the band — and not the largest raw Cartesian component. The two are
different numbers; §3 keeps the three force conventions of this work apart.

Slurm jobs behind the 33 geometries:

| run | job | tasks | outcome |
|---|---|---|---|
| single-reaction validation, rxn1320 | **10686096** | 1 | converged, 4 h 10 min; TS RMSD 0.0134 Å against the def2-TZVP reference, barrier shift +8.2 meV |
| main array | **10686662** | 45 | **31** converged (markers dated 29–30 July 2026) |
| resume of five non-converged bands | **10767681** | 5 | **2** converged — rxn10054 in 1 h 18 min, rxn5690 in 6 h 25 min; the other three cancelled after 23 h 48 min to 24 h 52 min without reaching fmax |

`31 + 2 = 33`. The resume
([`pipeline/job_neb_omol25_resume.sh`](../pipeline/job_neb_omol25_resume.sh))
restarts from the last band in `neb.db` and backs the previous state up to
`neb_before_resume.db` first; the three cancelled reactions (rxn7949, rxn3107,
rxn0894) therefore had **two** attempts and are still among the 12 excluded.

### The measurement chain at each point

Three ORCA runs per reaction. The settings block is **byte-identical** to the
audit block of `pipeline/job_orca_omol25_probe.sh` — same `METHOD`, `HEAD` and
`SCFCOMMON` strings, verified by direct comparison:

```
METHOD='wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3'
HEAD='%pal nprocs 12 end
%maxcore 3500'
SCFCOMMON='  Thresh 1e-12
  TCut   1e-13
  MaxIter 300'
```

| run | header | yields |
|---|---|---|
| `rks_sp` | `! RKS $METHOD EnGrad` — **no** stability analysis | E_RKS, F_RKS |
| `uks_sp` | `! UKS $METHOD` + `STABPerform true` + `STABRestartUHFifUnstable true` | ⟨S²⟩, selects the surface |
| `uks_engrad` | `! UKS $METHOD EnGrad MORead` with `%moinp "uks_start.gbw"` | E_BS, F_BS |

Charge and multiplicity are `* xyzfile 0 1` throughout; the electron count from
the H5 task file is compared against ORCA's own `Number of Electrons` and a
mismatch aborts the task. The restricted run deliberately omits the stability
analysis: the restricted solution is wanted there even where it is not the
ground state, because that is the comparison.

| table | job | script | outputs |
|---|---|---|---|
| `hinge_t1x.csv` | **10773547** | [`pipeline/job_hinge_t1x.sh`](../pipeline/job_hinge_t1x.sh) | `~/orca_hinge_t1x/<rxn>/` |
| `hinge_omol25.csv` | **10773167** | [`pipeline/job_hinge_omol25.sh`](../pipeline/job_hinge_omol25.sh) | `~/orca_hinge25/<rxn>/` |

**E_BS is read from `uks_engrad`, not from `uks_sp`.** ORCA returns a different
total energy for the same solution depending on whether the job is a plain
single point or an `EnGrad` — about 2.4·10⁻⁵ Ha at this setup, plausibly the
final COSX/VV10 grid treatment. Measured on the stable rows, where the two
surfaces coincide and the energies must be equal: EnGrad against EnGrad agrees
to sub-nanohartree, EnGrad against single point leaves a systematic 0.66 meV in
every breaking depth. The first version of the collector used `uks_sp` and every
depth was too small by that amount; the null probe below exposed it.

### Column definitions

| column | definition |
|---|---|
| `s2_ts` | ⟨S²⟩ of `uks_sp` at the tabulated point |
| `f_rks` | `max_i` \|F_i\| over all 3N Cartesian components of the `rks_sp` gradient, eV Å⁻¹ |
| `f_bs` | the same quantity from `uks_engrad` |
| `ratio` | `f_bs / f_rks` |
| `depth_mev` | E_RKS − E_BS in meV, both from `EnGrad` runs |
| `f_ref` | **Table 1 only.** `max_i` \|F_i\| of the label point on its **own** level, read directly from `~/data/Transition1x.h5`, field `transition_state/wB97x_6-31G(d).forces`. The dataset stores forces in eV Å⁻¹ already. |
| `group` | **imported** from [`results/paper_reactions.csv`](../results/paper_reactions.csv), column `group_rxn` |
| `group_local` | derived here, from `s2_ts` |

`f_rks` and `f_bs` use the same convention as `f_dft_max` in the audit table —
largest Cartesian component — and **not** the ASE per-atom-norm convention that
the NEB criterion uses. §3 keeps the three conventions apart.

**The two class columns are not the same thing, and only one is authoritative
here.** `group` is a *reaction* label derived from the **model geometries**:
unstable when at least one of the three model transition states has
`unstable_ts = 1` (§5). `group_local` is the class of **the point actually
tabulated**, from ⟨S²⟩ of its own `uks_sp` under the same 0.05 rule that forms
`unstable_ts`. Both are valid and answer different questions; they need not
agree, because the class depends on the geometry. **Every check and every
reported number in this section runs against `group_local`**; `group` is carried
alongside for traceability.

### Reading the T1x table

At their own level of theory the label points **are** saddle points: `f_ref` has
a median of **0.0145 eV Å⁻¹** across the 45 rows, with 44 of them below 0.05 and
a single exception, rxn5690 at 0.3660, whose label transition state is not
converged in the dataset itself. Recomputed at the training level the same
points are displaced **uniformly** — `f_rks` medians 0.6088 (stable, n = 27) and
0.5885 (unstable, n = 18), a factor of about 40 over `f_ref` — and because both
groups move alike, this displacement is the level change
ωB97x/6-31G(d) → ωB97M-V/def2-TZVPD and nothing else. Only in the unstable group
does the surface effect come on top of it (`f_bs` 1.6359, ratio **2.80**), and
once the level offset is removed by re-optimising the saddles at the training
level — the lower block of the frozen core table below — that surface effect is
all that remains, at a factor of **32**.

The contrast between the two blocks is therefore not that the broken-symmetry
force grows: it barely moves, 1.6359 to 1.8695. It is that the denominator falls
by a factor of fourteen, 0.5885 to 0.0420, once one stops measuring the surface
effect through a level mismatch.

### Checks and bounds enforced by the build script

`pipeline/hinge_tables.py` aborts if any of these fails. All pass, on both
tables.

- all three runs terminated normally in every row · n = 45 and n = 33 · both
  forces and ⟨S²⟩ present in every row · `group` found for every row
- `group_local` self-consistent: no row called stable has ⟨S²⟩ above 0.05
- **null probe, force, stable rows:**
  `|f_bs − f_rks| < max(10⁻³, 0.005 · f_rks)` eV Å⁻¹
- null probe, energy, stable rows: `|depth_mev| < 1`
- every unstable row has `depth_mev > 0`, i.e. E_RKS above E_BS
- every unstable row has `f_bs > f_rks`, exceptions frozen by name
- `uks_engrad` and `uks_sp` on the same side of the class boundary in every row
- `f_ref` present for every row of Table 1
- every one of the 12 reactions missing from Table 2 has a documented reason
- the eight core medians are frozen

**Why the stable-row bound is mixed and not a constant.** SCF convergence noise
on forces scales with the size of the forces, not absolutely. At the label
geometries the forces sit near 0.6 eV Å⁻¹; at the re-optimised saddles near
0.04, roughly a fifteenth. A fixed bound calibrated for the latter is too tight
for the former: rxn7945 is stable (⟨S²⟩ = 0.0030, depth below 0.005 meV, so
demonstrably the same solution) and differs by 1.87·10⁻³ eV Å⁻¹, which at
`f_rks` = 0.5815 is 0.3 % relative. The relative term takes over only above
`f_rks` = 0.2 eV Å⁻¹, so the test at the re-optimised geometries is unchanged.

**Frozen core numbers.** These eight medians are the paper-table numbers and are
asserted by the script; a drift beyond 5·10⁻⁴ eV Å⁻¹ (forces) or 5·10⁻³ (ratios)
aborts the run, so the table in the text and the table on disk cannot drift
apart.

| geometry | `group_local` | n | `f_rks` | `f_bs` | `ratio` |
|---|---|---|---|---|---|
| T1x label | stable | 27 | 0.6088 | 0.6087 | 1.00 |
| T1x label | unstable | 18 | 0.5885 | 1.6359 | 2.80 |
| def2-TZVPD re-optimised | stable | 18 | 0.0391 | 0.0392 | 1.00 |
| def2-TZVPD re-optimised | unstable | 15 | 0.0420 | 1.8695 | 32.36 |

Medians in eV Å⁻¹. The group sizes are the same under `group` and under
`group_local` — the two disagreeing reactions swap in opposite directions.

### Frozen findings

**The two `group`/`group_local` switchers.** The same two reactions disagree in
both tables, with the same sign. The check fires if the set changes.

| rxn | `group` | `group_local` | ⟨S²⟩ here, Table 1 / Table 2 |
|---|---|---|---|
| rxn1147 | stable | unstable | 0.5542 / 0.5562 |
| rxn10054 | unstable | stable | −0.000000 / 0.000000 |

rxn1147 is broken at the reference saddle but at none of the three model
transition states; rxn10054 the reverse, and it also carries the shallowest
breaking depths in the whole set (0.6, 1.0 and 21.6 meV at the model
geometries).

**The three tilt rows.** Among the **unstable** rows of Table 1, `f_bs < f_rks`
at exactly rxn4113, rxn6196 and rxn7957 — ratios 0.974, 0.395 and 0.671. This is
not a contradiction but the level offset: at the label geometry the change from
ωB97x/6-31G(d) to ωB97M-V/def2-TZVPD dominates *both* forces (`f_rks` there runs
0.48 to 1.08 eV Å⁻¹ for these three), and the ordering can invert. At the
re-optimised geometries it happens in no unstable row. The set is frozen by
name. On stable rows `f_bs` naturally falls marginally either side of `f_rks`;
that is what the null probe covers, not this check.

### Accounting: the 12 reactions without a converged saddle

All twelve share one reason, read from the files and not assumed: no `converged`
marker, the band optimiser never reached fmax 0.05. Last band residual from
`~/orca_neb_omol25/<rxn>/neb.log`:

| rxn | last fmax | | rxn | last fmax |
|---|---|---|---|---|
| rxn4519 | 0.0592 | | rxn4060 | 0.0876 |
| rxn7060 | 0.0600 | | rxn0894 | 0.0952 |
| rxn3107 | 0.0619 | | rxn4004 | 0.1056 |
| rxn0101 | 0.0637 | | rxn4003 | 0.1076 |
| rxn7937 | 0.0750 | | rxn1154 | 0.1118 |
| rxn1061 | 0.0847 | | rxn7949 | 0.1170 |

eV Å⁻¹, ASE projected band force. Three of them (rxn7949, rxn3107, rxn0894) had
a second attempt in the resume job and were cancelled after roughly a day.

**Does the truncation bias the class balance?** It can be answered by
measurement rather than argued, because all twelve *are* calculated in Table 1
at their label geometries. Crossed against `group_local` from Table 1:
**9 stable, 3 unstable** — 25 % unstable against 40 % over the full 45. The
exclusion is therefore **depleted** in unstable reactions, not enriched. Ready
as a table footnote:

> Twelve of the 45 reactions have no CI-NEB transition state at this level —
> the band optimiser did not reach fmax 0.05, last band residual 0.0592 to
> 0.1170 eV Å⁻¹ — and the exclusion is depleted in unstable reactions
> (3 of 12, 25 %, against 40 % overall).

**Thirteen of the 33 rows in Table 2 have `f_rks` above 0.05 eV Å⁻¹**, median
0.0404 over all 33, maximum 0.1668 (rxn1150); the others are rxn8832 0.1496,
rxn4513 0.1292, rxn1147 0.0985, rxn4522 0.0985, rxn4500 0.0901, rxn4518 0.0892,
rxn6196 0.0812, rxn4113 0.0810, rxn2553 0.0788, rxn9246 0.0697, rxn0346 0.0551,
rxn4498 0.0515. This is a property of band convergence, not a defect of the
measurement, and it has two causes that compound. First, the NEB criterion is
the projected band force in ASE per-atom-norm convention while `f_rks` is the
largest raw Cartesian component — different numbers at the same point. Second,
the `converged` marker means *the ODE solver did not exit with an exception*,
not *the tolerance was reached*; the mechanism is set out in §3 under **Run
accounting**, and in `results/neb_runs.csv` 21 of 133 marked runs do not meet
the criterion.

It does not harm the hinge test, which compares two forces at the **same** point
and treats `f_rks` as a measured reference rather than an assumed zero. The
separation survives with room to spare: the largest `f_rks` in the table,
0.1668, is still an order of magnitude below the median `f_bs` of the unstable
rows, 1.8695.
