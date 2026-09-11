# Marks benchmark stability analysis — Step 1 inventory

Paper: J. Marks, J. Vandezande, J. Gomes, *Reliable and Efficient Automated
Transition-State Searches with Machine-Learned Interatomic Potentials*,
[arXiv:2604.00405](https://arxiv.org/abs/2604.00405) (v1, 1 Apr 2026; only
version; not yet journal-published).

Reference level in that work: **ωB97X-V/def2-TZVP**, Q-Chem 6.0, SG-2 grid.

## 1. Zenodo record 19379882 — not obtained

Unreachable. Every request returns `504 Gateway Time-out` from three
independent vantage points:

| Route | Result |
|---|---|
| This workstation, `curl` + `Invoke-WebRequest` | 504 (3 retries) |
| DTU cluster `slid.fysik.dtu.dk`, `curl` | 504 |
| `https://doi.org/10.5281/zenodo.19379882` (DOI resolver) | 504 |

`https://zenodo.org/` itself also 504s, so this is a Zenodo-side outage, not a
local proxy or a missing record — **the existence of record 19379882 could not
be confirmed or refuted.** Worth retrying later.

Independently: **arXiv v1 cites no Zenodo DOI and no GitHub URL.** Its data
availability statement reads in full:

> The data supporting this study, including input geometries and scripts used to
> perform the transition state benchmarking calculations, will be made publicly
> available in a GitHub repository upon publication of this work and archived
> with a DOI.

The arXiv e-print source (2.4 MB) was downloaded and contains only `output.tex`,
`output.bbl` and figures — **no geometries**.

## 2. Reference geometries — obtained from the authors' own repository

Substitute source: **[`thegomeslab/fsm`](https://github.com/thegomeslab/fsm)**
(Gomes lab; commit tree of `main`, pushed 2024-11-14), the FSM implementation
released with the companion paper Marks & Gomes, *Incorporation of Internal
Coordinates Interpolation into the Freezing String Method*,
[arXiv:2407.09763](https://arxiv.org/abs/2407.09763) /
[JCTC 10.1021/acs.jctc.5c01492](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01492).

This is the right substitute rather than a generic literature fallback:

* same first author and same group as the benchmark paper;
* the companion paper states its reference transition states are at
  **ωB97X-V/def2-TZVP** — the identical level to the benchmark paper;
* it ships `ts.xyz`, `chg` and `mult` for **every** Baker and Sharada reaction.

Baker & Chan 1996 / Sharada et al. primary geometries were therefore *not*
needed. Copied verbatim to `geoms/`.

Coverage: **24/24 Baker rows and 9/9 Sharada rows — complete. Nothing missing.**
Baker directory `07` does not exist in the repo; the paper likewise lists 24
rows, and the paper's rows 1–24 map onto the 24 directories in ascending order.
That mapping is confirmed independently by charge and formula, not just by
order: paper row 7 = `08_formyloxyethyl` (C3H5O2, doublet), row 15 =
`16_h2po4_anion` (charge −1), row 19 = `20_hconh3_cation` (charge +1).

**Caveat still open:** the repo does not itself state the level of its
`ts.xyz`; that comes from the companion paper's text. Before interpreting any
result, a ωB97X-V/def2-TZVP `ENGRAD` on each structure will confirm it is a
stationary point at this level (‖∇E‖ ≈ 0). This is a quality control, not an
optimisation.

## 3. Charge and multiplicity

Taken from the repo's own `chg`/`mult` files, cross-checked against electron
parity from the formula. Full table in `inventory.csv`.

**Open-shell — RKS question not applicable (3):**

| Set | # | Directory | Formula | chg | mult |
|---|---|---|---|---|---|
| Baker | 4 | `04_ch3o` | CH3O | 0 | 2 |
| Baker | 5 | `05_cyclopropyl` | C3H5 | 0 | 2 |
| Baker | 7 | `08_formyloxyethyl` | C3H5O2 | 0 | 2 |

These get a UKS single point with ⟨S²⟩ logged, and
`notes="open-shell, RKS question not applicable"`.

**Charged but closed-shell (still in scope):** Baker 15 `16_h2po4_anion`
(−1, 50 e⁻) and Baker 19 `20_hconh3_cation` (+1, 24 e⁻).

### Two defects in the repo metadata

1. **`sharada/04_ethane_dehydrogenation` has `chg`/`mult` swapped**: the files
   contain `chg=1`, `mult=0`. Multiplicity 0 is unphysical. The geometry is an
   exact duplicate of `baker/12_ethane_h2_abstraction` (C2H6), which correctly
   carries `chg=0`, `mult=1`. **Corrected to 0 / 1.**
2. **`sharada/02_silane` is mislabelled relative to the paper.** The paper's
   Sharada row 2 reads "SiH2 + H2 → SiH4", but the file is C2H8Si (11 atoms) —
   silylene insertion into *ethane*, the same chemistry as Baker row 17. It is
   not a copy of the Baker structure though (max |Δx| = 0.146 Å), and the paper
   gives the two rows different gradient counts, so it was run as its own case.
   Kept as a distinct structure; the paper's row label is the error.

## 4. Distinct structures: 28, not 29

The paper says "29 unique reactions (24 from the Baker set and 5 from the
Sharada set)" and elsewhere "4 of which are shared". Comparing the actual
geometries, **five** Sharada reference TSs coincide with Baker ones:

| Sharada | Baker | max abs coordinate difference |
|---|---|---|
| `03_ethanal` | `14_vinyl_alcohol` | 0.000000 Å — exact |
| `04_ethane_dehydrogenation` | `12_ethane_h2_abstraction` | 0.000000 Å — exact |
| `06_diels_alder` | `09_parentdielsalder` | 0.000000 Å — exact |
| `05_bicyclobutane` | `06_bicyclobutane` | 0.000043 Å |
| `01_formaldehyde` | `03_h2co` | 0.001411 Å |
| `02_silane` | `18_silylene_insertion` | 0.146 Å — kept distinct |

The corresponding rows in the paper's tables are numerically identical across
all 12 model/workflow columns for those five, confirming they are the same
calculation.

So: **24 Baker + Sharada {02, 07, 08, 09} = 28 distinct structures.** The five
duplicates are carried in the output CSV as their own rows with the Baker twin's
numbers and a `notes` pointer, so both the 33-row and the 28-structure views are
available.

Of the 28: **25 closed-shell singlets get the RKS stability question**, 3 are
open-shell.

Largest system: `sharada/09_icr`, C18H32O5Si, 56 atoms, 194 electrons — the one
genuinely expensive single point.

## 5. Cross-tabulation data — extracted and validated

The per-reaction MLIP results are **fully recoverable from the paper's LaTeX
source**, so step 5 does not depend on Zenodo at all. All four tables parsed:

| Table | Algorithm | Set | Reactions | Runs |
|---|---|---|---|---|
| `tab:fsm_baker` | FSM | Baker | 24 | 288 |
| `tab:FSM_sharada` | FSM | Sharada | 9 | 108 |
| `tab:NEB_baker` | CI-NEB | Baker | 24 | 288 |
| `tab:NEB_sharada` | CI-NEB | Sharada | 9 | 108 |

792 rows → `marks_per_reaction.csv` (columns: algorithm, set, reaction_id, name,
model, workflow, gradients, success, failure_mode).

**Validation: all 48 footer columns (6 models × 2 workflows × 4 tables)
reproduce the paper's printed success rate *and* mean success cost exactly.**

Two things that had to be right to achieve that, both documented in the parser:

* A trailing LaTeX comment on Baker CI-NEB row 22 masks row 23 if comments are
  stripped per-segment rather than per-line; and rows 11 and 19 of that table
  exist twice, once commented out.
* **Failure mode (c) counts as a success.** Footnote (c) is "converges to
  correct TS with spurious imaginary frequency, additional cost to eliminate
  frequency … was added". Two such cells are italicised anyway (CI-NEB Baker
  AIMNet2 row 19; CI-NEB Sharada AIMNet2 row 8), but the paper's own success
  rates and mean costs count them as successes — 70.8 % and 33.3 % respectively
  are only reproducible that way. The superscript wins over the italics.

Failure modes: (a) alternate first-order saddle, (b) local minimum, (c) spurious
imaginary frequency [success], (d) no P-RFO convergence in 250 cycles, (e) SCF
convergence error, (f) multiple strong imaginary frequencies, (g) P-RFO step
failure, (h) internal-coordinate back-transformation failure.

## 6. Compute environment

ORCA **5.0.4** confirmed on the DTU cluster — `Program Version 5.0.4 - RELEASE`
in the existing `~/cheap_stab` outputs, loaded as
`module load gompi/2023a ORCA/5.0.4-gompi-2023a`. The module tree resolves only
on the compute nodes, not on the `slid` login node, so everything runs through
SLURM (`xeon24el8`), exactly as `pipeline/job_orca_cheap_stability.sh` already
does. Queue is empty (0 jobs), limit 20 concurrent.
