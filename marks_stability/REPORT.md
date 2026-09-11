# RKS stability of the Marks benchmark reference transition states

Reference transition states of J. Marks, J. Vandezande, J. Gomes,
[arXiv:2604.00405](https://arxiv.org/abs/2604.00405), classified RKS-stable
or RKS-unstable at **that paper's own level, wB97X-V/def2-TZVP** -- deliberately
not the wB97M-V this project trains against, since the point is to measure the
instability on Marks' surface.

## Result

| | distinct structures |
|---|---|
| stable | 24 |
| **unstable** | **1** |
| open-shell (question not applicable) | 3 |
| total | 28 |

The single unstable case is **bicyclo[1.1.0]butane -> trans-butadiene** (Baker 6 / Sharada 5):
lowest stability eigenvalue -0.010400 au, <S**2> = 0.192139, depth 13.72 meV.

The stable cases are not marginal. Ranked by the lowest stability eigenvalue,
the smallest positive margin is an order of magnitude clear of the
marginally-stable band (+0.001 to +0.008 au) seen in this project's
Transition1x multireference set:

| structure | lambda_min (au) |
|---|---|
| baker_19_hnccs | +0.021315 |
| sharada_07_hexadiene | +0.034288 |
| sharada_02_silane | +0.043848 |
| baker_18_silylene_insertion | +0.043988 |
| baker_12_ethane_h2_abstraction | +0.045061 |
| baker_25_hcnh2 | +0.048901 |

## Method

Per structure, ORCA 5.0.4:

1. `! RKS wB97X-V def2-TZVP def2/J RIJCOSX TightSCF DEFGRID3 EnGrad` -- the
   restricted reference energy, and the gradient as a check that the geometry
   really is a stationary point at this level.
2. `! UKS ...` with `%scf STABPerform true STABRestartUHFifUnstable true end`,
   which rotates into the broken-symmetry solution when one is lower.

Open-shell structures get step 1 only (as UKS), with <S**2> logged.
depth_meV = (E_RKS - E_BS) * 27211.386.

## Quality control

* Every SCF converged; every ORCA run terminated normally.
* **<S\*\*2> = 0.000000 for all 24 stable cases** (QC b).
* Verdicts are taken from ORCA's explicit "Stability Analysis indicates a
  {stable, UNSTABLE} HF/KS wave function" line. Nothing is inferred from the
  absence of an UNSTABLE line -- an earlier version of the collector did that
  and turned a batch of failed runs into 14 apparent "stable" results.
* Max |gradient| over all structures 2.85e-04 Eh/bohr, median 1.31e-04: the geometries are genuine wB97X-V/def2-TZVP
  stationary points, which is what licenses calling them *the* reference TSs.
* Control (QC c): bicyclobutane was expected unstable after Hait
  (B3LYP/def2-SVP, <S**2> = 0.24) and came out unstable, <S**2> = 0.19. The
  functional and basis differ, so the values are not directly comparable; the
  qualitative agreement is what the control tests.

One systematic effect worth recording: every stable case shows a small
**positive RKS->UKS energy offset** growing with system size (0.06 meV for HCN
to 8.94 meV for the 56-atom Ireland-Claisen) despite <S**2> = 0 exactly, i.e.
numerical difference between the restricted and unrestricted code paths, not
physics. It matters only because the depth mixes the two paths. A plain UKS
single point on bicyclobutane (<S**2> = 0.000000) puts the offset at
+0.39 meV, so the depth is 13.72 meV as defined above and 14.11 meV against a
same-path reference. The verdict does not depend on it.

## Inventory and deviations

Full detail in [INVENTORY.md](INVENTORY.md). The short version:

* **Zenodo record 19379882 was never reachable** -- 504 from this workstation,
  from the DTU cluster and from the DOI resolver, with `zenodo.org` itself
  down, so its existence could be neither confirmed nor refuted. arXiv v1 in
  any case cites no Zenodo DOI and no GitHub URL; its data statement promises
  them "upon publication".
* Geometries instead come from **[`thegomeslab/fsm`](https://github.com/thegomeslab/fsm)**,
  the Gomes lab's own FSM code released with the companion paper (Marks &
  Gomes, [arXiv:2407.09763](https://arxiv.org/abs/2407.09763) / JCTC), whose
  reference transition states are at the identical wB97X-V/def2-TZVP level. It
  carries `ts.xyz`, `chg` and `mult` for all 24 Baker and all 9 Sharada
  reactions -- complete, nothing missing.
* **28 distinct structures, not 29.** Five Sharada reference TSs coincide with
  Baker ones (three bit-identical, two within 0.0015 A) and their paper table
  rows are identical across all 12 columns. The paper's own "24 + 5 = 29"
  does not hold geometrically. All 33 table rows are kept in the CSV, with the
  duplicates carrying a note.
* Two defects in the repository metadata: `sharada/04` has `chg`/`mult`
  swapped (1/0, multiplicity 0 being unphysical; corrected to 0/1), and
  `sharada/02` is C2H8Si -- silylene insertion into ethane -- not the
  "SiH2 + H2 -> SiH4" its paper row claims.
* Parsing the paper's tables surfaced that **failure mode (c) counts as a
  success** in the authors' own aggregates despite being italicised in two
  cells; all 48 footer columns reproduce exactly only under that reading.

## Stability class vs. MLIP search outcome

Counts only; no test is applied, and none would be meaningful.

| algorithm | class | runs | failures | failure rate | mean gradients |
|---|---|---|---|---|---|
| FSM | stable | 336 | 33 | 9.8% | 13.36 |
| FSM | unstable | 24 | 6 | 25.0% | 22.44 |
| FSM | open-shell | 36 | 4 | 11.1% | 8.09 |
| CI-NEB | stable | 336 | 109 | 32.4% | 6.42 |
| CI-NEB | unstable | 24 | 4 | 16.7% | 6.30 |
| CI-NEB | open-shell | 36 | 12 | 33.3% | 5.00 |

**The unstable class is one reaction.** It appears as two table rows because
Sharada repeats it. Any rate quoted for it describes a single reaction across
12 model/workflow combinations, and the FSM and CI-NEB numbers point in
opposite directions (25.0% against 9.8%, then 16.7% against 32.4%). There is
no signal here.

### Where the OMol25 models actually fail

Not on the unstable reaction. Ranked by failures among the four OMol25-trained
models (eSEN-S, UMA-S, UMA-M, MACE-OMol25) over both algorithms:

| set | # | reaction | class | OMol25 failures |
|---|---|---|---|---|
| baker | 22 | HNC + H2 -> H2CNH | stable | 13 |
| baker | 24 | HCNH2 -> HCN + H2 | stable | 10 |
| baker | 7 | formyloxyethyl 1,2-migration | open-shell | 8 |
| baker | 8 | parent Diels-Alder cycloaddition | stable | 8 |
| baker | 15 | H2O + PO3- -> H2PO4- | stable | 8 |
| sharada | 9 | silyl ketene acetal -> silyl ester Ireland-Claisen | stable | 8 |
| baker | 9 | s-tetrazine -> 2HCN + N2 | stable | 7 |
| baker | 11 | CH3CH3 -> CH2CH2 + H2 | stable | 7 |
| ... | | | | |
| baker | 6 | bicyclo[1.1.0]butane -> trans-butadiene | **unstable** | **2** |

Every one of the worst cases is RKS-stable, including the two the paper itself
singles out as characteristic failures (Baker 11 and 24), which sit at
lambda_min = +0.045 and +0.049. The unstable reaction is among the
better-behaved ones for these models: its only OMol25 failures are two
FSM-native runs, and all four models succeed on it under CI-NEB in 3-4
gradients.

A comment left in the paper's LaTeX source, on the worst failure case
(Baker 22), points at the actual mechanism:

> all of these off target saddle points correspond to an in place cis addition
> while the lower energy in plane trans addition is the reference.
> potentially(?) an internal coords failure here

That is the wrong saddle point on a well-behaved surface. Consistent with the
dominant failure modes across all 792 runs being (a) alternate first-order
saddle and (b) local minimum; mode (e), SCF convergence error, occurs twice.

## Limitation

With 24 of 25 closed-shell reference transition states stable, this benchmark
cannot answer whether RKS instability predicts transition-state-search
failure -- there is no power to detect such a link even if one exists. The
Baker and Sharada sets are classical, well-behaved organic reactions; that
they are almost uniformly stable is itself the finding. The question belongs
to a set selected for multireference character, such as this project's
Transition1x subset, where 18 of 26 are unstable.

## Files

* `marks_stability.csv` -- the deliverable: set, reaction_id, name, E_RKS_Ha,
  stable, E_BS_Ha, depth_meV, S2_BS, notes, plus structure, lambda_min_au and
  max_grad_Eh_bohr.
* `marks_per_reaction.csv` -- 792 rows parsed from the paper's four tables.
* `INVENTORY.md`, `CROSSTAB.md`, `orca_outputs/`, `geoms/`, `scripts/`.
