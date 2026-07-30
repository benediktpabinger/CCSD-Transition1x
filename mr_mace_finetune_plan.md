# MR-Specialist MACE Fine-Tuning Plan

## Motivation

Current MLIPs (MACE-bare, UMA, eSEN) find transition state geometries that deviate significantly
from CASSCF-optimized references for high multireference (MR) reactions. The deviation is not
a convergence artifact — for the highest-MR reactions, the single-reference PES has wrong topology
and the models converge to qualitatively incorrect saddle points.

Broken-symmetry DFT (BS-UKS NEB) was tested as an intermediate reference. For rxn7949 (highest
MR reaction, FOD rank 1), BS-UKS found a third stationary point 0.65 Å from both the CASSCF TS
and the ORCA RKS TS — worse than plain RKS. The mixed singlet/triplet PES is physically wrong for
genuine diradical TSs. This approach was abandoned.

The proposed solution: fine-tune MACE on CASSCF-level training data sampled from high-MR reactions
in the Transition1x training set, producing a specialist model used only in the high-MR regime.

---

## Train/Test Separation

**Test set (fixed, never used for training):** the 23 MR benchmark reactions evaluated in this work.

**Training data source:** high-MR reactions identified within the Transition1x training set. Geometry
sampling comes from the existing Transition1x DFT NEB paths; only the energies and forces are
replaced with CASSCF-level values.

---

## Pipeline

### Stage 1 — Screen Transition1x Training Set

Identify high-MR reactions using the T1 diagnostic (CCSD-level single point at the TS geometry).
T1 > 0.02 flags MR character for organic molecules.

- Input: ~10k reactions in Transition1x training set
- Run T1 at TS geometry for each reaction (2–5 min each)
- Select top ~200–500 reactions by T1 score
- Output: list of high-MR training reactions

Alternative cheap proxy: FOD score at PBE/6-31G* level (already computed for some reactions).

---

### Stage 2 — CASSCF Single-Points on Existing DFT Geometries

**Decision:** CASSCF single-points on existing Transition1x NEB geometries (Option A), not full
CASSCF NEB re-optimization. Rationale: the DFT NEB images are already sampled; recomputing
energies/forces at CASSCF level is sufficient to teach the model the correct PES shape in the TS
region. Full CASSCF NEB would cost ~13 hours/reaction × 200 reactions and introduces the orbital
tracking problem under geometry optimization — Option A avoids this.

**Cost estimate:**
- 200 reactions × 10 NEB images × 20 min CASSCF gradient / 100 parallel jobs ≈ 7 hours wall time

**Active space selection:** AutoCAS (DMRG-based orbital entanglement) at the TS image, run once
per reaction. AutoCAS is strictly better than AVAS here: no orientation dependence, no bloating,
active space selected by actual entanglement rather than AO projection. Cost is ~15–30 min per
reaction run once — negligible relative to the CASSCF gradient evaluations.

Requires: Block2 (DMRG, PySCF-compatible) + AutoCAS Python package on the cluster.
Fallback if unavailable: AVAS with `['C 2p', 'N 2p', 'O 2p', 'F 2p']`.

**Orbital tracking strategy:**
1. Run AutoCAS at the TS image (highest-energy DFT NEB image) → get CAS(n, m) and converged MOs
2. Fix CAS(n, m) for all images of this reaction (do not re-run AutoCAS per image)
3. Propagate MOs outward from the TS: use converged MOs from image i as initial guess for image i±1
4. Work toward reactant and product separately, never starting from the endpoints

This keeps all images on the same electronic state. CASSCF rotates orbitals within the fixed (n,m)
space — it does not re-select which orbitals are active.

MO coefficients are geometry-dependent (defined in the atom-centered AO basis), so they cannot be
reused directly across geometries. The propagation uses projection:

```python
from pyscf.scf.addons import project_mo_nr2nr
mo_init = project_mo_nr2nr(mol_image_i, mo_converged_i, mol_image_j)
```

This computes S_j⁻¹ · <χ_j|χ_i> · C_i — the best approximation of image i's MOs expressed in
image j's AO basis. Adjacent NEB images differ by small geometry steps so the projection is close
to the true MOs, and CASSCF converges quickly without exploring other states.

**What CASSCF convergence requires:**

CASSCF is a two-level optimization that iterates until both criteria are met:

1. Energy change ΔE < threshold (default 1e-8 Eh; 1e-6 Eh is sufficient for training data)
2. Orbital gradient ||dE/dθ|| < threshold — gradient w.r.t. orbital rotation angles goes to zero

Each macro iteration consists of:
- Solve CI problem (FCI or DMRG within the active space) → CI coefficients and density matrices
- Compute orbital gradient from density matrices
- Rotate MOs to reduce gradient (micro iterations)
- Check ΔE and ||grad||

Typically 10–30 macro iterations. Both energy and orbital gradient must converge — energy alone
is not sufficient, because a partially converged CASSCF can show small ΔE while still having a
large orbital gradient, producing inaccurate forces.

**Failure modes and solutions:**

**(a) Active space too large**
AutoCAS returns CAS(14,14) or larger → CI space ~10⁸ determinants → slow or memory crash.
Solution: cap at CAS(12,12). Drop the least-entangled orbitals above the limit. Flag the reaction
if truncation is severe.

**(b) MO projection quality**
Check the orbital overlap matrix after projection: diagonal elements <χ_j|ψ_i_projected>.
If any element < 0.5, the projection failed for that orbital (geometry step too large or
that orbital has no analog at the new geometry). Flag the reaction.

**(c) SCF not converged**
CASSCF macro iterations reach max_cycle without meeting convergence criteria.
Solution: increase max_cycle, adjust DIIS settings, or loosen convergence threshold for
training data (1e-5 Eh is acceptable). If still failing, exclude that image.

**(d) Root flipping — the most dangerous failure**
CASSCF converges to an excited state instead of the ground state. Both energy and gradient
appear converged, but the CI vector corresponds to a different electronic state. This happens
most often at the reactant and product images, where the active orbitals from the TS are
nearly doubly occupied and the ground/excited state gap is small.

Detection per image: check the leading CI vector weight. Expected behavior:
- TS: ~0.5–0.7 (genuine diradical)
- Reactant/product: >0.85 (near single-reference)
Unexpected drop (e.g. 0.4 at an intermediate image) → likely state flip.

Solutions in order of preference:
1. Restart that image from the opposite direction (project from image i+2 instead of i-1)
2. Use SA-CASSCF (state-average over 2 states) for the problematic image — smoother but
   energetically less accurate for the ground state
3. Exclude that reaction from training data

**(e) Active orbital rotation out**
CASSCF is allowed to mix active orbitals with inactive (doubly-occupied) ones. At geometries
far from the TS, an active orbital that was half-occupied at the TS may become nearly doubly
occupied, and CASSCF may swap it out for a different inactive orbital.
Detection: track occupation numbers across images. A discontinuous jump (e.g. 0.8 → 1.98
between adjacent images) indicates an orbital swap.
Solution: after convergence, re-sort active orbitals by overlap with the previous image's
active orbitals. If the active space character changed, restart with the re-sorted MOs.

**Failure detection — full path validation:**
After computing all images, plot E_CASSCF along the NEB path. Accept the reaction for training
only if:
- Energy profile is smooth and unimodal (one hump at the TS)
- No jumps > 0.1 eV between adjacent images
- Leading CI weights vary smoothly along the path
- Occupation numbers vary continuously

Expect ~20–30% of reactions to fail this check and be excluded. Budget for this in the
200-reaction selection.

**Output per reaction:** list of (geometry, E_CASSCF, F_CASSCF) tuples for ~10 images.
Total training set: ~2000–5000 configurations.

---

### Stage 2b — Active Learning Loop (if needed)

Option A trains on CASSCF forces at DFT geometries. For the highest-MR reactions, the DFT and
CASSCF reaction paths may diverge significantly. After initial training, MACE-MR NEB may explore
regions of configuration space not covered by the training data — the model extrapolates and
predictions become unreliable. The active learning loop fixes this iteratively.

**Stopping criterion first:** compare NEB paths from consecutive model versions on a held-out
subset of training reactions. If max RMSD between v(k) and v(k-1) paths < 0.02 Å, the model
is self-consistent with its training data — converged. In practice 2–3 rounds is typically
sufficient.

**Round 0 — Bootstrap**
```
CASSCF single-points at DFT NEB geometries  →  train MACE-MR v0
```

**Round k — Iterate**
```
1. Run MACE-MR v(k-1) NEB on training reactions
   → new NEB paths, potentially shifted away from DFT paths

2. Uncertainty filter (committee model)
   Train N=5 MACE-MR models with different random seeds.
   For each new geometry, compute variance of predicted forces across the committee.
   Keep only geometries where variance > threshold (out-of-distribution, model extrapolating).
   Discard geometries the model already handles confidently.

3. Compute CASSCF only at uncertain geometries
   → ~200–500 new CASSCF calculations per round (not all new geometries)
   → use same orbital tracking protocol as Stage 2

4. Add new (geometry, E_CASSCF, F_CASSCF) to training set
   → retrain MACE-MR v(k)
```

**Why the uncertainty filter is essential:** without it, every round requires CASSCF at all
new NEB path geometries — thousands of calculations per round. The committee variance identifies
specifically which geometries are out-of-distribution. Round 1 might add ~500 new points,
round 2 ~100, round 3 converged.

**Expected convergence intuition:**
- Round 0: model trained on DFT geometries → NEB stays near DFT path → mostly in-distribution
- Round 1: NEB shifts toward CASSCF path → some new regions → compute CASSCF there
- Round 2: model knows the CASSCF path region → NEB converges there → few uncertain geometries
- Round 3: converged

**Important:** run the active learning loop on training reactions only. The 23 benchmark
reactions are never used during this process — they are evaluated only at the end.

---

### Stage 3 — Fine-Tune MACE

- Base model: MACE-MP (pretrained foundation model)
- Training data: MR configurations from Stage 2 + a small fraction of general DFT data for
  regularization (prevents catastrophic forgetting of general chemistry)
- Fine-tuning: `mace_run_train --foundation_model mace_mp`
- Training format: extended XYZ with energies and forces (standard MACE input)
- Output: MACE-MR (specialist model)

Active space size varies across reactions — this is fine. MACE sees only (geometry, energy, forces)
and does not know about the active space. CASSCF interaction energies are comparable across
reactions with different active space sizes because the energy reference (isolated atom contributions)
is consistent.

---

### Stage 4 — Deployment with MR Routing

For any new reaction:
1. Compute T1 diagnostic at the approximate TS geometry (cheap single-point)
2. If T1 < 0.02: use standard MACE-MP (general model)
3. If T1 ≥ 0.02: use MACE-MR (specialist model)

The specialist model is not used as a general replacement — only in the regime where it was trained.

---

### Stage 5 — Evaluation on Benchmark

Run NEB with MACE-MR on the 23 MR benchmark reactions. Compute TS RMSD against CASSCF OptTS
reference. Compare to:
- MACE-bare (current best general model)
- ORCA DFT (upper bound)

Primary question: does MACE-MR reduce TS RMSD for the top-MR reactions (where MACE-bare currently
fails), without degrading on the lower-MR reactions?

---

## Open Questions

- **Option A sufficiency:** if the DFT and CASSCF reaction paths diverge significantly for the
  highest-MR reactions, the fine-tuned model will NEB into regions not covered by training data.
  If this occurs, the fix is one iteration: run MACE-MR NEB → compute CASSCF on those new
  geometries → retrain. This is the active learning loop.

- **AutoCAS software setup:** Block2 and the AutoCAS Python package need to be installed on the
  DTU cluster. Block2 is PySCF-compatible and pip-installable; AutoCAS is available from the Reiher
  group. If setup is not feasible, fall back to AVAS with `['C 2p', 'N 2p', 'O 2p', 'F 2p']`.

- **T1 diagnostic cost at scale:** screening 10k reactions at CCSD level may be expensive. A
  cheaper proxy (spin contamination in UHF, or FOD at DFT level) could pre-filter before applying
  the full T1 criterion.

- **Regularization data:** how much general DFT data to mix in during fine-tuning is not yet
  determined. Too little → catastrophic forgetting. Too much → MR correction washed out.
