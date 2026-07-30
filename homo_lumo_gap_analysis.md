# HOMO–LUMO Gap Analysis at TS Geometries

## Goal

Compute HOMO–LUMO gaps for transition-state (TS) and reactant geometries of all 26 top-FOD benchmark reactions using DFT, as a complementary MR diagnostic alongside FOD, NOONs, and NEVPT2 corrections.

A small TS gap relative to the reactant gap signals near-degenerate frontier orbitals at the TS — consistent with multireference character.

---

## Method

**Level of theory:** PBE/def2-TZVP (PySCF 2.x, `dft.RKS`)

| Setting | Value |
|---------|-------|
| Functional | PBE |
| Basis | def2-TZVP |
| Grid level | 3 |
| Convergence threshold | 1×10⁻⁹ Ha |
| Max SCF cycles | 300 |
| Geometry source | ORCA NEB TS / ORCA NEB reactant (wB97M-V/def2-TZVP optimised) |

**Stability analysis:** After a converged RKS solution, `mf.stability(internal=True, external=True)` is called. The external stability check detects RKS→UKS instabilities (triplet instability). If found, a UKS calculation is run from the broken-symmetry density matrix to quantify the energy lowering.

**Fallback strategy for non-converging TS geometries:** Standard DIIS sometimes fails at strongly MR TS geometries. A sequential fallback is applied:
1. `level_shift=0.3`, DIIS
2. `level_shift=0.5`, DIIS
3. `level_shift=0.5, damp=0.5`, DIIS
4. Newton–Raphson (`mf.newton()`, no level shift)

The first strategy that converges is used; its label is recorded in the results JSON.

**Code:** `_homo_lumo_sp.py` (standard run), `_homo_lumo_levelshift.py` (fallback run)  
**SLURM jobs:** 10670226 (array 0–25, xeon24el8, 4 CPU, 12 GB), 10670258 (array 0–4, restart for failures)  
**Output:** `~/homo_lumo_results/{rxn}.json` on the cluster

---

## Results

Sorted by TS HOMO–LUMO gap, ascending. All reactant calculations converged with standard DIIS (12–13 iterations). No RKS→UKS instabilities were found in any converged TS.

| Reaction | TS gap (eV) | TS iters | Convergence method | R gap (eV) | FOD rank |
|----------|-------------|----------|--------------------|------------|----------|
| rxn4113  | 0.553       | 15       | standard DIIS      | 6.24       |  4 |
| rxn0101  | 0.603       | 15       | standard DIIS      | 5.75       | 23 |
| rxn8885  | 0.629       |  6       | Newton–Raphson     | 5.80       |  5 |
| rxn7945  | 0.700       | 17       | standard DIIS      | 3.89       |  6 |
| rxn7937  | 0.730       | 16       | standard DIIS      | 3.89       |  7 |
| rxn8832  | 0.769       | 17       | standard DIIS      | 4.99       |  2 |
| rxn0896  | 0.778       | 14       | standard DIIS      | 4.80       | 11 |
| rxn7949  | 0.785       | 17       | standard DIIS      | 3.89       |  1 |
| rxn1320  | 0.825       |  5       | Newton–Raphson     | 5.58       |  3 |
| rxn3107  | 0.848       | 49       | level_shift=0.3    | 5.85       | 13 |
| rxn0346  | 0.935       | 15       | standard DIIS      | 4.66       |  9 |
| rxn7060  | 0.957       | 15       | standard DIIS      | 3.91       | 15 |
| rxn1147  | 0.973       | 16       | standard DIIS      | 5.10       | 21 |
| rxn6196  | 0.985       | 15       | standard DIIS      | 5.85       |  8 |
| rxn5691  | 1.034       | 16       | standard DIIS      | 3.36       | 16 |
| rxn10054 | 1.062       | 14       | standard DIIS      | 4.99       | 25 |
| rxn1283  | 1.062       |  5       | Newton–Raphson     | 4.95       | 17 |
| rxn7936  | 1.068       | 14       | standard DIIS      | 3.89       | 20 |
| rxn8827  | 1.100       | 14       | standard DIIS      | 4.99       | 18 |
| rxn8837  | 1.118       |  5       | Newton–Raphson     | 4.99       | 14 |
| rxn4522  | 1.156       | 15       | standard DIIS      | 4.60       | 19 |
| rxn4518  | 1.166       | 17       | standard DIIS      | 4.60       | 12 |
| rxn1150  | 1.196       | 14       | standard DIIS      | 5.10       | 10 |
| rxn10005 | 1.212       | 17       | standard DIIS      | 5.20       | 24 |
| rxn7957  | 1.239       | 15       | standard DIIS      | 3.89       | 26 |
| rxn0894  | 1.512       | 16       | standard DIIS      | 4.80       | 22 |

**TS gap range:** 0.55–1.51 eV (all 26 reactions)  
**Reactant gap range:** 3.4–6.2 eV  
**Convergence failures (standard DIIS):** 5 reactions — rxn8885, rxn1320, rxn3107, rxn8837, rxn1283 — all recovered with level_shift=0.3 or Newton–Raphson.  
**RKS→UKS instabilities:** none found in any of the 26 converged TS calculations.

---

## Notes

- All TS gaps are dramatically smaller than reactant gaps, consistent with near-degenerate frontier orbitals at the TS for all 26 reactions.
- The 5 DIIS failures occur at TS geometries of strongly MR reactions (FOD ranks 3, 5, 13, 14, 17). The failure mode is DIIS collapse at ~15–16 cycles, not a cycle-limit hit, reflecting a genuinely ill-conditioned single-reference problem at those geometries.
- Newton–Raphson converges in 5–6 macro-iterations for the 4 NR cases; these micro-iterations internally use conjugate gradient solves and are not directly comparable to DIIS cycle counts.
- The T1 diagnostic was not computed (see `multireference_screening.md`). FOD and NOONs serve as the primary MR proxies; the PBE gap provides an independent electronic-structure indicator.
- Geometry source for TS: `~/orca_neb_results/{rxn}/transition_state.xyz` (ORCA NEB, wB97M-V/def2-TZVP). Geometry source for reactant: `~/orca_neb_results/{rxn}/reactant.xyz`.
