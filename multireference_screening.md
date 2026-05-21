# Multireference Benchmark

## Goal

Identify the reactions in the Transition1x test set with the strongest multireference (MR) character. These reactions form a **"tough" benchmark** — a subset where single-reference methods (wB97X-D3, MACE) are expected to struggle, and where high-level multireference methods (NEVPT2, CASSCF) are necessary to get the right answer.

This matters because:
- MACE is trained on wB97X-D3 single-reference DFT. If the true wavefunction is strongly multireference, the training data itself is unreliable.
- A tough benchmark isolates cases most likely to reveal systematic failures of single-reference models.
- Prior work on rxn0103 showed the gap can be hundreds of meV above chemical accuracy (NEVPT2 barrier 4287 meV vs wB97X-D3 4656 meV).

---

## Step 1: Reaction Selection via FOD Screening

**FOD** (Fractional Occupation number weighted Density, Grimme & Hansen 2015) screens all 279 converged test-set TS geometries cheaply.

A DFT calculation is run at high electronic temperature (T_el = 5000 K) with Fermi-Dirac smearing. Near-degenerate orbitals acquire fractional occupations. The MR index is:

```
NFOD = Σᵢ |nᵢ - n⁰ᵢ|
```

where `nᵢ` is the smeared occupation and `n⁰ᵢ` is the integer reference (0 or 2). Strongly correlated orbitals with occupation ≈ 1 contribute most.

**Approximate thresholds:**
| NFOD | Interpretation |
|------|---------------|
| < 0.05 | Negligible MR character |
| 0.05–0.5 | Mild |
| 0.5–1.5 | Significant MR character |
| > 1.5 | Strongly multireference |

**Why FOD over T1 diagnostic:**
- FOD is a DFT single-point — 10–20× cheaper than CCSD (needed for T1).
- T1 and FOD give consistent rankings in practice.
- Screening 279 reactions with T1 would take days; FOD finishes in one batch job.

**Why not CASSCF directly for screening:**
- CASSCF requires choosing an active space per reaction — not practical to automate reliably at scale.
- FOD requires no such choice.

**Implementation:**

| Setting | Value | Reason |
|---------|-------|--------|
| Functional | PBE | Cheap; functional choice negligible for FOD |
| Basis | def2-SVP | Grimme standard |
| Temperature | 5000 K | Grimme standard; probes near-degenerate orbitals |
| Code | PySCF | smearing via `pyscf.scf.addons.smearing_` |
| Geometry | ORCA NEB TS | wB97M-V/def2-TZVP optimised TS — most relevant geometry |

**Reactions screened:** 279 (all test reactions with a converged NEB TS). The top 10 by NFOD form the benchmark set.

**Benchmark reactions (top 10 by NFOD):**
`rxn7949, rxn8832, rxn1320, rxn4113, rxn8885, rxn7945, rxn7937, rxn6196, rxn0346, rxn1150`

---

## Step 2: Level-of-Theory Ladder

All four methods are evaluated as single points on the same **wB97M-V/def2-TZVP optimised geometries** (from the ORCA NEB results). This means geometry and electronic-structure effects are separated: differences between methods reflect level of theory only, not geometry relaxation.

The geometry effect at a stationary point is second-order (∝ ½ΔgᵀHΔg, which vanishes since Δg=0 at R and P, and is small at the TS for a well-converged geometry).

| Step | Method | Geometries | Notes |
|------|--------|-----------|-------|
| 1 | wB97X-D3/6-31G(d) | All NEB images (1240 total) | ORCA, full path |
| 2 | wB97M-V/def2-TZVP | All NEB images (1240 total) | Free from neb.db |
| 3 | CCSD(T)/def2-TZVP | R, TS, P only | PySCF, 24 OMP threads |
| 4 | NEVPT2/AVAS/def2-TZVP | R, TS, P only | PySCF, see below |

Steps 1 and 2 give energy profiles across the full NEB path, allowing barrier extraction from the last 10 images. Steps 3 and 4 give only stationary-point energies.

---

## Step 3: CCSD(T) Single Points

RHF → RCCSD → CCSD(T) with PySCF on R, TS, P. Straightforward single-reference calculation; no active space needed.

**MR diagnostic from CCSD(T) output:** The triples correction `(T)` is larger at the TS than at R/P for MR systems. Formally, (T) is unreliable when MR character is strong (the perturbative triples expansion diverges). Reactions rxn7949 and rxn8832 show the largest TS–R difference in the triples correction (−0.028 Ha), flagging them as the most problematic for CCSD(T).

---

## Step 4: NEVPT2/AVAS

### Approach

CASSCF with an AVAS-selected active space, followed by strongly-contracted NEVPT2 (SC-NEVPT2). All calculations use def2-TZVP.

**Why AVAS instead of manual active space selection:**
AVAS (Atomic Valence Active Space, Sayfutyarova et al. 2017) automates active space construction by projecting molecular orbitals onto a set of target atomic orbital (AO) types. It is reproducible and avoids human bias in orbital selection. The tradeoff is that the resulting active space may not be optimal — see reliability notes below.

**Why AVAS at TS, projected to R and P:**
- The active space is defined once at the TS, where MR character is strongest and most physically relevant.
- The converged TS MO coefficients are projected onto the R and P geometries using `mcscf.project_init_guess`. This ensures R, TS, and P share the same physical orbital space, making the barriers meaningful (they reflect the same electrons and orbitals throughout).
- Running AVAS independently at R, TS, P could select different active spaces, making the energy differences unphysical.

**AO targets and threshold:**
```
Target AOs:  C 2pz, N 2p, O 2pz, F 2pz
Threshold:   0.2
```

This follows the approach validated end-to-end for rxn0103, which gave a manageable (16e, 10o) active space and a physically sensible barrier. Active spaces across the 10 reactions range from (14e, 10o) to (18e, 13o).

`2pz` (m=0 component of p) preferentially targets π-type orbitals. For the 10 MR reactions identified by FOD — all organic H/C/N/O/F molecules — MR character is dominated by near-degenerate π and π* orbitals at the TS. Using all three p components (`2p`) selects the entire carbon σ-skeleton as well, producing active spaces of (32–34e, 22–26o) that are intractable for FCI-based CASSCF.

`N 2p` (all three components) is kept for nitrogen because N lone pairs (which may not align with z) are frequently part of the active chemistry.

The theoretical limitation of `2pz` (orientation-dependence for non-planar molecules) is accepted as a known approximation, documented here, and can be checked post-hoc via natural orbital occupancies.

**CASSCF convergence settings:**
```
max_cycle_macro = 1000
max_stepsize    = 0.05   (orbital rotation damping)
conv_tol        = 1e-8   (default)
```

`max_stepsize=0.05` damps orbital rotation steps to prevent oscillation between near-degenerate configurations — a common failure mode for CASSCF on MR systems. Without damping, 4 of the 10 reactions failed after 500 iterations with the energy oscillating by ~10⁻⁵ Ha between steps. With damping, those reactions are retried. If CASSCF still does not converge after 1000 damped iterations, the reaction is excluded from the NEVPT2 benchmark.

### Reliability and Limitations

NEVPT2/AVAS results should be interpreted carefully:

**1. Active space quality is not manually validated.**
The AVAS threshold determines which orbitals are included. Too high a threshold: important orbitals are excluded, the active space is incomplete, and NEVPT2 misses correlation energy. Too low: the active space becomes intractably large. The threshold=0.2 with `2pz` targets is a compromise validated on rxn0103, not a guarantee for all reactions.

**2. Diagnostic: natural orbital occupancies.**
After each CASSCF converges, the natural orbital occupancies (`nat_occ`) are saved. Trustworthy results have at least one occupation in the range 0.02–1.98 (indicating genuine MR character is captured in the active space). If all occupations are near 0 or 2, the active space likely missed the relevant orbitals despite having CASSCF converged — the result looks good numerically but is physically wrong.

**3. CASSCF convergence to local minima.**
CASSCF is not guaranteed to find the global orbital minimum. The projected TS MOs provide a physically motivated starting guess for R and P, which reduces (but does not eliminate) this risk. Reactions where CASSCF converged but nat_occs look suspicious warrant re-examination with a different initial guess.

**4. NEVPT2 itself is approximate.**
For very strongly MR systems (like some biradicals), NEVPT2 is a second-order perturbation theory on top of CASSCF. Higher-order effects or larger active spaces (MRCI, DMRG) would be needed for chemical accuracy. The most MR reactions in this set (rxn7949, rxn8832, largest triples correction delta) are where NEVPT2 is least reliable.

**5. Active space size varies across reactions.**
Each reaction's AVAS selects a different (nelecas, ncas). Reactions with larger active spaces have more complete correlation treatment. The benchmark results are not from a uniform method — this should be stated clearly when reporting.

### Post-processing checks (to run after results are collected)

- Flag any reaction where all nat_occ values are > 1.95 or < 0.05 → active space likely incomplete.
- Flag any reaction where CASSCF did not converge after 1000 damped iterations → result excluded.
- Report active space (ncas, nelecas) for each reaction alongside the barrier.
- Compare NEVPT2 and CCSD(T) barriers: large disagreement (> 200 meV) on a reaction where CCSD(T) triples correction is small suggests NEVPT2 active space is incomplete.

---

## Results

### Barrier Table: CCSD(T) vs NEVPT2

All barriers in meV. NEVPT2 uses AVAS active space defined at TS, projected to R/P.

| Reaction | Active Space | CCSD(T) fwd | CCSD(T) rev | NEVPT2 fwd | NEVPT2 rev | ΔNEVPT2–CCSD(T) fwd | Reliability |
|----------|-------------|-------------|-------------|------------|------------|---------------------|-------------|
| rxn7949  | (16e, 12o)  | 3209.6      | 3382.9      | 3253.9     | 3154.9     | +44                 | Reliable    |
| rxn8832  | (18e, 13o)  | 2621.4      | 1945.2      | 2540.3     | 2230.6     | −81                 | Reliable    |
| rxn1320  | (16e, 10o)  | 3051.2      | 3213.4      | 3146.7     | 3414.2     | +96                 | ⚠ Red flag |
| rxn4113  | —           | 5345.6      | 4411.9      | —          | —          | —                   | ✗ Failed   |
| rxn8885  | (14e, 11o)  | 3563.7      | 2330.9      | 3642.7     | 2143.6     | +79                 | Reliable    |
| rxn7945  | (16e, 12o)  | 3923.3      | 875.0       | 3943.3     | 1019.9     | +20                 | Reliable    |
| rxn7937  | (16e, 12o)  | 3858.3      | 763.7       | 3764.2     | 778.6      | −94                 | Reliable    |
| rxn6196  | (14e, 12o)  | 4281.8      | 687.9       | 4180.8     | 540.0      | −101                | Reliable    |
| rxn0346  | (14e, 10o)  | 3336.0      | 1353.0      | 3212.9     | 1110.3     | −123                | Reliable    |
| rxn1150  | (14e, 10o)  | 3460.0      | 756.6       | 3362.0     | 481.0      | −98                 | ⚠ Red flag |

For the 7 reliable reactions the NEVPT2–CCSD(T) spread on forward barriers is −123 to +96 meV, consistent with NEVPT2 accuracy on moderately MR systems.

---

### Natural Orbital Occupancy Assessment

After CASSCF converges, natural orbital occupancies are computed from the 1-RDM. An occupation is "fractional" if 0.05 < n < 1.95. Fractional occupations at a geometry indicate the active space genuinely captures MR character there. Near-0 or near-2 occupations at a particular geometry mean CASSCF is essentially HF at that point — the active orbitals are fully occupied or empty — and the NEVPT2 correction to the *barrier* at that geometry is meaningless.

**Criterion:**
- ≥1 fractional occupation at reactant AND TS AND product → balanced active space → **Reliable**
- 0 fractional occupations at any geometry → active space idle there → **Red flag**

| Reaction | Reactant fractional | TS fractional (key pair) | Product fractional | Assessment |
|----------|--------------------|--------------------------|--------------------|------------|
| rxn7949  | 6 (1.942, 1.928, 0.075, …) | 8 (1.357/**0.650**) | 6 (1.944, 1.928, 0.075, …) | Reliable — balanced; strong bond-breaking at TS |
| rxn8832  | 4 (1.941, 1.940, 0.065, …) | 6 (1.380/**0.628**) | 6 (1.677/**0.329**) | Reliable — MR at TS and product; product is partial biradical |
| rxn1320  | **0** (max non-2: 0.024) | 2 (1.483/**0.523**) | 2 (1.938, 0.065) | ⚠ Red flag — active space idle at reactant; forward barrier artificially low |
| rxn4113  | — | — | — | ✗ CASSCF did not converge at product; result excluded |
| rxn8885  | 6 (1.944, 1.941, 0.067, …) | 6 (1.940, 1.931, 0.102, …) | 4 (1.947, 1.940, 0.061, …) | Reliable — mild, uniform MR throughout; consistent active space |
| rxn7945  | 6 (1.942, 1.929, 0.074, …) | 6 (1.947, 1.922, 0.088, …) | 6 (1.781/**0.228**) | Reliable — balanced; product has most MR (partial biradical character) |
| rxn7937  | 6 (1.944, 1.929, 0.073, …) | 6 (1.946, 1.926, 0.086, …) | 4 (1.803/**0.207**) | Reliable — balanced; product has most MR |
| rxn6196  | 6 (1.945, 1.941, 0.062, …) | 6 (1.839/**0.171**) | 8 (1.942, 1.940, 0.067, …) | Reliable — TS has most MR; active space well-defined throughout |
| rxn0346  | 2 (1.930, **0.077**) | 2 (1.724, **0.287**) | 2 (1.928, **0.080**) | Reliable — mild MR at R/P, strong at TS; minimal but consistent active space |
| rxn1150  | **1** (0.051 only) | 4 (1.933, 1.900, 0.114, …) | 4 (1.938, 1.880, 0.128, …) | ⚠ Red flag — reactant essentially SR (0.051 barely above threshold); TS-biased active space; forward barrier unreliable |

**Summary: 7 reliable, 2 red flag (rxn1320, rxn1150), 1 failed (rxn4113).**

#### rxn1320 red flag: detail

AVAS at the TS selects orbitals with occupations 1.483/0.523 — a strongly bond-breaking orbital pair. At the reactant, those same orbitals are completely filled/empty (all occupations ≥ 1.977 or ≤ 0.024). The active space describes the TS correctly but is essentially HF at the reactant. The NEVPT2 correction at R is near zero, while at TS it is substantial — so the computed barrier reflects the TS correlation without a matching correction at the reactant. The forward barrier will be biased. CCSD(T) is used as the reference for rxn1320.

#### rxn1150 red flag: detail

Same failure mode. The only fractional occupation at the reactant is 0.051 — just barely above the detection threshold. NEVPT2 adds essentially no correlation energy at the reactant while adding significant correlation at TS and product. The forward barrier (NEVPT2: 3362 meV vs CCSD(T): 3460 meV) is −98 meV lower, and the reverse barrier (NEVPT2: 481 meV vs CCSD(T): 757 meV) is −276 meV lower. The large deviation on the reverse barrier, where the product is also MR (1.938, 1.880, 0.128, 0.074) but the reactant is not, is consistent with this interpretation.

#### rxn4113: CASSCF convergence failure at product

CASSCF converged at TS and reactant but failed at the product geometry even after 1000 damped iterations. The product likely has a qualitatively different electronic structure (possibly a diradical or near-degenerate closed-shell / open-shell pair) that requires a different initial guess or a larger active space. The CCSD(T) barrier is used as reference. rxn4113 is excluded from the NEVPT2 benchmark.

---

## Scripts

| File | Purpose |
|------|---------|
| `pipeline/screen_fod.py` | Per-reaction FOD at TS, writes `fod_results/{rxn}.json` |
| `pipeline/job_fod_screen.sh` | SLURM: 12 nodes × ~24 reactions, 6 parallel workers |
| `pipeline/collect_fod.py` | Aggregate, rank by NFOD, save `fod_results/fod_ranking.json` |
| `pipeline/mr_benchmark_setup.py` | Extract NEB geometries, generate ORCA wB97X-D3 inputs |
| `pipeline/job_mr_sp.sh` | SLURM: wB97X-D3 SPs on all 1240 NEB images |
| `pipeline/mr_benchmark_collect_sp.py` | Collect wB97X-D3 and wB97M-V barriers from last 10 images |
| `pipeline/mr_benchmark_ccsdt.py` | CCSD(T)/def2-TZVP on R, TS, P via PySCF |
| `pipeline/job_mr_ccsdt.sh` | SLURM array: one xeon24el8 node per reaction, 12h |
| `pipeline/mr_benchmark_nevpt2.py` | NEVPT2/AVAS on R, TS, P via PySCF (AVAS at TS, project to R/P) |
| `pipeline/job_mr_nevpt2.sh` | SLURM array: one xeon24el8 node per reaction, 24h |

---

## TODO

- [ ] Compute all relevant levels of theory

---

## References

- Grimme, S. & Hansen, A. (2015). A Practicable Real-Space Measure and Visualization of Static Electron-Correlation Effects. *Angew. Chem. Int. Ed.*, 54, 12308.
- Sayfutyarova, E. R., Sun, Q., Chan, G. K.-L. & Knizia, G. (2017). Automated Construction of Molecular Active Spaces from Atomic Valence Orbitals. *J. Chem. Theory Comput.*, 13, 4063.
- Angeli, C., Cimiraglia, R. & Malrieu, J.-P. (2001). N-electron valence state perturbation theory: a fast implementation of the strongly contracted variant. *Chem. Phys. Lett.*, 350, 297.
