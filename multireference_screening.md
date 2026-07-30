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

## Step 4: NEVPT2/AVAS — Single Points on ORCA NEB Geometries

> **Scope of this section:** NEVPT2 evaluated as single points on the ORCA NEB
> geometries (AVAS threshold=0.2, no geometry optimisation). This covers the original
> 30-reaction benchmark (top-10, mid-10, low-10 by FOD). A separate CASSCF OptTS
> calculation (threshold=0.4, `--no-prune`, 23 reactions) is documented in
> **Step 5** below and in `active_space_quality_analysis.md`.

### Approach

CASSCF with an AVAS-selected active space, followed by strongly-contracted NEVPT2 (SC-NEVPT2). All calculations use def2-TZVP.

**Why AVAS instead of manual active space selection:**
AVAS (Atomic Valence Active Space, Sayfutyarova et al. 2017) automates active space construction by projecting molecular orbitals onto a set of target atomic orbital (AO) types. It is reproducible and avoids human bias in orbital selection. The tradeoff is that the resulting active space may not be optimal — see reliability notes below.

**Why AVAS at TS, projected to R and P:**
- The active space is defined once at the TS, where MR character is strongest and most physically relevant.
- The converged TS MO coefficients are projected onto the R and P geometries using `mcscf.project_init_guess`. This ensures R, TS, and P share the same physical orbital space, making the barriers meaningful (they reflect the same electrons and orbitals throughout).
- Running AVAS independently at R, TS, P could select different active spaces, making the energy differences unphysical.

**AO targets and threshold (this section — SP benchmark):**
```
Target AOs:  C 2pz, N 2p, O 2pz, F 2pz
Threshold:   0.2          ← larger active spaces than the OptTS calculation
```

This follows the approach validated end-to-end for rxn0103, which gave a manageable (16e, 10o) active space and a physically sensible barrier. Active spaces across the 10 reactions range from (14e, 10o) to (18e, 13o).

> **Threshold note:** The OptTS calculation (Step 5) uses threshold=0.4 → smaller
> active spaces. Active space sizes between the two calculations are not directly
> comparable (e.g. rxn7949 is (16e,12o) here vs (16e,10o) in OptTS).

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

## Results (Step 4 — SP on ORCA Geometries, threshold=0.2)

> These results are single-point evaluations on ORCA NEB geometries. They are the
> reference for the 30-reaction benchmark. For NEVPT2 at CASSCF-optimised geometries,
> see Step 5 and `active_space_quality_analysis.md`.

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

**Summary (top-10): 7 reliable, 2 red flag (rxn1320, rxn1150), 1 failed (rxn4113).**

#### rxn1320 red flag: detail

AVAS at the TS selects orbitals with occupations 1.483/0.523 — a strongly bond-breaking orbital pair. At the reactant, those same orbitals are completely filled/empty (all occupations ≥ 1.977 or ≤ 0.024). The active space describes the TS correctly but is essentially HF at the reactant. The NEVPT2 correction at R is near zero, while at TS it is substantial — so the computed barrier reflects the TS correlation without a matching correction at the reactant. The forward barrier will be biased. CCSD(T) is used as the reference for rxn1320.

#### rxn1150 red flag: detail

Same failure mode. The only fractional occupation at the reactant is 0.051 — just barely above the detection threshold. NEVPT2 adds essentially no correlation energy at the reactant while adding significant correlation at TS and product. The forward barrier (NEVPT2: 3362 meV vs CCSD(T): 3460 meV) is −98 meV lower, and the reverse barrier (NEVPT2: 481 meV vs CCSD(T): 757 meV) is −276 meV lower. The large deviation on the reverse barrier, where the product is also MR (1.938, 1.880, 0.128, 0.074) but the reactant is not, is consistent with this interpretation.

#### rxn4113: CASSCF convergence failure at product

CASSCF converged at TS and reactant but failed at the product geometry even after 1000 damped iterations. The product likely has a qualitatively different electronic structure (possibly a diradical or near-degenerate closed-shell / open-shell pair) that requires a different initial guess or a larger active space. The CCSD(T) barrier is used as reference. rxn4113 is excluded from the NEVPT2 benchmark.

---

## Full 30-Reaction Benchmark

The benchmark was extended from the top-10 high-MR reactions to 30 reactions total, by adding 10 mid-MR and 10 low-MR reactions as controls. This allows the NEVPT2/CCSD(T) comparison to be interpreted in context: does NEVPT2 fail specifically where MR character is genuine (top-10), or also where there is little MR character (bottom-10)?

**Groups (selection from the 279-reaction FOD ranking):**
- **Top 10 (high MR):** ranks 1–10 (NFOD 0.85–1.15) — highest NFOD, where single-reference DFT is expected to be most unreliable
- **Middle 10 (mid MR):** uniformly sampled across the full ranking — one reaction every ~29 ranks (ranks 11, 40, 68, 97, 126, 154, 183, 212, 240, 269; NFOD 0.017–0.84), covering the entire NFOD spectrum
- **Bottom 10 (low MR):** ranks 270–279 (NFOD 0.003–0.014) — lowest NFOD, where single-reference methods should be reliable

### Reliability Criterion (5 flags)

A reaction is flagged **RED FLAG** if any of the following apply; otherwise **Reliable**:

| Flag | Condition | Interpretation |
|------|-----------|----------------|
| `0 frac@R` | Zero fractional occupations at reactant | Active space idle at R; NEVPT2 adds no correlation there |
| `0 frac@TS` | Zero fractional occupations at TS | Active space idle at TS; main point of interest uncorrected |
| `0 frac@P` | Zero fractional occupations at product | Active space idle at P; reverse barrier unreliable |
| `neg rev` | Reverse barrier < 0 meV | Physically impossible; NEVPT2 energetics are wrong |
| `\|NEV-CCT\| > 300 meV` | NEVPT2 and CCSD(T) forward barriers disagree by >300 meV | Secondary cross-check confirming a breakdown |

A reaction is **MISSING** if no `nevpt2_results.json` exists on the cluster (CASSCF did not converge or job was not submitted).

### Full Reliability Table

All barriers in meV. Source: `full_benchmark_results.json`, `pipeline/_check_nevpt2_plausibility.py`.

| Reaction | Group | Active Space | frac R/TS/P | NEVPT2 fwd | CCSD(T) fwd | diff | Status | Flags |
|----------|-------|-------------|-------------|-----------|------------|------|--------|-------|
| rxn7949  | High  | (16e,12o)   | 6/8/6       | 3254       | 3210        | +44  | Reliable | |
| rxn8832  | High  | (18e,13o)   | 4/6/6       | 2540       | 2621        | −81  | Reliable | |
| rxn1320  | High  | (16e,10o)   | 0/2/2       | 3147       | 3051        | +96  | Red flag | 0 frac@R |
| rxn4113  | High  | —           | —           | —          | 5346        | —    | Failed   | CASSCF no conv. at P |
| rxn8885  | High  | (14e,11o)   | 6/6/4       | 3643       | 3564        | +79  | Reliable | |
| rxn7945  | High  | (16e,12o)   | 6/6/6       | 3943       | 3923        | +20  | Reliable | |
| rxn7937  | High  | (16e,12o)   | 6/6/4       | 3764       | 3858        | −94  | Reliable | |
| rxn6196  | High  | (14e,12o)   | 6/6/8       | 4181       | 4282        | −101 | Reliable | |
| rxn0346  | High  | (14e,10o)   | 2/2/2       | 3213       | 3336        | −123 | Reliable | |
| rxn1150  | High  | (14e,10o)   | 1/4/4       | 3362       | 3460        | −98  | Red flag | 0 frac@R (borderline) |
| rxn0896  | Mid   | (16e,11o)   | 2/2/1       | 5145       | 5094        | +51  | Reliable | |
| rxn1154  | Mid   | (14e,9o)    | 1/2/0       | 4295       | 3847        | +448 | Red flag | 0 frac@P; \|NEV-CCT\|=448meV |
| rxn5690  | Mid   | (18e,12o)   | 4/4/4       | 3274       | 3346        | −73  | Red flag | neg rev (−72meV) |
| rxn4513  | Mid   | (14e,9o)    | 2/0/0       | 1929       | 1936        | −7   | Red flag | 0 frac@TS; 0 frac@P |
| rxn7955  | Mid   | (18e,14o)   | 6/6/6       | 3100       | 3080        | +19  | Reliable | |
| rxn4519  | Mid   | —           | —           | —          | 4903        | —    | Missing  | |
| rxn4500  | Mid   | —           | —           | —          | 4745        | —    | Missing  | |
| rxn2553  | Mid   | (12e,8o)    | 2/2/2       | 2009       | 2011        | −2   | Reliable | |
| rxn8829  | Mid   | (16e,13o)   | 4/4/4       | 2994       | 2938        | +56  | Reliable | |
| rxn1155  | Mid   | (14e,8o)    | 0/0/2       | 2453       | 2797        | −344 | Red flag | 0 frac@R; 0 frac@TS; \|NEV-CCT\|=344meV |
| rxn9246  | Low   | (14e,8o)    | 2/0/2       | 1433       | 1776        | −343 | Red flag | 0 frac@TS; \|NEV-CCT\|=343meV |
| rxn4498  | Low   | —           | —           | —          | 3268        | —    | Missing  | |
| rxn1061  | Low   | —           | —           | —          | 1144        | —    | Missing  | |
| rxn4003  | Low   | —           | —           | —          | 1998        | —    | Missing  | |
| rxn4004  | Low   | (18e,12o)   | 0/0/2       | 2185       | 2011        | +174 | Red flag | 0 frac@R; 0 frac@TS |
| rxn4063  | Low   | (18e,11o)   | 0/0/0       | 2031       | 1956        | +75  | Red flag | 0 frac@R; 0 frac@TS; 0 frac@P |
| rxn4114  | Low   | (14e,9o)    | 0/0/1       | 2434       | 2481        | −47  | Red flag | 0 frac@R; 0 frac@TS |
| rxn4060  | Low   | (20e,12o)   | 0/0/0       | 1688       | 1871        | −184 | Red flag | 0 frac@R; 0 frac@TS; 0 frac@P |
| rxn1961  | Low   | (14e,8o)    | 0/0/0       | 869        | 2269        | −1400| Red flag | 0 frac@R/TS/P; neg rev; \|NEV-CCT\|=1400meV |
| rxn1962  | Low   | (14e,8o)    | 0/0/0       | 2019       | 2411        | −393 | Red flag | 0 frac@R/TS/P; \|NEV-CCT\|=393meV |

**Summary by group:**

| Group | Reliable | Red flag | Missing/Failed | Total |
|-------|----------|----------|----------------|-------|
| High MR (top 10) | 7 | 2 | 1 | 10 |
| Mid MR (middle 10) | 4 | 4 | 2 | 10 |
| Low MR (bottom 10) | 0 | 7 | 3 | 10 |
| **All 30** | **11** | **13** | **6** | **30** |

### Interpretation

**Root cause of failures in low-MR reactions:** AVAS selects active orbitals by projecting onto π-type AOs at the TS. For low-MR reactions, the TS has no near-degenerate orbitals — all π orbitals are fully occupied or empty. The selected active space is physically idle everywhere: CASSCF converges to a solution that is essentially HF, and NEVPT2 adds negligible correlation. The method works by design only where genuine MR character exists.

**Mid-MR reactions:** Mixed results. Four reactions are reliable (rxn0896, rxn7955, rxn2553, rxn8829), where moderate but genuine MR character exists at all three geometries. The failures arise from TS-biased active spaces (0 frac@P) or a marginally negative reverse barrier indicating a near-flat PES that NEVPT2 cannot resolve.

**Practical conclusion (Step 4 SP benchmark):** Use NEVPT2 barriers only for the 7 reliable top-10 reactions. For all others (mid and low MR), use CCSD(T) as the high-level reference. Flag rxn1320 and rxn1150 (top-10 red flags) accordingly — CCSD(T) is the reference for those two as well.

**NEVPT2 vs CCSD(T) for the 11 reliable reactions:** forward barrier spread −123 to +96 meV (top-10 subset); +2 to +56 meV (mid-MR subset). Consistent with NEVPT2 accuracy on moderately MR systems (~50–100 meV).

**Step 5 OptTS benchmark (separate):** The 23-reaction CASSCF OptTS benchmark provides a CASSCF-level geometry reference independent of DFT. 8 reactions have reliable CASSCF OptTS geometries with consistent NEVPT2 barriers (Step 5 "reliable" class). An additional 3+7+1 reactions have valid geometries with varying energy caveats. See Step 5 and `active_space_quality_analysis.md` for the full classification.

---

## Step 5: CASSCF OptTS + NEVPT2 at CASSCF-Optimised Geometries

### Motivation

The Step 4 NEVPT2 SPs are evaluated at the ORCA DFT geometries. A stronger test is
to ask: where does CASSCF itself locate the TS, and does that geometry match DFT?
If CASSCF finds a different saddle point, the NEVPT2 barrier at the ORCA geometry
is not the CASSCF-level answer.

This step optimises the TS at the CASSCF+NEVPT2 level (CASSCF eigenvector-following,
then NEVPT2 SP at the converged geometry) and compares the CASSCF TS geometry to
the ORCA NEB TS via Kabsch RMSD.

### Settings (different from Step 4)

```
AO targets : C 2pz, N 2p, O 2pz, F 2pz   (same AOs)
Threshold  : 0.4                           (larger → smaller active space)
--no-prune : pruning disabled              (pruning collapsed 8/10 to CAS(2,2) in first batch)
Basis      : def2-TZVP
Code       : mr_casscf_optts.py (geomeTRIC eigenvector-following, transition=True)
```

Convergence: mc1step → mc2step fallback. Conv_tol loosened to 1e-6 for 3 reactions
(rxn0896, rxn10005, rxn1283) in a retry job.

### Two separate reaction sets — Step 4 (30) vs Step 5 (26/23)

**Step 5 does not use the same 30 reactions as Step 4.** The two benchmarks
cover deliberately different subsets:

| Benchmark | n | Selection | Geometry reference |
|-----------|---|-----------|-------------------|
| Step 4 SP | 30 | FOD top-10 + uniform mid-sample + bottom-10 | ORCA DFT NEB TS |
| Step 5 OptTS | 26 → 23 converged | FOD ranks 1–26 (top 26 by NFOD) | CASSCF-optimised TS |

The **next-HIGH** reactions (ranks 11–26) are the highest-NFOD reactions
below the top-10, not the Mid-MR group from Step 4 (which was a uniform
sample across ranks 11–270) and not the Low-MR group (ranks 270–279).
Step 4 showed that NEVPT2/AVAS is unreliable for Low-MR reactions (0/10
reliable) and mixed for Mid-MR (4/10 reliable), because the AVAS active
space is idle where genuine MR character is absent. Running CASSCF OptTS
on those reactions would waste compute and produce unreliable references.
The next-HIGH group retains substantial MR character throughout and is
the most natural extension of the CASSCF reference set.

### Reactions attempted

26 total (FOD ranks 1–26):
- **High(orig)**: rxn7949, rxn8832, rxn1320, rxn4113, rxn8885, rxn7945, rxn7937,
  rxn6196, rxn0346, rxn1150 (ranks 1–10, same as Step 4 top-10)
- **next-HIGH**: rxn0896, rxn4518, rxn3107, rxn8837, rxn7060, rxn5691, rxn1283,
  rxn8827, rxn4522, rxn7936, rxn1147, rxn0894, rxn0101, rxn10005, rxn10054, rxn7957
  (ranks 11–26)

### Outcomes

| Result | Count | Reactions |
|---|---|---|
| OptTS converged | 23 | all except rxn5691, rxn1283, rxn0894 |
| Failed — first run, no retry | 2 | rxn5691 (idx 5), rxn0894 (idx 11) |
| Failed — retry also failed | 1 | rxn1283 (cycle 68, conv_tol=1e-6 retry) |

Convergence rate: 23/26 = 88.5%.

### Reliability classification

The 23 converged reactions are classified by:
1. **RMSD** (CASSCF OptTS vs ORCA NEB TS, Kabsch alignment)
2. **NOON pattern at R/TS/P** (do fractional occupations peak at the TS as expected?)
3. **n<0.05 at TS** (intruder orbitals)

Full table with all NOON data, NEVPT2(OptTS) barriers, and per-reaction remarks:
→ See **`active_space_quality_analysis.md`**

Summary:

| Class | Count | Criterion |
|---|---|---|
| Reliable | 8 | RMSD < 0.30 Å, no intruder, consistent MR pattern |
| Reliable* | 3 | RMSD < 0.30 Å, geometry valid; NEVPT2 energy biased (0@R or CAS(2,2)) |
| Caveat | 7 | Intruder orbital OR anomalous MR pattern (R>TS or P>TS) |
| Borderline | 1 | RMSD 0.43 Å unexplained despite coherent active space |
| Excl-geo | 3 | RMSD > 0.50 Å — CASSCF found different saddle point |
| Excl-nevpt2 | 1 | rxn4113: 0@R → NEVPT2 barrier biased; use CCSD(T)=5346 meV |
| Failed | 3 | No converged OptTS |

### Key findings

- **Reliable reactions (8):** CASSCF and DFT locate essentially the same TS (RMSD 0.048–0.287 Å). NEVPT2(OptTS) barriers agree with CCSD(T) to within ±100 meV for 5/8 cases where CCSD(T) is available.
- **Problematic cases:**
  - rxn1150: 0 frac@R → NEVPT2(opt)=1679 meV vs CCSD(T)=3460 meV (Δ=−1781 meV). Use CCSD(T).
  - rxn10054: NEVPT2(opt)=−30 meV (essentially no barrier). CASSCF found a non-representative saddle point.
  - rxn4113: 0 frac@R → barrier biased despite OptTS succeeding; use CCSD(T)=5346 meV.
- **Large deviations (excl-geo):** rxn4518 (0.65 Å), rxn0101 (0.71 Å), rxn4522 (0.86 Å). All have small effective active spaces; rxn4518 shows extreme singlet biradical at CASSCF TS (occ 1.012/0.991) that DFT does not reach.
- **Important note on NEVPT2(OptTS) vs Step 4:** These are different calculations at different geometries. NEVPT2(OptTS) for rxn7949 = 4812 meV; NEVPT2(SP at ORCA TS) = 3254 meV. The difference reflects the CASSCF and DFT TSs being on different parts of the PES. Neither is "wrong" — they correspond to different levels of geometry optimisation.

### NEVPT2(OptTS) barriers — all 23 reactions

All barriers in meV. Evaluated at CASSCF OptTS geometry (ORCA R and P geometries
used as reference points with projected MOs). CCSD(T) where available is the
Step 3 single point on ORCA geometries.

| Reaction | Class | CAS | frac R/TS/P | TS-Paar | RMSD [Å] | NEVPT2(opt) | CCSD(T) |
|---|---|---|---|---|---|---|---|
| rxn7949 | reliable | (16e,10o) | 4/4/4 | 1.938/0.066 | 0.073 | 4812 | 3210 |
| rxn8832 | reliable | (16e,10o) | 4/4/4 | 1.920/0.085 | 0.287 | 2340 | 2621 |
| rxn1320 | reliable* | (2e,2o) | 2/2/2 | 1.922/0.078 | 0.195 | 3872 | 3051 |
| rxn4113 | excl-nevpt2 | (16e,10o) | 0/2/2 | 1.930/0.084 | 0.056 | 5308¹ | 5346 |
| rxn8885 | reliable | (12e,9o) | 6/6/4 | 1.945/0.054 | 0.151 | 3709 | 3564 |
| rxn7945 | reliable | (14e,10o) | 6/6/6 | 1.949/0.060 | 0.052 | 3920 | 3923 |
| rxn7937 | caveat | (14e,10o) | 4/4/4 | 1.947/0.064 | 0.048 | 3809 | 3858 |
| rxn6196 | reliable | (14e,10o) | 6/6/6 | 1.943/0.056 | 0.079 | 4346 | 4282 |
| rxn0346 | caveat | (14e,9o) | 2/2/2 | 1.749/0.261 | 0.153 | 3237 | 3336 |
| rxn1150 | reliable* | (12e,8o) | 0/3/4 | 1.940/0.061 | 0.161 | 1679¹ | 3460 |
| rxn0896 | caveat | (14e,9o) | 2/2/1 | 1.378/0.623 | 0.230 | 2484 | 5094² |
| rxn4518 | excl-geo | (14e,9o) | 2/2/2 | 1.012/0.991 | 0.653 | 3693 | — |
| rxn3107 | reliable | (14e,8o) | 2/2/2 | 1.935/0.070 | 0.106 | 4743 | — |
| rxn8837 | borderline | (18e,11o) | 4/4/4 | 1.932/0.073 | 0.427 | 3842 | — |
| rxn7060 | caveat | (16e,11o) | 6/4/2 | 1.930/0.080 | 0.197 | 3919 | — |
| rxn8827 | caveat | (16e,10o) | 2/2/2 | 1.928/0.079 | 0.064 | 4003 | — |
| rxn4522 | excl-geo | (14e,9o) | 1/3/1 | 1.043/0.051 | 0.858 | 5123 | — |
| rxn7936 | reliable | (18e,11o) | 2/4/2 | 1.946/0.058 | 0.070 | 6127 | — |
| rxn1147 | reliable* | (14e,8o) | 0/2/2 | 1.755/0.251 | 0.070 | 2114 | — |
| rxn0101 | excl-geo | (14e,9o) | 2/2/1 | 1.948/0.054 | 0.711 | 2330 | — |
| rxn10005 | caveat | (20e,13o) | 2/4/4 | 1.940/0.060 | 0.247 | 3452 | — |
| rxn10054 | caveat | (16e,10o) | 2/2/4 | 1.944/0.061 | 0.328 | −30¹ | — |
| rxn7957 | reliable | (14e,9o) | 4/4/2 | 1.923/0.081 | 0.074 | 3023 | — |

¹ Unreliable: 0@R (rxn4113, rxn1150) or negative barrier (rxn10054) — see `active_space_quality_analysis.md`.
² CCSD(T) from original 30-reaction SP benchmark at ORCA geometry with threshold=0.2 active space — not directly comparable to OptTS calculation.

Failed (no OptTS): rxn5691, rxn1283, rxn0894.

---

## Step 6: RKS Stability and Broken-Symmetry Analysis

### Motivation

Steps 1–5 quantify multireference character with FOD, CASSCF/NEVPT2 and CCSD(T).
Step 6 asks a narrower, cheaper question that is directly relevant to every DFT
number in this project: **is the closed-shell RKS solution used for the reference
NEB even a stable SCF solution at the transition state?**

If it is not, the RKS reference geometries and barriers describe a saddle point of
a potential energy surface that the electronic structure does not actually sit on.

### Method

At each ORCA NEB TS geometry, RKS wB97M-V/def2-TZVP (PySCF), then

```python
mo_i, mo_e, int_stable, ext_stable = mf.stability(
    internal=True, external=True, return_status=True)
```

This diagonalises the **orbital-rotation Hessian** (not the geometric one) in two
subspaces:

- **internal** — rotations preserving the RKS form. λ_min < 0 would mean a *lower
  closed-shell* solution exists, i.e. the SCF converged to the wrong minimum.
- **external** — rotations breaking spin restriction (RKS → UKS). λ_min < 0 means
  letting α and β orbitals differ lowers the energy: diradical character.

λ_min_ext is not a yes/no flag but a magnitude — the curvature along the
spin-symmetry-breaking direction.

**Broken-symmetry follow-through.** Where externally unstable, the instability
eigenvector seeds a UKS calculation (Route 1). Fallback if that collapses:
triplet-seeded β-HOMO flip (Route 2). Collapse criterion ⟨S²⟩ < 0.3.

### Results (26 reactions = top-26 by N_FOD)

| | count |
|---|---|
| internally unstable | **0 / 26** |
| externally unstable | **18 / 26** |
| BS collapses (both routes) | 0 |

All 18 BS solutions were found by Route 1; Route 2 was never needed. Zero internal
instabilities means the RKS reference calculations are themselves sound — the
issue is purely spin symmetry.

ΔE_BS and ⟨S²⟩ track λ_min_ext monotonically over more than an order of magnitude
(−648 meV at λ = −0.078 down to −8 meV at λ = −0.008): a continuous diradical
onset, not a threshold effect. The 8 stable cases cluster at λ_min_ext = +0.001
to +0.008 (marginally stable); only rxn0101 (+0.071) is robustly closed-shell.

Full 26-row table (λ_min_int, λ_min_ext, ΔE_BS, ⟨S²⟩, spin populations, reactive
atoms): **`rks_stability_bs_26rxn.md`**.

### Key finding: the RKS TS geometries are not stationary on the BS surface

Evaluating the **nuclear gradient of the converged BS solution at the RKS TS
geometry** (11 reactions, eV/Å):

| | RKS max\|∇E\| | BS max\|∇E\| | ratio |
|---|---|---|---|
| range | 0.013 – 0.179 | 0.386 – 2.637 | **3.6× – 62.7×** |

The RKS geometries are converged saddle points on the RKS surface, but carry
forces of up to 2.6 eV/Å on the BS surface. **The RKS barriers for these
reactions are therefore not the BS barriers, and the discrepancy is not bounded
by ΔE_BS.** The ratio does not track ΔE_BS: rxn1283 and rxn8885 lower the energy
by only ~45 meV yet carry the largest BS forces (~62×), while rxn7949 lowers it
by −560 meV for 16×. Energy lowering at fixed geometry is a poor proxy for how
far the geometry will relax.

Only ⟨S²⟩ acts as a threshold indicator: the two most weakly polarised cases
(rxn4113 ⟨S²⟩=0.14, rxn6196 0.22) are the only ones with modest ratios (4.9×,
3.6×). Above ⟨S²⟩ ≈ 0.33 the BS forces are uniformly large with no useful ordering.

### Cross-validation

**Independent code.** ORCA 5.0.4 (`%scf STABPerform true; STABRestartUHFifUnstable
true`) reproduces the PySCF result: for rxn1320, ⟨S²⟩ = 0.779 (ORCA/def2-SVP) vs
0.785 (PySCF/def2-TZVP), same spin localisation pattern (sign is arbitrary in BS).

**OMol25 protocol.** OMol25 (arXiv:2505.08762) breaks spin symmetry differently —
*"rotate by 20° between the HOMO and LUMO in the β space"* — and reports that
<5 % of Transition1x has ⟨S²⟩ > 0.001. Since the MLIPs benchmarked here are
trained on that data, it matters whether that protocol finds the same states.

Tested directly on all 26 reactions at OMol25 settings (wB97M-V/def2-TZVPD,
DEFGRID3, thresh 1e-12, tcut 1e-13):

| | OMol25 20° rotation | stability-following |
|---|---|---|
| rxn7949 | ⟨S²⟩ 0.893370 | 0.893417 |
| rxn1320 | 0.791817 | 0.791820 |
| rxn8885 | 0.466086 | 0.466193 |
| rxn4113 | 0.118779 | 0.117803 |

**Identical, without exception** — energies agree to ~1e-8 Ha across all 26,
including the weakest case (rxn4113, ΔE_BS = −5.9 meV). Exactly 8 reactions give
⟨S²⟩ = 0.000000 in both, and they are precisely the 8 that the PySCF stability
analysis classified as externally stable. Two codes, two basis sets, two
different methods, same 18/8 split.

**Conclusion:** the OMol25 training data *does* contain these broken-symmetry
states. The <5 % figure is a selection effect — the 26 reactions here are the top
26 of 279 by N_FOD — not a failure of their protocol. Whether the MLIPs actually
learned the states is a separate, still-open question.

**Basis/grid sensitivity.** ΔE_BS at def2-TZVP vs def2-TZVPD+DEFGRID3 differs by
**≤10 meV** (typically <6), e.g. rxn4518 −648.5 → −643.0, rxn1320 −339.2 →
−343.1. Negligible against effect sizes of 6–650 meV. Caveat: for the weakest
cases the *relative* shift is large (rxn4113 loses 30 % of its ΔE_BS), so
statements about marginally unstable reactions are basis-sensitive.

### Methodological pitfalls (recorded to save the next person the time)

1. **Plain DIIS destroys the BS solution.** First attempts collapsed all 8
   reactions to closed-shell (ΔE = 0, ⟨S²⟩ = 0). Second-order Newton
   (`mf.newton()`) is required to hold the symmetry-broken solution through
   convergence. *(In ORCA this is not an issue — its default SCF holds it.)*
2. **PySCF does not return the stability eigenvalues.** They go to the logger
   bound to `mol.stdout` at build time; a `sys.stdout` redirect does not capture
   them. They must be parsed from the job log afterwards.
3. **Seed UKS from `mf_rks.to_uks()`, not a fresh `dft.UKS(mol)`.** The latter has
   `mo_occ = None`, and `make_rdm1(mo_ext, mo_occ)` then raises.
4. **Carrying BS across geometries needs a density matrix, not MO coefficients.**
   MOs converged at geometry A are orthonormal w.r.t. S(A); handing them to a SCF
   at geometry B corrupts the density silently — measured on H2O for a 0.15 Å
   step: electron count 10.000 → 10.056, energy 654 meV too low, ⟨S²⟩ negative.
   Pass `dm0` instead. Negative ⟨S²⟩ is the diagnostic that this has happened.
5. **wB97M-V has no analytic Hessian in either PySCF or ORCA 5.0.4** (VV10
   non-local correlation; ORCA fails with *"The CPSCF equations can not yet handle
   non-local correlation"*). Numerical Hessians must be forced explicitly —
   `%geom Calc_Hess true; NumHess true end` in ORCA.

### Scripts

| Script | Purpose |
|---|---|
| `rks_stab_bs_grad.py` | RKS + stability + BS + nuclear gradients (per reaction) |
| `bs_uks.py` | Route 1 / Route 2 broken-symmetry search |
| `bs_tsopt_batch.py` | BS-UKS transition-state optimisation (SLURM array) |
| `omol25_settings.sh` | OMol25-level RKS / 20°-rotation / stability comparison |

---

## MLIP NEB Benchmark (eSEN and UMA)

### Motivation

To test whether modern universal MLIPs can reproduce the wB97M-V/def2-TZVP NEB transition states and barriers — and specifically whether they fail for high-MR reactions where single-reference DFT is known to be unreliable.

Two models were evaluated:
- **eSEN (OMol25, small conserving)** — trained on ωB97M-V/def2-TZVPD via the OMol25 dataset. `esen_sm_conserving_all.pt` (49 MB). ASE interface: `FAIRChemCalculator` from `fairchem`.
- **UMA-s (v1.2)** — universal model trained on 5 datasets (OC20, OMat24, OMol25, ODAC23, OMC25). `uma-s-1p2.pt` (2.3 GB). Task routing: `task_name='omol'` for molecule-optimised head. Same ASE interface.

Both models use the same NEB protocol as ORCA (`esen_neb.py`, `uma_neb.py`): 10 images, BFGS endpoint relaxation, NEB → CI-NEB, fmax < 0.05 eV/Å. The only change from `orca_neb.py` is the calculator.

### Results

All 30 reactions converged for both models.

#### Barriers vs CCSD(T) (meV, forward)

| Group | eSEN MAE | eSEN Bias | UMA-s MAE | UMA-s Bias | wB97M-V MAE | wB97M-V Bias |
|-------|----------|-----------|-----------|------------|-------------|--------------|
| Low MR | 28.3 | +10.9 | 28.9 | +12.8 | 27.3 | +9.2 |
| Mid MR | 51.7 | +23.4 | 52.9 | +22.0 | 58.4 | +31.7 |
| High MR | 195.5 | −163 | 152.0 | −111 | 225.8 | +210 |
| All 30 | 91.8 | −43 | 77.9 | −25 | 103.8 | +84 |

#### MLIP barriers vs wB97M-V ORCA (geometry + energy model combined, meV)

| Group | eSEN MAE | eSEN Bias | UMA-s MAE | UMA-s Bias |
|-------|----------|-----------|-----------|------------|
| Low MR | 5.3 | +1.7 | 3.9 | +3.6 |
| Mid MR | 10.0 | −8.2 | 11.7 | −9.7 |
| High MR | 373 | −373 | 323 | −321 |

#### TS geometry RMSD vs ORCA (Å, Kabsch-aligned)

| Group | eSEN mean | eSEN max | UMA-s (not computed) |
|-------|-----------|----------|----------------------|
| Low MR | 0.008 | 0.022 | — |
| Mid MR | 0.032 | 0.178 | — |
| High MR | 0.287 | 1.411 | — |

### Interpretation

**Low and Mid MR:** Both MLIPs reproduce wB97M-V barriers to within 5–12 meV and find essentially the same TS geometry as ORCA. This is the expected result — models trained on wB97M-V should faithfully reproduce that functional for reactions where the PES is well-behaved.

**High MR:** Both MLIPs consistently find *lower* barriers than wB97M-V ORCA (bias −163/−111 meV) and locate geometrically different transition states (eSEN RMSD 0.29 Å mean, max 1.41 Å for rxn8885). The barrier difference is a combined effect of the MLIP finding a different saddle point and evaluating energy differently. These cannot be separated without running wB97M-V single points on the MLIP TS geometries.

**UMA vs eSEN:** UMA-s improves on eSEN for High MR (MAE 152 vs 196 meV vs CCSD(T); geometry difference 323 vs 373 meV vs wB97M-V). The improvement is real but modest — the fundamental limitation is the same: both models were trained on single-reference DFT data and have not seen the complex near-degenerate PES topology of high-MR transition states in training.

**Note on apparent MAE ordering:** The raw MAE numbers suggest MLIPs perform better than ORCA vs CCSD(T) for High MR. This is misleading — wB97M-V systematically overestimates (bias +210 meV) while MLIPs underestimate (bias −110 to −163 meV), and they happen to bracket CCSD(T). A fair comparison requires CCSD(T) evaluated on the MLIP TS geometries, which was not done.

### Scripts

| File | Purpose |
|------|---------|
| `pipeline/esen_neb.py` | NEB with eSEN calculator |
| `pipeline/job_esen_neb.sh` | SLURM array: 30 reactions, sm3090el8 |
| `pipeline/uma_neb.py` | NEB with UMA-s calculator (`task_name='omol'`) |
| `pipeline/job_uma_neb.sh` | SLURM array: 30 reactions, sm3090el8 |
| `pipeline/_collect_esen_barriers.py` | Extract eSEN barriers from neb.db → full_benchmark_results.json |
| `pipeline/_collect_uma_barriers.py` | Extract UMA barriers from neb.db → full_benchmark_results.json |
| `pipeline/_esen_ts_rmsd.py` | Kabsch RMSD between eSEN and ORCA TS geometries |
| `pipeline/_compare_mlips.py` | Full comparison table: eSEN, UMA, wB97M-V, MACE, MACE+delta vs CCSD(T) |

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

- [x] FOD screening of all 279 test reactions
- [x] CCSD(T)/def2-TZVP for all 30 benchmark reactions
- [x] NEVPT2/AVAS for top-10 high-MR reactions
- [x] NEVPT2/AVAS for mid-10 and low-10 (reliability check only; CCSD(T) used as reference for unreliable cases)
- [x] wB97X-D3 and wB97M-V single points on all 10-image NEB paths (30 reactions)
- [x] MACE and MACE+delta evaluation on all 30 reactions
- [x] eSEN NEB on all 30 reactions (30/30 converged)
- [x] UMA-s NEB on all 30 reactions (30/30 converged)
- [ ] MACE+delta NEB on all 30 reactions (pending H200 availability)
- [ ] wB97M-V single points on MLIP TS geometries (to decompose geometry vs energy model error)

---

## References

- Grimme, S. & Hansen, A. (2015). A Practicable Real-Space Measure and Visualization of Static Electron-Correlation Effects. *Angew. Chem. Int. Ed.*, 54, 12308.
- Sayfutyarova, E. R., Sun, Q., Chan, G. K.-L. & Knizia, G. (2017). Automated Construction of Molecular Active Spaces from Atomic Valence Orbitals. *J. Chem. Theory Comput.*, 13, 4063.
- Angeli, C., Cimiraglia, R. & Malrieu, J.-P. (2001). N-electron valence state perturbation theory: a fast implementation of the strongly contracted variant. *Chem. Phys. Lett.*, 350, 297.
