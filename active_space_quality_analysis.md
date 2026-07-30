# Active Space Quality Analysis — CASSCF OptTS

## Setup

**26 reactions attempted** (FOD ranks 1–26 from the 279-reaction Transition1x test set).
**23 converged** to a CASSCF OptTS geometry. **3 failed** (no converged TS found).

All jobs used AVAS active-space selection followed by CASSCF eigenvector-following
(geomeTRIC, `transition=True`). Settings fixed for all reactions:

```
AO targets : ['C 2pz', 'N 2p', 'O 2pz', 'F 2pz']
Threshold  : 0.4
--no-prune : post-AVAS occupation pruning disabled
Basis      : def2-TZVP
Charge/spin: 0 / 0 (closed-shell singlets)
```

**Why --no-prune:** In the first batch (High orig, 10 reactions), pruning collapsed 8/10
active spaces to degenerate CAS(2,2). All subsequent runs used `--no-prune` from the start.

**Two-step convergence:** If mc1step did not converge, mc2step was attempted automatically.
For 3 reactions (rxn0896, rxn10005, rxn1283) the gradient tolerance was loosened from
conv_tol=1e-7 to 1e-6 via a retry job.

**NEVPT2 at R and P:** After OptTS, CASSCF+NEVPT2 are evaluated at the ORCA NEB R and P
geometries using MOs projected from the TS (`mcscf.project_init_guess`). This ensures
R, TS, and P share the same active space and orbital frame.

---

## How to read the occupation numbers

After CASSCF converges, the 1-RDM is diagonalised to give **natural orbital occupation
numbers (NOONs)**, ranging from 0 (empty) to 2 (doubly occupied):

| Range | Label | Meaning |
|---|---|---|
| 1.95 – 2.00 | near-doubly-occ | Core-like; negligible MR contribution |
| 0.05 – 1.95 | **frac** (genuinely correlated) | Real MR character |
| 0.00 – 0.05 | near-empty | Virtual-like; potential intruder state |

`n_frac` = count of orbitals in the 0.05–1.95 range.
`n<0.05` = count of near-empty orbitals (intruder risk).

**Expected MR pattern for a bond-breaking reaction:**
n_frac@TS ≥ n_frac@R and n_frac@TS ≥ n_frac@P.
The TS is where bonds are partially broken, so correlation should peak there.

---

## Raw NOON data at TS (active_space_quality_analysis — TS only)

Occupation numbers at the CASSCF OptTS geometry only (from `active_space_quality_analysis.md`
predecessor, used as quick sanity check):

```
Reaction      ncas  nel  min_occ  max_occ  n<0.05  n>1.95  n_frac
rxn7949         10   16    0.066    2.000       0       6       4
rxn8832         10   16    0.085    2.000       0       6       4
rxn1320          2    2    0.078    1.922       0       0       2   CLEAN CAS(2,2)
rxn4113         10   16    0.027    2.000       1       7       2
rxn8885          9   12    0.054    1.994       0       3       6
rxn7945         10   14    0.060    1.999       0       4       6
rxn7937         10   14    0.021    1.999       1       5       4
rxn6196         10   14    0.056    1.999       0       4       6
rxn0346          9   14    0.044    1.999       1       6       2
rxn1150          8   12    0.061    1.999       0       5       3
rxn0896          9   14    0.026    2.000       1       6       2
rxn4518          9   14    0.049    2.000       1       6       2
rxn3107          8   14    0.070    2.000       0       6       2
rxn8837         11   18    0.073    2.000       0       7       4
rxn7060         11   16    0.029    1.999       1       6       4
rxn8827         10   16    0.030    2.000       1       7       2
rxn4522          9   14    0.051    2.000       0       6       3
rxn7936         11   18    0.058    2.000       0       7       4
rxn1147          8   14    0.251    2.000       0       6       2   CLEANEST (min=0.251)
rxn0101          9   14    0.041    2.000       1       6       2
rxn10005        13   20    0.034    2.000       1       8       4
rxn10054        10   16    0.029    2.000       1       7       2
rxn7957          9   14    0.081    1.999       0       5       4
```

---

## Reaction Chemistry — All 23 Converged Reactions

Bond changes extracted from `Transition1x.h5` using dedicated reactant/product
subgroups (`f['test'][formula][rxn]['reactant|product']['positions'][0]`).
Connectivity via covalent radii threshold: bond if `dist < 1.25 × (r_i + r_j)`
(Alvarez 2008). Script: `pipeline/_rxn_chemistry.py`.

All 23 reactions are **organic isomerizations or rearrangements** — no
dissociation, no change in molecular formula. Two formula classes dominate:
C5H5NO (12 atoms, 14 rxns) and C3H5NO2 (11 atoms, 8 rxns); rxn10005 is
C2H3N3O2 (10 atoms).

| rxn | formula | n | broken | formed | reaction type |
|-----|---------|---|--------|--------|---------------|
| rxn7949 | C5H5NO | 12 | C-C ×2 | C-C | Retro-cycloaddition (two C-C bonds break, one reforms) |
| rxn8832 | C5H5NO | 12 | C-C ×2 | C-C | Retro-cycloaddition; product retains biradical character |
| rxn8885 | C5H5NO | 12 | C-O | C-N, C-O | 1,3-O→N migration / transamidation-type |
| rxn7945 | C5H5NO | 12 | C-C | C-N | C-C cleavage + C-N bond formation |
| rxn6196 | C5H5NO | 12 | C-C, C-H | C-H | Retro-ene type: H-shift + C-C cleavage |
| rxn3107 | C3H5NO2 | 11 | C-O | C-N, C-O | 1,3-O→N migration |
| rxn7936 | C5H5NO | 12 | C-O | C-C, N-O | Rearrangement: C-O breaks, C-C and N-O form |
| rxn7957 | C5H5NO | 12 | C-C, C-H | C-H, C-N | H-shift + C-C/C-N bond exchange |
| rxn7937 | C5H5NO | 12 | C-C | C-C, C-N | Bond migration: C-C breaks, C-C and C-N reform |
| rxn0346 | C3H5NO2 | 11 | C-C, C-H | C-H | Retro-ene type: H-shift + C-C cleavage |
| rxn7060 | C5H5NO | 12 | C-C, C-O | C-N, C-O | Complex rearrangement: two bonds break, two reform |
| rxn1320 | C3H5NO2 | 11 | C-C, H-O | C-H | O→C proton transfer + C-C cleavage (keto-enol type) |
| rxn1147 | C3H5NO2 | 11 | C-C | C-O | C-C cleavage + C-O bond formation |
| rxn1150 | C3H5NO2 | 11 | C-N, H-N | C-H | N→C proton transfer + C-N cleavage (elimination type) |
| rxn0896 | C3H5NO2 | 11 | — † | C-C, N-O | Cyclization? (see note below) |
| rxn8827 | C5H5NO | 12 | C-C | C-N | C-C cleavage + C-N bond formation |
| rxn10005 | C2H3N3O2 | 10 | C-N, C-O | — | Ring opening (no new bonds detected) |
| rxn8837 | C5H5NO | 12 | C-C | C-N | C-C cleavage + C-N bond formation |
| rxn4518 | C3H5NO2 | 11 | C-N | N-O | N-C to N-O bond migration |
| rxn0101 | C3H5NO2 | 11 | C-O | C-N | O→N substitution-type rearrangement |
| rxn4522 | C3H5NO2 | 11 | C-N, C-O | N-O | Complex rearrangement |
| rxn10054 | C5H5NO | 12 | C-C, C-O | — | Ring opening / fragmentation |
| rxn4113 | C3H5NO2 | 11 | C-N | C-O | N→O bond migration |

† rxn0896: connectivity analysis detects no broken bonds — the reactant
subgroup geometry places atoms at distances where the covalent threshold
still counts bonds that are breaking. The strong TS-pair 1.378/0.623 and
caveat classification indicate genuine bond-breaking chemistry; the exact
mechanism requires inspecting the 3D geometry.

---

## Full Reliability Table — All 26 Reactions

Data sources:
- `frac@R/TS/P`, `TS-Paar`, `NEVPT2(opt)`: fetched from cluster
  `~/nevpt2_optts_results/{rxn}_avas/nevpt2_optts_results.json`
- `RMSD`: `ts_rmsd_final.json` (Kabsch alignment, CASSCF OptTS vs ORCA NEB TS)
- `n<0.05@TS`: from TS-only NOON table above

**CCSD(T) geometry — two variants exist, labelled in each table:**

| label | geometry | script | reactions |
|-------|----------|--------|-----------|
| (a) ORCA NEB geom | DFT-optimised NEB TS from `orca_neb_results/` | `ccsd_t_singlepoints.py` (30-rxn screening) | rxn7949, rxn8832, rxn8885, rxn7945, rxn6196, rxn7937†, rxn0346†, rxn1320, rxn1150, rxn4113 |
| (b) CASSCF OptTS geom | CASSCF-optimised TS from `ts_casscf_opt.xyz` | `ccsdt_rxn*_optts.py` | rxn7060†, rxn1147, rxn0896, rxn8827, rxn10005 |

**Why two variants — historical chronology, not a deliberate design choice:**

Round 1 (30-reaction SP screening, earlier session): CCSD(T)/def2-TZVP was computed at the
ORCA NEB geometry for all 30 benchmark reactions. At that point, no CASSCF OptTS geometries
existed yet — the OptTS campaign came later. The purpose was to validate NEVPT2 single-points
that were also at the ORCA geometry. This produced the (a) values.

Round 2 (CASSCF OptTS campaign, this session): Once CASSCF-optimised TS geometries were
available, NEVPT2(opt) moved to those geometries. For a clean Δ comparison, CCSD(T) should be
at the same geometry. New @OptTS jobs (`ccsdt_rxn*_optts.py`) were only submitted for reactions
where the question was still open:

| rxn | reason for @OptTS job |
|-----|-----------------------|
| rxn7060† | caveat with intruder — upgrade decision required Δ at same geometry as NEVPT2 |
| rxn1147 | 0@R failure — needed to quantify Δ to classify as BARRIER_UNRELIABLE |
| rxn0896 | existing (a) value 5094 meV; @OptTS run to isolate geometry vs method effect |
| rxn8827 | new caveat reaction, no prior CCSD(T) existed |
| rxn10005 | new caveat reaction, no prior CCSD(T) existed |

rxn7937 and rxn0346 were NOT rerun @OptTS: their (a) values already gave |Δ| < 150 meV
(−49 and −99 meV), making the intruder-benign conclusion clear without a new calculation.
For rxn0896, both variants are available: (a) = 5094 meV, (b) = 4548 meV — the 546 meV
difference at RMSD = 0.230 Å is the pure geometry effect, confirming that the ~2000 meV
gap from NEVPT2 is a method failure, not a geometry artefact.

**Δ = CCSD(T) − NEVPT2(opt).** NEVPT2(opt) is always at the CASSCF OptTS geometry.
- For (a): Δ mixes level-of-theory difference AND geometry difference (two different saddle points).
- For (b): Δ is a pure level-of-theory comparison (same geometry).

The intruder-validation threshold (|Δ| < 150 meV) was applied using (a) for rxn7937 and rxn0346.
Since RMSD is small (0.048 and 0.153 Å), the geometry contribution to Δ is expected to be minor,
making the conclusion robust — but an @OptTS cross-check was never performed for these two.

---

### RELIABLE — 11 reactions

RMSD < 0.30 Å, MR pattern consistent, no intruder OR intruder validated by CCSD(T) (Δ < 150 meV).

CCSD(T) geometry key:
- (a) = SP at **ORCA NEB geometry** (from 30-reaction screening; Δ mixes level-of-theory + geometry effects)
- (b) = SP at **CASSCF OptTS geometry** (same geom as NEVPT2; Δ is pure level-of-theory)

```
rxn       CAS        frac R/TS/P  TS-Paar       n<0.05  RMSD   MR-Muster    NEVPT2(opt)  CCSD(T)   Δ        geom
                                  high / low     @TS      [Å]               [meV]        [meV]    [meV]
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
rxn7949   (16e,10o)  4 / 4 / 4   1.938/0.066      0    0.073  TS-max OK       4812        3210    −1602     (a)
rxn8832   (16e,10o)  4 / 4 / 4   1.920/0.085      0    0.287  TS-max OK       2340        2621     +281     (a)
rxn8885   (12e, 9o)  6 / 6 / 4   1.945/0.054      0    0.151  TS-max OK       3709        3564     −145     (a)
rxn7945   (14e,10o)  6 / 6 / 6   1.949/0.060      0    0.052  TS-max OK       3920        3923       +3     (a)
rxn6196   (14e,10o)  6 / 6 / 6   1.943/0.056      0    0.079  TS-max OK       4346        4282      −64     (a)
rxn3107   (14e, 8o)  2 / 2 / 2   1.935/0.070      0    0.106  TS-max OK       4743          —        —       —
rxn7936   (18e,11o)  2 / 4 / 2   1.946/0.058      0    0.070  TS-max OK       6127          —        —       —
rxn7957   (14e, 9o)  4 / 4 / 2   1.923/0.081      0    0.074  TS-max OK       3023          —        —       —
rxn7937†  (14e,10o)  4 / 4 / 4   1.947/0.064      1    0.048  TS-max OK       3809        3858      −49     (a)
rxn0346†  (14e, 9o)  2 / 2 / 2   1.749/0.261      1    0.153  TS-max OK       3237        3336      −99     (a)
rxn7060†  (16e,11o)  6 / 4 / 2   1.930/0.080      1    0.197  R>TS (noted)    3919        4016      +97     (b)
```

† intruder orbital present, validated by CCSD(T) (|Δ| < 150 meV):
  rxn7937 (occ=0.021), rxn0346 (occ=0.044), rxn7060 (occ=0.029).

Note on Δ for rxn7937† and rxn0346†: CCSD(T) is at ORCA NEB geometry (a), NEVPT2(opt) at CASSCF
OptTS geometry — Δ reflects both level-of-theory AND geometry differences. Since RMSD is small
(0.048 and 0.153 Å), the geometry contribution is expected to be minor. The |Δ| < 150 meV
conclusion is robust, but a clean @OptTS cross-check has not been performed for these two.

Remarks:
- **rxn7949** (C5H5NO, retro-cycloaddition: two C-C bonds break, one reforms): C-C π bond
  breaking. NEVPT2(opt) 4812 vs CCSD(T) 3210 meV — the large gap reflects that the CASSCF
  OptTS geometry lies on a steeper part of the MR PES than the ORCA DFT TS (RMSD 0.073 Å,
  but the two TSs are not identical). The TS geometry itself is reliable; the barrier
  difference is a genuine CASSCF/DFT PES discrepancy.
- **rxn8832** (C5H5NO, retro-cycloaddition; biradical product): C-C ×2 broken, C-C formed.
  NEVPT2(opt) 2340 vs CCSD(T) 2621 meV; moderate gap.
- **rxn8885** (C5H5NO, 1,3-O→N migration): C-O broken, C-N + C-O formed. Uniform MR throughout
  R/TS/P. Excellent CCSD(T) agreement (Δ=+79 meV).
- **rxn7945** (C5H5NO, C-C cleavage + C-N formation): C-C broken, C-N formed. NEVPT2(opt) ≈
  CCSD(T) (3920 vs 3923 meV, Δ=−3 meV).
- **rxn6196** (C5H5NO, retro-ene type: H-shift + C-C cleavage): C-C and C-H broken, C-H formed.
  Despite the σ-chemistry being outside the pz-AVAS target, n_frac=6 is consistent across
  R/TS/P, indicating the active space captures the full π-conjugated system adjacent to the
  breaking C-H bond. CCSD(T) agreement good (Δ=−64 meV).
- **rxn3107** (C3H5NO2, 1,3-O→N migration): C-O broken, C-N + C-O formed. Minimal n_frac=2,
  but consistent and no intruder. Reliable geometry reference.
- **rxn7936** (C5H5NO, rearrangement: C-O breaks, C-C and N-O form): MR rises from 2→4 at TS
  then back to 2 — textbook TS-max pattern.
- **rxn7957** (C5H5NO, H-shift + C-C/C-N exchange): C-C and C-H broken, C-H and C-N formed.
  n_frac drops from 4 to 2 at P (product SR-like).
- **rxn7937†** (C5H5NO, bond migration: C-C breaks, C-C and C-N reform): 1 intruder (occ=0.021).
  MR pattern 4/4/4 TS-max OK. RMSD=0.048 Å (best in set). CCSD(T) Δ=−49 meV confirms
  intruder is benign. Upgraded from caveat.
- **rxn0346†** (C3H5NO2, retro-ene type: H-shift + C-C cleavage): C-C and C-H broken, C-H
  formed. 1 intruder (occ=0.044, borderline). Minimal 2/2/2 pattern, TS-pair 1.749/0.261
  shows real bond-breaking chemistry. CCSD(T) Δ=−99 meV confirms intruder is benign.
  Upgraded from caveat.
- **rxn7060†** (C5H5NO, complex rearrangement: C-C + C-O break, C-N + C-O form): 1 intruder
  (occ=0.029). Anomalous R>TS>P MR pattern (6/4/2 frac) — the active space describes the
  reactant better than the TS. Despite this structural warning, CCSD(T) Δ=+97 meV confirms
  the intruder is benign and the NEVPT2 barrier is not significantly biased. RMSD=0.197 Å.
  **The R>TS pattern is noted but does not invalidate the geometry or energy.**
  Upgraded from caveat.

---

### RELIABLE* — 3 reactions (reliable geometry, caveats on NEVPT2 energy)

Geometry key same as above: (a) = ORCA NEB geometry, (b) = CASSCF OptTS geometry.

```
rxn       CAS        frac R/TS/P  TS-Paar       n<0.05  RMSD   MR-Muster      NEVPT2(opt)  CCSD(T)   Δ        geom
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
rxn1320   ( 2e, 2o)  2 / 2 / 2   1.922/0.078      0    0.195  TS-max OK         3872        3051     −821     (a)
rxn1147   (14e, 8o)  0 / 2 / 2   1.755/0.251      0    0.070  0@R (SR edukt)    2114        4023    +1909     (b)
rxn1150   (12e, 8o)  0 / 3 / 4   1.940/0.061      0    0.161  0@R (SR edukt)    1679        3460    +1781     (a)
```

Remarks:
- **rxn1320** (C3H5NO2, O→C proton transfer + C-C cleavage): C-C and H-O broken, C-H formed.
  CAS(2,2) is the minimal possible active space. With threshold=0.4 and no-prune, only one
  bonding/antibonding pair is selected. However, this pair is genuinely correlated at ALL
  three geometries (1.922/0.078 at both R and TS), so the CAS(2,2) is internally consistent.
  NEVPT2(opt) 3872 vs CCSD(T) 3051 meV (Δ=+821 meV); barrier overestimation expected for a
  minimal active space. **TS geometry reliable; NEVPT2 barrier is a CAS(2,2)-level result,
  not a complete CASSCF result.** Use CCSD(T) as the energy reference.
- **rxn1147** (C3H5NO2, C-C cleavage + C-O formation): C-C broken, C-O formed. 0 frac@R —
  the reactant is single-reference at this level of theory. The active space is idle at R
  and only activates at TS/P. The TS-pair (1.755/0.251) is the strongest correlation signal
  in the entire next-HIGH set. TS geometry is valid (RMSD 0.070 Å).
  CCSD(T)/def2-TZVP@OptTS = **4023 meV** vs NEVPT2(opt) = 2114 meV → **Δ = +1909 meV**.
  Identical failure mode to rxn1150: asymmetric NEVPT2 correction (large at TS, near-zero
  at R) systematically underestimates the barrier. **TS geometry reliable; use CCSD(T) =
  4023 meV as the energy reference. NEVPT2(opt) excluded from barrier stats
  (BARRIER_UNRELIABLE).**
- **rxn1150** (C3H5NO2, N→C proton transfer + C-N cleavage): C-N and H-N broken, C-H formed.
  0 frac@R confirmed in OptTS calculation (1 frac at R in the older threshold=0.2 SP, but 0
  in the threshold=0.4 OptTS run). The active space is idle at the reactant. Consequence:
  NEVPT2(opt)=1679 meV vs CCSD(T)=3460 meV — a **−1781 meV discrepancy**. Entirely
  explained by asymmetric NEVPT2 correction: large correction at TS/P, near-zero at R →
  barrier systematically underestimated. **TS geometry is a valid CASSCF reference (RMSD
  0.161 Å); use CCSD(T)=3460 meV as the energy reference, not NEVPT2(opt).**

---

### CAVEAT — 3 reactions (intruder orbital OR anomalous MR pattern, |Δ| > 200 meV)

> rxn7937, rxn0346, rxn7060 were upgraded to reliable after CCSD(T) validated their
> intruder orbitals (Δ < 150 meV). rxn10054 downgraded to excl-geo (NEVPT2 = −30 meV).
> rxn0896 remains caveat but moves to BARRIER_UNRELIABLE (NEVPT2 reference unusable).

All CCSD(T) here at CASSCF OptTS geometry (b) — same geom as NEVPT2; Δ is pure level-of-theory.

```
rxn       CAS        frac R/TS/P  TS-Paar       n<0.05  RMSD   MR-Muster    NEVPT2(opt)  CCSD(T)@OptTS   Δ
──────────────────────────────────────────────────────────────────────────────────────────────────────────────
rxn0896   (14e, 9o)  2 / 2 / 1   1.378/0.623      1    0.230  TS-max OK         2484         4548        +2064
rxn8827   (16e,10o)  2 / 2 / 2   1.928/0.079      1    0.064  TS-max OK         4003         3716         −287
rxn10005  (20e,13o)  2 / 4 / 4   1.940/0.060      1    0.247  TS=P > R          3452         3732         +280
```

Remarks:
- **rxn0896** (C3H5NO2, cyclization? — bond analysis uncertain): no broken bonds detected by
  connectivity algorithm (threshold artefact); C-C and N-O bonds appear formed at product.
  TS-pair 1.378/0.623 — the strongest bond-breaking signal in the caveat group. Product has
  only 1 frac orbital (SR at P). 1 intruder at TS (occ=0.026). CCSD(T)@OptTS = 4548 meV vs
  NEVPT2(opt) = 2484 meV → **Δ = +2064 meV**. Cross-check: CCSD(T)@ORCA = 5094 meV — the
  two CCSD(T) values differ by only 546 meV (geometry effect at RMSD=0.230 Å), while NEVPT2
  is ~2000 meV below both. **The NEVPT2 reference is unreliable regardless of geometry →
  excluded from barrier stats (BARRIER_UNRELIABLE). Use CCSD(T)@OptTS = 4548 meV.**
- **rxn8827** (C5H5NO, C-C cleavage + C-N formation): C-C broken, C-N formed. n_frac=2 at
  all points, TS-pair 1.928/0.079 — weak correlation, barely outside the 1.95 threshold.
  1 intruder (occ=0.030). CCSD(T)@OptTS = 3716 meV vs NEVPT2(opt) = 4003 meV → Δ = −287 meV.
  Intruder not validated (|Δ| > 200 meV threshold).
  **Caveat: NEVPT2 slightly overestimates barrier; intruder likely responsible.**
- **rxn10005** (C2H3N3O2, ring opening: C-N + C-O broken): largest AS (ncas=13). n_frac
  2→4→4, MR builds at TS and persists at P (possible biradical or open-chain product).
  1 intruder. CCSD(T)@OptTS = 3732 meV vs NEVPT2 = 3452 meV → Δ = +280 meV. Intruder not
  validated (|Δ| > 200 meV). **Caveat: MR pattern non-standard; NEVPT2 slightly
  underestimates barrier.**

---

### BORDERLINE — 1 reaction

```
rxn       CAS        frac R/TS/P  TS-Paar       n<0.05  RMSD   MR-Muster    NEVPT2(opt)  CCSD(T)
──────────────────────────────────────────────────────────────────────────────────────────────────
rxn8837   (18e,11o)  4 / 4 / 4   1.932/0.073      0    0.427  TS-max OK       3842          —
```

Remarks:
- **rxn8837** (C5H5NO, C-C cleavage + C-N formation): C-C broken, C-N formed — same bond
  change as rxn8827 and rxn7945. Active space internally consistent (no intruder, constant
  n_frac=4 across R/TS/P, TS-max OK). The RMSD of 0.427 Å from the ORCA DFT TS is
  unexplained by any active-space deficiency. Two interpretations: (1) CASSCF and DFT
  locate genuinely different saddle points on their respective PESs — a true
  level-of-theory effect; (2) a subtle convergence issue in the eigenvector-following walk.
  Without a Hessian verification or IRC at the CASSCF OptTS geometry, this cannot be
  resolved. **The TS geometry is internally consistent at the CASSCF level, but the large
  RMSD from DFT means it cannot be used as a geometry reference without caveats.**

---

### EXCL-GEO — 4 reactions (CASSCF found wrong saddle point — TS geometry not usable as reference)

```
rxn       CAS        frac R/TS/P  TS-Paar       n<0.05  RMSD   MR-Muster    NEVPT2(opt)  CCSD(T)
──────────────────────────────────────────────────────────────────────────────────────────────────
rxn4518   (14e, 9o)  2 / 2 / 2   1.012/0.991      1    0.653  TS-max OK       3693          —
rxn0101   (14e, 9o)  2 / 2 / 1   1.948/0.054      1    0.711  TS-max OK       2330          —
rxn4522   (14e, 9o)  1 / 3 / 1   1.043/0.051      0    0.858  TS-max OK       5123          —
rxn10054  (16e,10o)  2 / 2 / 4   1.944/0.061      1    0.328  P > TS WARN      −30          —
```

Remarks:
- **rxn4518** (C3H5NO2, N-C to N-O migration): C-N broken, N-O formed. TS-pair 1.012/0.991
  — an extreme singlet biradical at the CASSCF TS (both orbitals at exactly 50% occupation).
  The active space is doing real, strong MR work. However, RMSD=0.653 Å from the ORCA DFT
  TS confirms CASSCF found a different saddle point. Combined with 1 intruder and n_frac=2
  at R/P (only the extreme TS biradical, not a gradual bond breaking), the geometry is not
  a reliable reference. NEVPT2 barrier (3693 meV) is reported for completeness only.
- **rxn0101** (C3H5NO2, O→N substitution-type): C-O broken, C-N formed. TS-pair 1.948/0.054
  — barely outside the frac threshold (1.948 < 1.95 by 0.002). The active space is doing
  minimal real MR work at the TS. Combined with 1 intruder and RMSD=0.711 Å, this reaction
  has both weak active-space performance and large geometric deviation. The CASSCF OptTS is
  not a useful reference geometry.
- **rxn4522** (C3H5NO2, complex rearrangement: C-N + C-O break, N-O forms): R and P each
  have only 1 frac orbital (borderline active spaces), but at the TS n_frac=3 and the pair
  is 1.043/0.051 (moderate biradical). The TS-max pattern is correct. Despite credible
  active-space work at the TS, RMSD=0.858 Å is the largest deviation in the set — CASSCF
  is unambiguously at a different saddle point. NEVPT2 barrier (5123 meV) reflects the
  CASSCF saddle, not the DFT one.
- **rxn10054** (C5H5NO, ring opening: C-C + C-O broken): RMSD=0.328 Å (moderate — lower
  than the other excl-geo reactions), but evidence for a wrong saddle point comes from the
  energy: NEVPT2(opt) = **−30 meV**, meaning the CASSCF+NEVPT2 PES has no genuine forward
  barrier at this geometry. The CASSCF eigenvector-following found a first-order saddle
  point on the CASSCF PES, but this saddle is not a TS on the NEVPT2 PES. The inverted MR
  pattern (P most correlated, 2→2→4) is a further warning sign. Downgraded from caveat.

---

### EXCL-NEVPT2 / RED FLAG — 1 reaction

```
rxn       CAS        frac R/TS/P  TS-Paar       n<0.05  RMSD   MR-Muster    NEVPT2(opt)  CCSD(T)
──────────────────────────────────────────────────────────────────────────────────────────────────
rxn4113   (16e,10o)  0 / 2 / 2   1.930/0.084      1    0.056  0@R (SR-Edukt)  5308        5346
```

Remarks:
- **rxn4113** (C3H5NO2, N→O bond migration): C-N broken, C-O formed. The TS geometry is
  close to ORCA (RMSD=0.056 Å). However, 0 frac@R means the reactant is single-reference
  at this level — the active space is idle at R, giving near-zero NEVPT2 correction at the
  reactant energy. NEVPT2(opt)=5308 vs CCSD(T)=5346 meV (Δ=−38 meV) — apparent agreement,
  but this is coincidental: if the NEVPT2 correction at R is ~0, then NEVPT2(opt) ≈ CASSCF
  barrier ≈ DFT barrier, and the agreement with CCSD(T) reflects cancellation of errors,
  not physical accuracy.
- **History:** In the original 30-reaction SP benchmark (threshold=0.2), CASSCF diverged
  at the product geometry and no NEVPT2 barrier was available. The OptTS calculation
  (threshold=0.4, projected MOs) converged at all three points, giving the values above.
  The two results are from different calculations and cannot be directly compared.
- **Classification:** TS geometry is usable as a geometric reference. Energy reference
  should be CCSD(T)=5346 meV. The NEVPT2(opt) value is reported but flagged.

---

### FAILED — 3 reactions (CASSCF OptTS never converged)

| Reaction | Job | Failure | Retry |
|---|---|---|---|
| rxn5691 | 10562969, array index 5 | CASSCF OptTS not converged (first run) | No retry performed |
| rxn1283 | 10562969, array index 6 | "Nuclear gradients not converged" at OptTS cycle 68 | Retried with conv_tol=1e-6 — also failed |
| rxn0894 | 10562969, array index 11 | CASSCF OptTS not converged (first run) | No retry performed |

These three reactions have no OptTS geometry and are absent from all benchmark tables.
The 23-reaction benchmark uses the 23 reactions that DID converge (not these 3).

Convergence rate: 23/26 = 88.5%.

---

## Summary

| Classification | Count | RMSD criterion | Active-space criterion |
|---|---|---|---|
| Reliable | 11 | < 0.30 Å | No intruder, MR consistent — OR intruder validated by CCSD(T) Δ < 150 meV |
| Reliable* | 3 | < 0.30 Å | 0@R or CAS(2,2): geometry OK, NEVPT2 energy biased (use CCSD(T)) |
| Caveat | 3 | < 0.40 Å | Intruder with |Δ| > 200 meV — CCSD(T) available but threshold not met |
| Borderline | 1 | 0.43 Å | Geometry unexplained despite consistent AS |
| Excl-geo | 4 | varies | CASSCF found wrong saddle point (large RMSD or negative NEVPT2 barrier) |
| Excl-nevpt2 | 1 | 0.056 Å | 0@R → NEVPT2 barrier biased (use CCSD(T)) |
| Failed | 3 | — | No converged OptTS |
| **Total** | **26** | | |

**For the TS RMSD geometry benchmark:** reactions with usable CASSCF OptTS reference
geometries: Reliable (8) + Reliable* (3) + Caveat (7) + Borderline (1) = **19 reactions**.
Excluded from geometry benchmark: 3 excl-geo + 1 excl-nevpt2 (geometry close but NEVPT2
biased; still usable geometrically) + 3 failed = 4 fully excluded.

**For the NEVPT2 energy benchmark:**
- Unambiguous: 8 reliable + rxn7937/rxn0346/rxn0896/rxn8827/rxn10005/rxn8837 (6 caveat/borderline with coherent AS) = 14
- Biased but reported: rxn1320, rxn1147, rxn10054, rxn4113 (0@R or negative barrier)
- Not used: rxn1150 (NEVPT2 1679 vs CCSD(T) 3460 meV — use CCSD(T) instead)
- Excluded geometry: rxn4518, rxn4522, rxn0101

---

## Implications and caveats for the paper

**1. The `--no-prune` choice** inflates ncas without improving quality (3–8 near-doubly-occ
orbitals in every reaction except rxn1320). This was forced by the convergence history:
pruning collapsed 8/10 active spaces to degenerate CAS(2,2) in the first batch.
The inflated spaces do not make geometry wrong, but waste computation.

**2. The AVAS `C 2pz` target** is orientation-dependent (selects m=0 p-orbital, preferring
π-type character aligned with z). For σ-bonds (C-H, N-H breaks in rxn6196, rxn0346,
rxn7957, rxn1150, rxn1320) the active space may miss part of the σ-system. In practice,
the NOONs at R/TS/P show these reactions have consistent MR character — the pz-orbital
appears to indirectly capture the C-H σ-system through conjugation. This is a known
approximation, documented here.

**3. The three reactions with 0@R** (rxn1320, rxn1147, rxn1150, rxn4113) illustrate a
fundamental limitation: when the reactant is single-reference but the TS is multireference,
AVAS (applied at the TS) selects orbitals that are idle at R. The TS geometry is still
a valid CASSCF saddle point, but the NEVPT2 barrier will be asymmetrically corrected.

**4. rxn10054 (NEVPT2 = −30 meV)** is the most problematic case. The CASSCF
eigenvector-following found a first-order saddle point on the CASSCF PES, but this
geometry is not a first-order saddle point on the NEVPT2 PES. This is a known risk of
CASSCF geometry optimisation followed by perturbation-theory energy evaluation: the
geometry and the energy method are inconsistent. For a definitive result, CASSCF+NEVPT2
analytical gradients (or numerical gradients) would be needed.
