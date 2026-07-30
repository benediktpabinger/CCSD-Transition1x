# RKS stability + broken-symmetry analysis — 26 MR benchmark reactions

Method: wB97M-V / def2-TZVP, PySCF, grids level 3, conv_tol 1e-10.
Geometries: ORCA NEB transition states (RKS).
Stability: `mf.stability(internal=True, external=True)`.
Broken symmetry: Route 1 = follow external instability eigenvector into UKS,
reconverged with second-order Newton (`mf_u.newton()`); Route 2 (triplet-seeded
beta-HOMO flip) held in reserve.

## Combined table (sorted by lambda_min_ext ascending)

| # | rxn | E_RKS (Ha) | int | λ_min_int | ext | λ_min_ext | ΔE_BS (meV) | ⟨S²⟩ | top-2 spin atoms | reactive atoms |
|---|-----|-----------|-----|-----------|-----|-----------|------------|------|------------------|----------------|
| 1 | rxn4518 | -322.330046 | ✓ | 0.41802 | ✗ | -0.07780 | -648.5 | 0.842 | N0=-0.905, C1=+0.762 | — |
| 2 | rxn7949 | -323.324256 | ✓ | 0.35498 | ✗ | -0.06315 | -559.6 | 0.893 | C3=-0.809, C2=+0.704 | C3-C5, C4-C5 |
| 3 | rxn8832 | -323.305679 | ✓ | 0.36977 | ✗ | -0.04926 | -428.0 | 0.870 | C4=+0.771, N0=-0.542 | — |
| 4 | rxn1320 | -322.377458 | ✓ | 0.43895 | ✗ | -0.04801 | -339.2 | 0.785 | C2=-0.751, C1=+0.520 | C2-H6, O0-H6 |
| 5 | rxn8837 | -323.268234 | ✓ | 0.42489 | ✗ | -0.04401 | -293.9 | 0.741 | C4=-0.701, C1=+0.394 | — |
| 6 | rxn0894 | -322.360824 | ✓ | 0.32772 | ✗ | -0.04014 | -190.2 | 0.580 | O5=+0.573, C0=-0.546 | — |
| 7 | rxn4522 | -322.320703 | ✓ | 0.40304 | ✗ | -0.03247 | -184.0 | 0.662 | N0=+0.597, C1=-0.595 | — |
| 8 | rxn5691 | -323.290407 | ✓ | 0.29454 | ✗ | -0.02902 | -155.9 | 0.629 | N6=-0.683, C4=+0.535 | — |
| 9 | rxn0346 | -322.306380 | ✓ | 0.38304 | ✗ | -0.02719 | -147.6 | 0.628 | N1=+0.538, C5=-0.485 | C5-H10, C2-C5 |
| 10 | rxn1147 | -322.373266 | ✓ | 0.29804 | ✗ | -0.02450 | -105.2 | 0.534 | C2=-0.597, C1=+0.265 | C1-C2, C1-O5 |
| 11 | rxn7957 | -323.324392 | ✓ | 0.35181 | ✗ | -0.02396 | -99.8 | 0.513 | N6=-0.622, C1=+0.295 | C1-H7, C5-H7 |
| 12 | rxn1283 | -322.327929 | ✓ | 0.36417 | ✗ | -0.01389 | -44.5 | 0.419 | C4=-0.515, O5=+0.237 | C4-O5, O2-O5 |
| 13 | rxn3107 | -322.364314 | ✓ | 0.28275 | ✗ | -0.01255 | -38.8 | 0.409 | N5=+0.483, C2=-0.351 | C2-O3, C2-N5 |
| 14 | rxn8885 | -323.295846 | ✓ | 0.24671 | ✗ | -0.01109 | -42.8 | 0.507 | N6=+0.539, C1=-0.319 | C1-O2, C1-N6 |
| 15 | rxn8827 | -323.281599 | ✓ | 0.38364 | ✗ | -0.01096 | -27.5 | 0.338 | C4=-0.377, C5=+0.255 | N0-C5, C4-C5 |
| 16 | rxn4113 | -322.366974 | ✓ | 0.26735 | ✗ | -0.00846 | -8.4 | 0.140 | N2=+0.253, O0=-0.193 | O0-C3, N2-C3 |
| 17 | rxn7060 | -323.255558 | ✓ | 0.36566 | ✗ | -0.00790 | -22.1 | 0.374 | N6=+0.382, C1=-0.359 | O0-C1, O0-C5 |
| 18 | rxn6196 | -323.272867 | ✓ | 0.29184 | ✗ | -0.00687 | -10.7 | 0.216 | N6=-0.265, C2=+0.134 | C2-C5, C2-H10 |
| 19 | rxn10054 | -323.355865 | ✓ | 0.31171 | ✓ | +0.00131 | — | — | — | C2-O3, C4-C6 |
| 20 | rxn10005 | -392.617305 | ✓ | 0.31366 | ✓ | +0.00335 | — | — | — | C4-N6, O3-C4 |
| 21 | rxn0896 | -322.350208 | ✓ | 0.22363 | ✓ | +0.00347 | — | — | — | — |
| 22 | rxn7945 | -323.326282 | ✓ | 0.22092 | ✓ | +0.00434 | — | — | — | C2-N6, C2-C4 |
| 23 | rxn1150 | -322.395015 | ✓ | 0.24616 | ✓ | +0.00498 | — | — | — | N3-H10, C2-N3 |
| 24 | rxn7937 | -323.328900 | ✓ | 0.23229 | ✓ | +0.00733 | — | — | — | C3-N6, C3-C4 |
| 25 | rxn7936 | -323.256002 | ✓ | 0.24459 | ✓ | +0.00814 | — | — | — | O0-N6, O0-C1 |
| 26 | rxn0101 | -322.421022 | ✓ | 0.32713 | ✓ | +0.07143 | — | — | — | C4-O5, N0-C4 |

18 of 26 externally unstable; 0 internally unstable; 0 BS collapses (all Route 1,
Route 2 never needed). ΔE_BS and ⟨S²⟩ both track λ_min_ext monotonically across
more than an order of magnitude (-648 meV at λ=-0.078 down to -8 meV at λ=-0.008):
a continuous diradical onset, not a threshold effect. The eight stable cases cluster
at λ_min_ext = +0.001 to +0.008 (marginally stable); only rxn0101 (+0.071) is
robustly closed-shell.

## BS gradient check (eV/Å, both solutions at the same RKS TS geometry)

| rxn | ΔE_BS (meV) | ⟨S²⟩ | RKS max\|∇E\| | RKS RMS\|∇E\| | BS max\|∇E\| | BS RMS\|∇E\| | BS/RKS max |
|-----|------------|------|--------------|--------------|-------------|-------------|-----------|
| rxn7949 | -559.6 | 0.893 | 0.1049 | 0.0305 | 1.6860 | 0.5642 | 16.1x |
| rxn0346 | -147.6 | 0.628 | 0.0519 | 0.0206 | 2.6126 | 0.7861 | 50.4x |
| rxn1147 | -105.2 | 0.534 | 0.0651 | 0.0229 | 1.8398 | 0.5793 | 28.2x |
| rxn7957 | -99.8 | 0.513 | 0.0265 | 0.0134 | 0.9010 | 0.3725 | 34.0x |
| rxn1283 | -44.5 | 0.419 | 0.0380 | 0.0165 | 2.3861 | 0.8036 | 62.7x |
| rxn8885 | -42.8 | 0.507 | 0.0423 | 0.0185 | 2.6373 | 0.6142 | 62.4x |
| rxn3107 | -38.8 | 0.409 | 0.0629 | 0.0199 | 1.6457 | 0.4754 | 26.1x |
| rxn8827 | -27.5 | 0.338 | 0.0263 | 0.0129 | 1.1278 | 0.3391 | 42.9x |
| rxn7060 | -22.1 | 0.374 | 0.0335 | 0.0095 | 1.7659 | 0.5817 | 52.8x |
| rxn6196 | -10.7 | 0.216 | 0.1793 | 0.0487 | 0.6385 | 0.1567 | 3.6x |
| rxn4113 | -8.4 | 0.140 | 0.0786 | 0.0218 | 0.3859 | 0.1113 | 4.9x |
| rxn10054 | — (stable) | — | 0.0135 | 0.0073 | — | — | — |
| rxn10005 | — (stable) | — | 0.0182 | 0.0083 | — | — | — |
| rxn1150 | — (stable) | — | 0.1746 | 0.0400 | — | — | — |
| rxn7937 | — (stable) | — | 0.0394 | 0.0144 | — | — | — |
| rxn7936 | — (stable) | — | 0.0273 | 0.0127 | — | — | — |
| rxn0101 | — (stable) | — | 0.0584 | 0.0148 | — | — | — |

Key result: the RKS TS geometries are converged saddle points on the RKS surface
(max|∇E| = 0.01-0.18 eV/Å), but on the BS surface the same geometries carry forces
of 0.6-2.6 eV/Å, i.e. 4x to 50x larger. The RKS TS geometries are therefore NOT
stationary points of the broken-symmetry surface, so the RKS barriers for these
reactions are not the BS barriers, and the discrepancy is not bounded by ΔE_BS.

The ratio does not track ΔE_BS. rxn1283 and rxn8885 lower the energy by only about
-45 meV yet carry the largest BS forces in the set (2.39 and 2.64 eV/Å, 62x), while
rxn7949 lowers it by -560 meV for a 16x ratio. Energy lowering at fixed geometry is
a poor proxy for how far the BS geometry will relax.

The only pattern that does hold: the two most weakly spin-polarized cases
(rxn4113 S^2=0.140 and rxn6196 S^2=0.216) are also the only two with modest ratios
(4.9x and 3.6x). Above S^2 ~ 0.33 the BS forces are uniformly large (26-63x) with
no useful ordering. So ⟨S²⟩ acts as a threshold indicator, not a magnitude predictor.

Caveat: rxn6196's ratio is partly deflated by an unusually large RKS gradient
(0.179 eV/Å, the largest in the set) rather than a small BS gradient.

Verification: bs_grad was computed on a fresh UKS object with mo_coeff/mo_occ set
but mo_energy left unset, which would corrupt the energy-weighted density matrix.
Tested directly: PySCF silently rebuilds mo_energy, and the gradients match a
reference calculation to 2e-7 Ha/Bohr. The numbers are sound.

## Cross-validation: independent code and the OMol25 protocol

Two checks, both at the same ORCA NEB TS geometries.

### 1. Independent code (ORCA)

ORCA 5.0.4 with `%scf STABPerform true; STABRestartUHFifUnstable true` reproduces
the PySCF result. rxn1320: ⟨S²⟩ = 0.779 (ORCA/def2-SVP) vs 0.785
(PySCF/def2-TZVP), same spin-localisation pattern — the overall sign is arbitrary
in a BS solution. ORCA reports "current solution is stable" after the restart,
i.e. the BS solution is a genuine minimum in orbital space. Magnetic coupling from
the same run: J = −1531 cm⁻¹, strongly antiferromagnetic.

### 2. The OMol25 protocol

OMol25 (arXiv:2505.08762, Sec. 2.7) breaks spin symmetry differently — *"rotate by
20° between the HOMO and LUMO in the β space"* — and reports (App. A) that **<5 %
of Transition1x has ⟨S²⟩ > 0.001**. The MLIPs benchmarked in this project are
trained on that data, so it matters whether that protocol reaches the same states.

Tested on all 26 reactions at **OMol25 settings**: wB97M-V/def2-TZVPD, def2/J,
RIJCOSX, TightSCF, DEFGRID3, `Thresh 1e-12`, `TCut 1e-13`. ORCA 5.0.4 (6.0.0 is
not installed; the paper notes these thresholds became defaults in later versions,
so setting them by hand reproduces that). Three single points per reaction:
**A** = RKS, **B** = UKS + 20° β rotation (OMol25), **C** = UKS + stability restart
(ours). 78/78 completed, no errors.

| rxn | E_RKS (Ha) | ⟨S²⟩ B (OMol25) | ⟨S²⟩ C (ours) | ΔE_BS (meV) |
|---|---|---|---|---|
| rxn7949 | -323.326102386 | 0.893370 | 0.893417 | -551.7 |
| rxn8832 | -323.307473395 | 0.872165 | 0.872109 | -425.3 |
| rxn4518 | -322.333191235 | 0.843592 | 0.843582 | -643.0 |
| rxn1320 | -322.380409196 | 0.791817 | 0.791820 | -343.1 |
| rxn8837 | -323.270149118 | 0.743003 | 0.742900 | -292.7 |
| rxn4522 | -322.323351893 | 0.668960 | 0.668877 | -185.9 |
| rxn5691 | -323.292429592 | 0.636036 | 0.635668 | -155.6 |
| rxn0346 | -322.310085462 | 0.619807 | 0.619880 | -138.8 |
| rxn0894 | -322.363613959 | 0.577380 | 0.577344 | -183.6 |
| rxn1147 | -322.377026723 | 0.547404 | 0.547287 | -108.9 |
| rxn7957 | -323.326423266 | 0.510914 | 0.510852 |  -94.6 |
| rxn8885 | -323.298858907 | 0.466086 | 0.466193 |  -32.9 |
| rxn1283 | -322.331087380 | 0.436097 | 0.436161 |  -45.1 |
| rxn3107 | -322.368966993 | 0.390092 | 0.390216 |  -33.8 |
| rxn7060 | -323.257407023 | 0.369919 | 0.370861 |  -20.3 |
| rxn8827 | -323.283870345 | 0.333754 | 0.333934 |  -25.1 |
| rxn6196 | -323.275188450 | 0.213589 | 0.211236 |  -10.3 |
| rxn4113 | -322.370228779 | 0.118779 | 0.117803 |   -5.9 |
| rxn0101 | -322.424685216 | 0.000000 | 0.000000 |    0.0 |
| rxn0896 | -322.353078879 | 0.000000 | 0.000000 |    0.0 |
| rxn10005 | -392.620312585 | 0.000103 | 0.000000 |    0.0 |
| rxn10054 | -323.357533357 | 0.000000 | 0.000000 |    0.0 |
| rxn1150 | -322.398182478 | 0.000000 | 0.000000 |    0.0 |
| rxn7936 | -323.258181226 | 0.000000 | 0.000000 |    0.0 |
| rxn7937 | -323.330887869 | 0.000001 | 0.000000 |    0.0 |
| rxn7945 | -323.328161657 | 0.000000 | 0.000000 |    0.0 |

**The two protocols are identical, without exception.** Energies agree to ~10⁻⁸ Ha
across all 26, including the weakest case (rxn4113, ΔE_BS = −5.9 meV, ⟨S²⟩ = 0.12).
Exactly 8 reactions give ⟨S²⟩ = 0 in both variants, and they are precisely the 8
that the PySCF stability analysis classified as externally stable. Two codes, two
basis sets, two different methods — same 18/8 split.

**Conclusion.** The OMol25 training data *does* contain these broken-symmetry
states. The <5 % figure is a selection effect — the 26 reactions here are the top
26 of 279 by N_FOD — not a failure of their protocol. An earlier working
hypothesis in this project, that the weak 20° guess would collapse and the models
therefore never saw these states, is **falsified**. It was extrapolated from
PySCF's DIIS behaviour (see pitfall 1 below), which does not transfer to ORCA.

Whether the MLIPs actually *learned* these states is a separate, still-open
question — this result only establishes that they could have.

### 3. Basis and grid sensitivity

ΔE_BS at def2-TZVP (PySCF, default grid) vs def2-TZVPD + DEFGRID3 (ORCA):

| rxn | TZVP | TZVPD | diff | | rxn | TZVP | TZVPD | diff |
|---|---|---|---|---|---|---|---|---|
| rxn4518 | -648.5 | -643.0 | +5.5 | | rxn7957 |  -99.8 |  -94.6 | +5.2 |
| rxn7949 | -559.6 | -551.7 | +7.9 | | rxn1283 |  -44.5 |  -45.1 | -0.6 |
| rxn8832 | -428.0 | -425.3 | +2.7 | | rxn8885 |  -42.8 |  -32.9 | **+9.9** |
| rxn1320 | -339.2 | -343.1 | -3.9 | | rxn3107 |  -38.8 |  -33.8 | +5.0 |
| rxn8837 | -293.9 | -292.7 | +1.2 | | rxn8827 |  -27.5 |  -25.1 | +2.4 |
| rxn0894 | -190.2 | -183.6 | +6.6 | | rxn7060 |  -22.1 |  -20.3 | +1.8 |
| rxn4522 | -184.0 | -185.9 | -1.9 | | rxn6196 |  -10.7 |  -10.3 | +0.4 |
| rxn5691 | -155.9 | -155.6 | +0.3 | | rxn4113 |   -8.4 |   -5.9 | +2.5 |
| rxn0346 | -147.6 | -138.8 | +8.8 | | | | | |

At most ~10 meV, typically under 6 — negligible against effect sizes of 6–650 meV
(ΔE_BS) and 0.6–2.6 eV/Å (BS gradients). **The benchmark does not need to be
recomputed at OMol25 settings on account of this.**

*Caveat:* for the weakest cases the *relative* shift is large — rxn4113 loses 30 %
of its ΔE_BS. Statements about marginally unstable reactions are basis-sensitive.

---

## BS-UKS TS optimization (secondary)

> **Secondary result.** The primary output of this analysis is the stability
> classification, the ΔE_BS / ⟨S²⟩ table and the BS gradient finding above. The
> TS optimizations were run to quantify how far the geometries actually move; they
> are not needed for any conclusion drawn above and are recorded here mainly so the
> driver-code pitfalls are not rediscovered.

Setup: geomeTRIC `transition=True`, BS solution maintained across every geometry
step, no frequency step (wB97M-V has no analytic Hessian — see pitfall 4).
14 reactions = the externally unstable ones carrying at least one model TS error
> 0.3 Å.

**Outcome (job 10682832):** 8 of 14 converged, 5 lost the BS solution during the
optimization (⟨S²⟩ < 0.3), 1 collapsed already at the starting geometry (rxn4113,
both routes, ⟨S²⟩ = 0.14 — consistent with it being the weakest case in the whole
set). Electron count conserved to 4×10⁻¹³ throughout, and step counts vary between
78 and 217 rather than all hitting the 150-step limit — both are evidence the
potential energy surface was self-consistent.

Notably, ⟨S²⟩ *increases* during the optimizations that converge (rxn8827
0.34 → 1.02, rxn1320 0.79 → 1.02): the geometry relaxes towards a more strongly
diradical structure, which is what the large BS gradients predicted.

### Driver-code pitfalls

Four submission attempts were needed. All defects were in the driver, not the
physics — each is easy to hit again.

1. **Plain DIIS destroys the BS solution in PySCF.** The first attempts collapsed
   all 8 reactions to closed-shell (ΔE = 0, ⟨S²⟩ = 0). Second-order Newton
   (`mf.newton()`) is required to hold it through convergence. *ORCA's default SCF
   does not have this problem* — which is why the OMol25-protocol comparison above
   works there.
2. **Seed UKS from `mf_rks.to_uks()`, not a fresh `dft.UKS(mol)`.** The latter has
   `mo_occ = None`, so `make_rdm1(mo_ext, mo_occ)` raises `TypeError: 'NoneType'
   object is not subscriptable`.
3. **A BS-maintaining kernel must be a class-level method, not an instance
   attribute.** `nuc_grad_method().as_scanner()` builds `self.base` as a *copy* of
   the SCF object. An instance-attribute kernel is inherited by the copy through
   `__dict__` but its closure writes results back to the original, leaving the
   copy's `mo_coeff` at `None` — the gradient then dereferences None on the first
   geometry step, every time.
4. **Carrying BS across geometries needs a density matrix, not MO coefficients.**
   MOs converged at geometry A are orthonormal w.r.t. S(A); handing them to an SCF
   at geometry B corrupts the density *silently*. Measured on H₂O for a 0.15 Å
   step: electron count 10.000 → 10.056, energy 654 meV too low, ⟨S²⟩ negative,
   `max|CᵀS(B)C − I|` = 0.166. Pass `dm0` instead. **Negative ⟨S²⟩ is the
   diagnostic** that this has happened.
5. **No analytic Hessian for wB97M-V in either code.** PySCF: `UKS Hessian for NLC
   functional`. ORCA 5.0.4: `ORCA_CPSCF: The CPSCF equations can not yet handle
   non-local correlation`. Numerical Hessians must be forced explicitly
   (`%geom Calc_Hess true; NumHess true end` in ORCA). Passing `hessian='first'`
   to geomeTRIC takes the analytic path and returns None.
6. **`geometric_solver.optimize()` discards the convergence flag** and returns the
   molecule only — it reports success even when geomeTRIC printed *"Geometry
   optimization failed to converge in N iterations"*. Use
   `geometric_solver.kernel()`, which returns `(converged, mol)`.

## Outstanding

- Whether the MLIPs learned the broken-symmetry states that OMol25's training data
  demonstrably contains — see cross-validation section 2.
- ORCA cross-check of the 6 non-converged TS optimizations, to separate genuine
  physical BS loss from a PySCF-specific artefact.
