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

## Gradients at model-predicted TS geometries

The gradient check above asks whether the *RKS reference* TS geometries are
stationary on the BS surface. This section asks the same of the geometries the
MLIPs predict. 18 externally unstable reactions x 4 geometry sources
(RKS-ref, UMA-S, UMA-M, eSEN) = 72 single points at wB97M-V/def2-TZVP
(PySCF, grids 3, conv_tol 1e-10). No optimisation. Job 10687985.

The RKS-ref rows are recomputed here rather than reused from the earlier run,
so every row is produced by identical code.

| rxn | Geometry | RKS max\|∇E\| | RKS RMS\|∇E\| | int | λ_min_int | ext | λ_min_ext | ΔE_BS (meV) | ⟨S²⟩ | BS max\|∇E\| | BS RMS\|∇E\| | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rxn0346 | RKS-ref | 0.0519 | 0.0206 | ✓ | 0.383038 | ✗ | -0.027187 | -147.6 | 0.628 | 2.6125 | 0.7861 |  |
| rxn0346 | UMA-S | 2.4732 | 0.7136 | ✓ | 0.348932 | ✗ | -0.022699 | -118.1 | 0.609 | 0.2437 | 0.0739 |  |
| rxn0346 | UMA-M | 2.5684 | 0.7404 | ✓ | 0.334612 | ✗ | -0.021260 | -110.4 | 0.609 | 0.1735 | 0.0632 |  |
| rxn0346 | eSEN | 2.7482 | 0.7904 | ✓ | 0.348778 | ✗ | -0.025359 | -145.8 | 0.654 | 0.4849 | 0.1260 |  |
| rxn0894 | RKS-ref | 0.0624 | 0.0252 | ✓ | 0.327717 | ✗ | -0.040138 | -190.2 | 0.580 | 1.3498 | 0.4495 |  |
| rxn0894 | UMA-S | 3.2894 | 0.9212 | ✓ | 0.213314 | ✗ | -0.112214 | -1278.2 | 0.986 | 0.7813 | 0.2924 |  |
| rxn0894 | UMA-M | — | — | — | — | — | — | — | — | — | — | **rks_not_converged** |
| rxn0894 | eSEN | 0.6602 | 0.1933 | ✓ | 0.353759 | ✗ | -0.277347 | -3992.0 | 1.028 | 0.7991 | 0.2041 |  |
| rxn1147 | RKS-ref | 0.0651 | 0.0229 | ✓ | 0.298042 | ✗ | -0.024502 | -105.2 | 0.534 | 1.8398 | 0.5793 |  |
| rxn1147 | UMA-S | 0.0774 | 0.0286 | ✓ | 0.507009 | ✓ | 0.006981 | — | — | — | — | stable, no BS solution |
| rxn1147 | UMA-M | 0.0496 | 0.0196 | ✓ | 0.510069 | ✓ | 0.006877 | — | — | — | — | stable, no BS solution |
| rxn1147 | eSEN | 0.0682 | 0.0274 | ✓ | 0.508809 | ✓ | 0.006603 | — | — | — | — | stable, no BS solution |
| rxn1283 | RKS-ref | 0.0380 | 0.0165 | ✓ | 0.364174 | ✗ | -0.013893 | -44.5 | 0.419 | 2.3861 | 0.8036 |  |
| rxn1283 | UMA-S | 1.3199 | 0.5156 | ✗ | -0.090272 | ✗ | -0.127208 | -2030.2 | 0.978 | 0.1597 | 0.0574 |  |
| rxn1283 | UMA-M | 1.5483 | 0.6448 | ✗ | -0.074092 | ✗ | -0.119619 | -1867.9 | 0.999 | 0.1251 | 0.0429 |  |
| rxn1283 | eSEN | 1.1161 | 0.4430 | ✗ | -0.072255 | ✗ | -0.125156 | -1944.1 | 0.987 | 0.1054 | 0.0372 |  |
| rxn1320 | RKS-ref | 0.0588 | 0.0211 | ✓ | 0.438950 | ✗ | -0.048011 | -339.2 | 0.785 | 2.0728 | 0.8134 |  |
| rxn1320 | UMA-S | 1.8300 | 0.5410 | ✓ | 0.611219 | ✗ | -0.049994 | -301.9 | 0.688 | 0.0687 | 0.0285 |  |
| rxn1320 | UMA-M | 1.7957 | 0.5225 | ✓ | 0.611217 | ✗ | -0.050712 | -310.3 | 0.695 | 0.0441 | 0.0155 |  |
| rxn1320 | eSEN | 1.7797 | 0.5285 | ✓ | 0.613786 | ✗ | -0.049769 | -299.3 | 0.686 | 0.1174 | 0.0321 |  |
| rxn3107 | RKS-ref | 0.0630 | 0.0199 | ✓ | 0.282750 | ✗ | -0.012545 | -38.8 | 0.409 | 1.6457 | 0.4754 |  |
| rxn3107 | UMA-S | 0.6384 | 0.2110 | ✓ | 0.258070 | ✗ | -0.005947 | -8.5 | 0.205 | 0.1633 | 0.0424 |  |
| rxn3107 | UMA-M | 0.6827 | 0.2160 | ✓ | 0.271357 | ✗ | -0.005248 | -6.4 | 0.176 | 0.0922 | 0.0293 |  |
| rxn3107 | eSEN | 0.7127 | 0.2039 | ✓ | 0.274507 | ✗ | -0.004329 | -4.4 | 0.148 | 0.1426 | 0.0469 |  |
| rxn4113 | RKS-ref | 0.0786 | 0.0218 | ✓ | 0.267352 | ✗ | -0.008458 | -8.4 | 0.140 | 0.3859 | 0.1112 |  |
| rxn4113 | UMA-S | 0.3258 | 0.0983 | ✓ | 0.258326 | ✗ | -0.009216 | -9.8 | 0.150 | 0.1728 | 0.0538 |  |
| rxn4113 | UMA-M | 4.0293 | 0.9904 | ✓ | 0.178517 | ✗ | -0.146553 | -1940.0 | 1.007 | 0.1848 | 0.0429 |  |
| rxn4113 | eSEN | 0.5853 | 0.1471 | ✓ | 0.247257 | ✗ | -0.011583 | -15.6 | 0.189 | 0.1860 | 0.0555 |  |
| rxn4518 | RKS-ref | 0.0681 | 0.0218 | ✓ | 0.418023 | ✗ | -0.077799 | -648.5 | 0.842 | 2.9493 | 0.8927 |  |
| rxn4518 | UMA-S | 2.2357 | 0.6248 | ✓ | 0.311186 | ✗ | -0.207068 | -2852.2 | 1.009 | 0.0582 | 0.0216 |  |
| rxn4518 | UMA-M | 2.3986 | 0.6673 | ✓ | 0.315961 | ✗ | -0.203599 | -2789.6 | 1.008 | 0.0554 | 0.0167 |  |
| rxn4518 | eSEN | 1.9617 | 0.5532 | ✓ | 0.304443 | ✗ | -0.214835 | -2974.5 | 1.009 | 0.0514 | 0.0229 |  |
| rxn4522 | RKS-ref | 0.0982 | 0.0347 | ✓ | 0.403045 | ✗ | -0.032467 | -184.0 | 0.662 | 1.8750 | 0.5945 |  |
| rxn4522 | UMA-S | 1.3031 | 0.4053 | ✓ | 0.340968 | ✗ | -0.166824 | -2209.6 | 1.005 | 0.0748 | 0.0230 |  |
| rxn4522 | UMA-M | 1.2875 | 0.4012 | ✓ | 0.341012 | ✗ | -0.166293 | -2193.4 | 1.005 | 0.0831 | 0.0266 |  |
| rxn4522 | eSEN | 1.2982 | 0.3998 | ✓ | 0.343062 | ✗ | -0.166243 | -2197.3 | 1.005 | 0.0730 | 0.0277 |  |
| rxn5691 | RKS-ref | 0.0407 | 0.0170 | ✓ | 0.294543 | ✗ | -0.029016 | -155.9 | 0.629 | 1.4192 | 0.5156 |  |
| rxn5691 | UMA-S | 2.9360 | 0.8707 | ✓ | 0.234457 | ✗ | -0.122424 | -1592.6 | 1.011 | 0.1537 | 0.0412 |  |
| rxn5691 | UMA-M | 2.5080 | 0.7929 | ✓ | 0.221110 | ✗ | -0.124513 | -1725.8 | 1.017 | 0.0853 | 0.0354 |  |
| rxn5691 | eSEN | 2.9647 | 0.8690 | ✓ | 0.235326 | ✗ | -0.122943 | -1601.7 | 1.011 | 0.0681 | 0.0284 |  |
| rxn6196 | RKS-ref | 0.1793 | 0.0487 | ✓ | 0.291837 | ✗ | -0.006866 | -10.7 | 0.216 | 0.6384 | 0.1567 |  |
| rxn6196 | UMA-S | 1.2876 | 0.3534 | ✓ | 0.211248 | ✗ | -0.017280 | -72.2 | 0.520 | 0.0900 | 0.0366 |  |
| rxn6196 | UMA-M | 1.2174 | 0.3553 | ✓ | 0.205821 | ✗ | -0.016170 | -64.2 | 0.500 | 0.1376 | 0.0546 |  |
| rxn6196 | eSEN | 1.2424 | 0.3601 | ✓ | 0.202834 | ✗ | -0.016708 | -68.8 | 0.515 | 0.1404 | 0.0532 |  |
| rxn7060 | RKS-ref | 0.0335 | 0.0095 | ✓ | 0.365660 | ✗ | -0.007904 | -22.1 | 0.374 | 1.7658 | 0.5817 |  |
| rxn7060 | UMA-S | 0.8820 | 0.2488 | ✓ | 0.270664 | ✓ | 0.002772 | — | — | — | — | stable, no BS solution |
| rxn7060 | UMA-M | 0.3342 | 0.1174 | ✓ | 0.337918 | ✓ | 0.001343 | — | — | — | — | stable, no BS solution |
| rxn7060 | eSEN | 1.1231 | 0.3027 | ✓ | 0.259207 | ✓ | 0.002878 | — | — | — | — | stable, no BS solution |
| rxn7949 | RKS-ref | 0.1049 | 0.0305 | ✓ | 0.354977 | ✗ | -0.063146 | -559.6 | 0.893 | 1.6860 | 0.5642 |  |
| rxn7949 | UMA-S | 1.3939 | 0.4088 | ✓ | 0.318806 | ✗ | -0.108034 | -1276.7 | 0.997 | 0.2476 | 0.0806 |  |
| rxn7949 | UMA-M | 1.5896 | 0.4472 | ✓ | 0.314711 | ✗ | -0.110179 | -1319.0 | 1.000 | 0.0514 | 0.0147 |  |
| rxn7949 | eSEN | 1.6405 | 0.4504 | ✓ | 0.314908 | ✗ | -0.111271 | -1338.8 | 1.001 | 0.0737 | 0.0232 |  |
| rxn7957 | RKS-ref | 0.0265 | 0.0134 | ✓ | 0.351785 | ✗ | -0.023963 | -99.8 | 0.513 | 0.9010 | 0.3725 |  |
| rxn7957 | UMA-S | 3.9200 | 0.9415 | ✓ | 0.488368 | ✗ | -0.061896 | -416.8 | 0.734 | 0.1374 | 0.0478 |  |
| rxn7957 | UMA-M | 3.4918 | 0.8590 | ✓ | 0.494234 | ✗ | -0.054658 | -332.5 | 0.685 | 0.1130 | 0.0398 |  |
| rxn7957 | eSEN | 3.8347 | 0.9351 | ✓ | 0.488375 | ✗ | -0.061366 | -410.4 | 0.731 | 0.1094 | 0.0429 |  |
| rxn8827 | RKS-ref | 0.0263 | 0.0129 | ✓ | 0.383643 | ✗ | -0.010957 | -27.5 | 0.338 | 1.1278 | 0.3391 |  |
| rxn8827 | UMA-S | 2.1917 | 0.6836 | ✓ | 0.339694 | ✗ | -0.085591 | -1026.0 | 1.001 | 0.1733 | 0.0682 |  |
| rxn8827 | UMA-M | 2.1912 | 0.6810 | ✓ | 0.325756 | ✗ | -0.085540 | -1051.1 | 1.007 | 0.1336 | 0.0502 |  |
| rxn8827 | eSEN | 2.3331 | 0.6870 | ✓ | 0.323730 | ✗ | -0.087262 | -1083.7 | 1.009 | 0.2284 | 0.0813 |  |
| rxn8832 | RKS-ref | 0.1420 | 0.0444 | ✓ | 0.369765 | ✗ | -0.049255 | -428.0 | 0.870 | 2.7328 | 0.7580 |  |
| rxn8832 | UMA-S | 2.6137 | 0.6449 | ✓ | 0.382434 | ✗ | -0.087258 | -1000.1 | 0.982 | 0.0963 | 0.0420 |  |
| rxn8832 | UMA-M | 2.7662 | 0.6782 | ✓ | 0.367552 | ✗ | -0.087835 | -1033.3 | 0.990 | 0.0754 | 0.0254 |  |
| rxn8832 | eSEN | 2.7699 | 0.6590 | ✓ | 0.375100 | ✗ | -0.088537 | -1032.6 | 0.988 | 0.2320 | 0.0711 |  |
| rxn8837 | RKS-ref | 0.0568 | 0.0178 | ✓ | 0.424887 | ✗ | -0.044006 | -293.9 | 0.741 | 1.6974 | 0.4736 |  |
| rxn8837 | UMA-S | 0.9658 | 0.2779 | ✓ | 0.114017 | ✗ | -0.244378 | 3421.0 | 0.000 | 1.3236 | 0.3941 | **invalid: S²=0, ΔE>0** |
| rxn8837 | UMA-M | 1.5087 | 0.4676 | ✓ | 0.205257 | ✗ | -0.225149 | -3230.9 | 1.007 | 0.7640 | 0.2927 |  |
| rxn8837 | eSEN | 0.1011 | 0.0401 | ✓ | 0.314190 | ✓ | 0.047278 | — | — | — | — | stable, no BS solution |
| rxn8885 | RKS-ref | 0.0423 | 0.0185 | ✓ | 0.246709 | ✗ | -0.011092 | -42.8 | 0.507 | 2.6373 | 0.6142 |  |
| rxn8885 | UMA-S | 0.8377 | 0.3474 | ✓ | 0.040854 | ✗ | -0.156460 | -2033.7 | 1.024 | 0.4845 | 0.1303 |  |
| rxn8885 | UMA-M | 0.6401 | 0.1575 | ✓ | 0.237274 | ✗ | -0.004207 | -5.0 | 0.175 | 0.1899 | 0.0427 |  |
| rxn8885 | eSEN | 1.6953 | 0.4319 | ✓ | 0.027910 | ✗ | -0.216220 | -3014.8 | 1.028 | 0.3753 | 0.1041 |  |

### Summary

| | RKS max\|∇E\| (eV/Å) | BS max\|∇E\| (eV/Å) |
|---|---|---|
| RKS reference geometries | 0.026 – 0.179 | 0.386 – 2.949 |
| model geometries | 0.050 – 4.029 | 0.044 – 1.324 |

The relation inverts: at the RKS reference geometries the RKS gradient is small
and the BS gradient large; at the model geometries it is the other way round.
The two are not the same electronic state, however — ⟨S²⟩ is typically ≈ 1.0 at
the model geometries (fully formed diradical) against 0.14–0.89 at the reference,
and ΔE_BS reaches −4.0 eV against −0.65 eV. Gradients of different states at
different geometries are being compared, not one quantity twice.

Three reactions are externally **stable** at the model geometries while unstable
at the reference, i.e. no BS solution exists there: rxn1147 (all three models),
rxn7060 (all three), rxn8837 (eSEN).

rxn1283 is the only reaction with an internally unstable *RKS* solution, and it
occurs at all three model geometries (λ_min_int = −0.090 / −0.074 / −0.072)
while the reference is internally stable (+0.364).

Two rows are unusable: rxn0894/UMA-M (RKS not converged) and rxn8837/UMA-S
(ΔE_BS = +3421 meV at ⟨S²⟩ = 0.000 — Newton ran into a higher solution).

## Stability of the broken-symmetry solutions themselves

The stability analyses above are run on the *RKS* solution and answer "does a BS
solution lie below RKS". They do not say whether the BS solution is itself the
lowest. This section runs `mf_u.stability(internal=True, external=True)` on the
converged BS solution. Job 10688500, 63 of 72 rows (skipping rows without a
valid BS solution).

Job 10687985 retained no orbitals, so each BS solution was re-converged by the
identical Route-1 path first. Column `repro Δ` is the deviation of the
re-converged ΔE_BS from the stored value.

| rxn | Geometry | ΔE_BS (meV) | ⟨S²⟩ | repro Δ | uks int | λ_min_int (UKS) | uks ext | λ_min_ext (UKS) | ΔE₂ (meV) | ⟨S²⟩₂ | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rxn0346 | RKS-ref | -147.6 | 0.628 | -0.000 | ✓ | 0.068574 | ✗ | -0.042040 | — | — |  |
| rxn0346 | UMA-S | -118.1 | 0.609 | 0.000 | ✓ | 0.062169 | ✗ | -0.045104 | — | — |  |
| rxn0346 | UMA-M | -110.4 | 0.609 | -0.000 | ✓ | 0.055859 | ✗ | -0.045232 | — | — |  |
| rxn0346 | eSEN | -145.8 | 0.654 | 0.000 | ✓ | 0.066333 | ✗ | -0.046739 | — | — |  |
| rxn0894 | RKS-ref | -190.2 | 0.580 | -0.000 | ✓ | 0.086594 | ✗ | -0.040222 | — | — |  |
| rxn0894 | UMA-S | -1278.2 | 0.986 | 0.000 | ✓ | 0.064935 | ✗ | -0.061291 | — | — |  |
| rxn0894 | UMA-M | — | — | — | — | — | — | — | — | — | *skipped: rks_not_converged* |
| rxn0894 | eSEN | -3992.0 | 1.028 | -0.000 | ✓ | 0.209918 | ✗ | **invalid** | — | — | λ_ext Davidson broke down (-3.2e+07) |
| rxn1147 | RKS-ref | -105.2 | 0.534 | -0.000 | ✓ | 0.056854 | ✗ | -0.028250 | — | — |  |
| rxn1147 | UMA-S | — | — | — | — | — | — | — | — | — | *skipped: no_bs_solution* |
| rxn1147 | UMA-M | — | — | — | — | — | — | — | — | — | *skipped: no_bs_solution* |
| rxn1147 | eSEN | — | — | — | — | — | — | — | — | — | *skipped: no_bs_solution* |
| rxn1283 | RKS-ref | -44.5 | 0.419 | 0.000 | ✓ | 0.035965 | ✗ | -0.030865 | — | — |  |
| rxn1283 | UMA-S | -2030.2 | 0.978 | -0.000 | ✓ | 0.002656 | ✗ | -0.145586 | — | — | λ_int ≈ 0, sign not reliable |
| rxn1283 | UMA-M | -1867.9 | 0.999 | 0.000 | ✗ | -0.000871 | ✗ | -0.144200 | -6.05 | 0.992 | **internally unstable, followed** |
| rxn1283 | eSEN | -1944.1 | 0.987 | 0.000 | ✓ | 0.001525 | ✗ | -0.149230 | — | — | λ_int ≈ 0, sign not reliable |
| rxn1320 | RKS-ref | -339.2 | 0.785 | -0.000 | ✓ | 0.098919 | ✗ | -0.051987 | — | — |  |
| rxn1320 | UMA-S | -301.9 | 0.688 | -0.000 | ✓ | 0.158547 | ✗ | -0.056004 | — | — |  |
| rxn1320 | UMA-M | -310.3 | 0.695 | -0.000 | ✓ | 0.160197 | ✗ | -0.056356 | — | — |  |
| rxn1320 | eSEN | -299.3 | 0.686 | 0.000 | ✓ | 0.157990 | ✗ | -0.056729 | — | — |  |
| rxn3107 | RKS-ref | -38.8 | 0.409 | -0.000 | ✓ | 0.028799 | ✗ | -0.029414 | — | — |  |
| rxn3107 | UMA-S | -8.5 | 0.205 | -0.000 | ✓ | 0.016535 | ✗ | -0.022486 | — | — |  |
| rxn3107 | UMA-M | -6.4 | 0.176 | -0.000 | ✓ | 0.015685 | ✗ | -0.022357 | — | — |  |
| rxn3107 | eSEN | -4.4 | 0.148 | 0.000 | ✓ | 0.013522 | ✗ | -0.021729 | — | — |  |
| rxn4113 | RKS-ref | -8.4 | 0.140 | -0.000 | ✓ | 0.031806 | ✗ | -0.022706 | — | — |  |
| rxn4113 | UMA-S | -9.8 | 0.150 | -0.000 | ✓ | 0.034576 | ✗ | -0.022159 | — | — |  |
| rxn4113 | UMA-M | -1940.0 | 1.007 | -0.000 | ✓ | 0.107490 | ✗ | -0.070008 | — | — |  |
| rxn4113 | eSEN | -15.6 | 0.189 | -0.000 | ✓ | 0.041910 | ✗ | -0.022333 | — | — |  |
| rxn4518 | RKS-ref | -648.5 | 0.842 | -0.000 | ✓ | 0.154267 | ✗ | -0.067885 | — | — |  |
| rxn4518 | UMA-S | -2852.2 | 1.009 | 0.000 | ✓ | 0.156446 | ✗ | -0.082645 | — | — |  |
| rxn4518 | UMA-M | -2789.6 | 1.008 | 0.000 | ✓ | 0.158042 | ✗ | -0.082706 | — | — |  |
| rxn4518 | eSEN | -2974.5 | 1.009 | -0.000 | ✓ | 0.154878 | ✗ | -0.083188 | — | — |  |
| rxn4522 | RKS-ref | -184.0 | 0.662 | -0.000 | ✓ | 0.075546 | ✗ | -0.039692 | — | — |  |
| rxn4522 | UMA-S | -2209.6 | 1.005 | -0.000 | ✓ | 0.158095 | ✗ | -0.082804 | — | — |  |
| rxn4522 | UMA-M | -2193.4 | 1.005 | -0.000 | ✓ | 0.157346 | ✗ | -0.082988 | — | — |  |
| rxn4522 | eSEN | -2197.3 | 1.005 | 0.000 | ✓ | 0.158617 | ✗ | -0.082689 | — | — |  |
| rxn5691 | RKS-ref | -155.9 | 0.629 | 0.000 | ✓ | 0.065179 | ✗ | -0.046802 | — | — |  |
| rxn5691 | UMA-S | -1592.6 | 1.011 | 0.000 | ✓ | 0.135145 | ✗ | -0.080627 | — | — |  |
| rxn5691 | UMA-M | -1725.8 | 1.017 | -0.000 | ✓ | 0.127962 | ✗ | -0.081654 | — | — |  |
| rxn5691 | eSEN | -1601.7 | 1.011 | 0.000 | ✓ | 0.134701 | ✗ | -0.080709 | — | — |  |
| rxn6196 | RKS-ref | -10.7 | 0.216 | 0.000 | ✓ | 0.025924 | ✗ | -0.034551 | — | — |  |
| rxn6196 | UMA-S | -72.2 | 0.520 | 0.000 | ✓ | 0.056298 | ✗ | -0.039073 | — | — |  |
| rxn6196 | UMA-M | -64.2 | 0.500 | 0.000 | ✓ | 0.053349 | ✗ | -0.038330 | — | — |  |
| rxn6196 | eSEN | -68.8 | 0.515 | 0.000 | ✓ | 0.054563 | ✗ | -0.038620 | — | — |  |
| rxn7060 | RKS-ref | -22.1 | 0.374 | 0.000 | ✓ | 0.019400 | ✗ | -0.029216 | — | — |  |
| rxn7060 | UMA-S | — | — | — | — | — | — | — | — | — | *skipped: no_bs_solution* |
| rxn7060 | UMA-M | — | — | — | — | — | — | — | — | — | *skipped: no_bs_solution* |
| rxn7060 | eSEN | — | — | — | — | — | — | — | — | — | *skipped: no_bs_solution* |
| rxn7949 | RKS-ref | -559.6 | 0.893 | 0.000 | ✓ | 0.105456 | ✗ | -0.049803 | — | — |  |
| rxn7949 | UMA-S | -1276.7 | 0.997 | -0.000 | ✓ | 0.107235 | ✗ | -0.062403 | — | — |  |
| rxn7949 | UMA-M | -1319.0 | 1.000 | -0.000 | ✓ | 0.106809 | ✗ | -0.061917 | — | — |  |
| rxn7949 | eSEN | -1338.8 | 1.001 | -0.000 | ✓ | 0.106873 | ✗ | -0.062062 | — | — |  |
| rxn7957 | RKS-ref | -99.8 | 0.513 | 0.000 | ✓ | 0.063038 | ✗ | -0.033232 | — | — |  |
| rxn7957 | UMA-S | -416.8 | 0.734 | 0.000 | ✓ | 0.167623 | ✗ | -0.048218 | — | — |  |
| rxn7957 | UMA-M | -332.5 | 0.685 | 0.000 | ✓ | 0.159910 | ✗ | -0.046522 | — | — |  |
| rxn7957 | eSEN | -410.4 | 0.731 | 0.000 | ✓ | 0.166304 | ✗ | -0.048230 | — | — |  |
| rxn8827 | RKS-ref | -27.5 | 0.338 | 0.000 | ✓ | 0.032411 | ✗ | -0.027028 | — | — |  |
| rxn8827 | UMA-S | -1026.0 | 1.001 | 0.000 | ✓ | 0.155533 | ✗ | -0.064176 | — | — |  |
| rxn8827 | UMA-M | -1051.1 | 1.007 | -0.000 | ✓ | 0.150463 | ✗ | -0.062874 | — | — |  |
| rxn8827 | eSEN | -1083.7 | 1.009 | 0.000 | ✓ | 0.152987 | ✗ | -0.065222 | — | — |  |
| rxn8832 | RKS-ref | -428.0 | 0.870 | 0.000 | ✓ | 0.098813 | ✗ | -0.042372 | — | — |  |
| rxn8832 | UMA-S | -1000.1 | 0.982 | 0.000 | ✓ | 0.171797 | ✗ | -0.061397 | — | — |  |
| rxn8832 | UMA-M | -1033.3 | 0.990 | 0.000 | ✓ | 0.166121 | ✗ | -0.059606 | — | — |  |
| rxn8832 | eSEN | -1032.6 | 0.988 | -0.000 | ✓ | 0.170420 | ✗ | -0.061287 | — | — |  |
| rxn8837 | RKS-ref | -293.9 | 0.741 | 0.000 | ✓ | 0.115905 | ✗ | -0.039299 | — | — |  |
| rxn8837 | UMA-S | — | — | — | — | — | — | — | — | — | *skipped: invalid_bs (dE=3420.988, S2=0.0)* |
| rxn8837 | UMA-M | -3230.9 | 1.007 | -0.000 | ✓ | 0.124880 | ✗ | **invalid** | — | — | λ_ext Davidson broke down (-3e+03) |
| rxn8837 | eSEN | — | — | — | — | — | — | — | — | — | *skipped: no_bs_solution* |
| rxn8885 | RKS-ref | -42.8 | 0.507 | -0.000 | ✓ | 0.020921 | ✗ | -0.028697 | — | — |  |
| rxn8885 | UMA-S | -2033.7 | 1.024 | 0.000 | ✓ | 0.000685 | ✗ | -0.095288 | — | — | λ_int ≈ 0, sign not reliable |
| rxn8885 | UMA-M | -5.0 | 0.175 | 0.000 | ✓ | 0.011184 | ✗ | -0.019390 | — | — |  |
| rxn8885 | eSEN | -3014.8 | 1.028 | -0.000 | ✓ | 0.005404 | ✗ | -0.096171 | — | — |  |

### Summary

| | |
|---|---|
| rows evaluated | 63 / 63 |
| UKS **internally** unstable | **1** |
| UKS **externally** unstable | **63** (2 with an invalid λ value) |
| reproducibility of ΔE_BS vs. stored | max deviation **0.000 meV** |

**The BS solutions are the lowest single-determinant solutions in the real,
spin-collinear subspace.** 62 of 63 are internally stable. The single exception,
rxn1283/UMA-M (λ_min_int = −0.000871), was followed once and gave a solution
6.05 meV lower at essentially unchanged ⟨S²⟩ (0.999 → 0.992) — no collapse.

All 63 are **externally** unstable, i.e. a solution of still lower symmetry
(GHF/GKS, non-collinear spin) lies below. That class is not computed here.

Three further rows have λ_min_int within 0.003 of zero, where the sign is not
numerically reliable (Davidson converges to ~1e-4…1e-5): rxn8885/UMA-S
(+0.000685), rxn1283/eSEN (+0.001525), rxn1283/UMA-S (+0.002656). Notably all
three model geometries of **rxn1283** sit at |λ_min_int| < 0.003 while its
reference geometry is at 0.036 — a factor of 20. This is the same reaction whose
RKS solution is internally unstable at every model geometry.

Two λ_min_ext values are unphysical and marked invalid: rxn0894/eSEN (−3.2e7)
and rxn8837/UMA-M (−3.0e3), against −0.019…−0.149 for all valid rows. PySCF
reports "UHF/UKS -> GHF/GKS instability" but the Davidson diagonalisation has
broken down. Both are rows with ΔE_BS < −3000 meV. λ_min_int and the BS
solutions themselves are unaffected.

Orbitals are stored this time under `~/uks_stab/{rxn}/bs_<tag>.npz` (and
`bs2_<tag>.npz` for the followed solution), so a re-analysis needs no
re-convergence.

| Script | Purpose |
|---|---|
| `grad_at_model_ts.py` | RKS + stability + BS + gradients at all four geometry sources |
| `uks_stability.py` | stability analysis of the converged BS solutions |

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
- Two rows still unusable and worth a retry with a different guess or a level
  shift: rxn0894/UMA-M (RKS not converged) and rxn8837/UMA-S (Newton ran into a
  higher solution, ΔE_BS = +3421 meV at ⟨S²⟩ = 0).
- The two λ_min_ext Davidson breakdowns (rxn0894/eSEN, rxn8837/UMA-M) could be
  retried with a tighter `stability` convergence setting if the GHF/GKS question
  matters; the internal stability of those rows is unaffected.

## Related: NEB reference at OMol25 settings

A separate run recomputed the reference NEB at OMol25 settings (def2-TZVPD,
DEFGRID3, tight integral thresholds) for 45 reactions — the top-26 by N_FOD plus
the mid-10 and low-10 control groups. 31 of 45 converged before the run was
stopped. Against the existing def2-TZVP reference:

| | |
|---|---|
| TS geometry shift | median **0.0020 Å**, max 0.0134 Å (n = 30) |
| barrier shift | **−31.6 to +10.3 meV** |

Two to three orders of magnitude smaller than the effects studied here (model TS
errors 0.07–2.8 Å, ΔE_BS 6–650 meV). **The benchmark does not need to be
recomputed at OMol25 settings**; the deviation is quantified and can be stated in
the methods section. All three MR groups are represented among the 30.

**Methodological finding from that run:** ORCA run with `%pal nprocs > 1` produces
NEB forces that stall at fmax ≈ 0.05–0.11 eV/Å and never reach the 0.05
convergence target, while the original serial run (`nprocs 1`) converged the same
reactions in 6–11 steps. Two diagnostic runs isolated it: neither def2-TZVP with
DEFGRID3 and tight thresholds, nor def2-TZVPD with the default grid, converged —
the only factor both share with the production run and lack in the original is the
MPI parallelisation. Affected reactions included those with the *lowest* MR
character, so it is not a chemistry effect. Relevant to any future NEB in this
project. Scripts: `pipeline/orca_neb_omol25.py`, `pipeline/job_neb_diag.sh`.
