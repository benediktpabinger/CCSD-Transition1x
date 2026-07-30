# CASSCF Single-Point Results at DFT TS Geometries

**Geometry source:** ORCA wB97M-V/def2-TZVP NEB transition states (`~/orca_neb_results/{rxn}/transition_state.xyz`)
**Method:** PySCF CASSCF/def2-tzvp, AVAS initial guess (`['C 2pz', 'N 2p', 'O 2pz', 'F 2pz']`, threshold=0.4), same active space as OptTS pipeline
**Jobs:** 10674792 (single-state, 22 reactions), 10674827 (SA-2roots retry, 4 oscillating reactions)
**Date:** 2026-07-24

---

## SA-CASSCF Retry Results (rxn1150, rxn0896, rxn8837)

**SA setup:** state_average_([0.5, 0.5]), fix_spin_(ss=0), max_cycle_macro=150, no other tweaks
**Note on CASCI NOONs:** the single-state script does not log NOONs mid-run; CASCI NOONs below are from a single CASCI step at the DFT geometry using AVAS orbitals — closest available proxy for final-iteration NOONs of the killed single-state runs.

---

### rxn1150 — (12e, 8o)

**Convergence:** YES — 1-step SA-CASSCF converged in 132 macro iterations (1844 JK, 398 micro)

**Energies and gap:**

| Root | Energy (Ha) | ⟨S²⟩ |
|------|-------------|-------|
| Root 0 | −320.72533499 | 0.0000 |
| Root 1 | −320.67634114 | 0.0000 |
| Gap (Root1 − Root0) | +0.04899385 Ha | +1333.7 meV |

**CASCI NOONs (proxy for final-iteration NOONs of killed single-state run):**
```
2.0000  1.9999  1.9999  1.9985  1.9793  1.9605  0.0317  0.0300
```

**State-specific NOONs (state-specific CASCI on SA orbitals), sorted descending, 4 dp:**

| NO | Root 0 | Root 1 | in (0.02,1.98) R0 | in (0.02,1.98) R1 |
|----|--------|--------|-------------------|-------------------|
| 1 | 1.9989 | 1.9992 | | |
| 2 | 1.9971 | 1.9986 | | |
| 3 | 1.9871 | 1.9970 | | |
| 4 | 1.9758 | 1.9880 | ✓ | |
| 5 | 1.9106 | 1.9138 | ✓ | ✓ |
| 6 | 1.8766 | 1.0186 | ✓ | ✓ |
| 7 | 0.1504 | 1.0017 | ✓ | ✓ |
| 8 | 0.1034 | 0.0833 | ✓ | ✓ |

Root 0 NOONs in (0.02, 1.98): `1.9758  1.9106  1.8766  0.1504  0.1034`
Root 1 NOONs in (0.02, 1.98): `1.9138  1.0186  1.0017  0.0833`

**Root 0 vs OptTS CASSCF NOONs (nat_occ from nevpt2_optts_results.json, same (12e,8o)):**

| NO | Root 0 (DFT geom) | OptTS geom | Δ |
|----|-------------------|------------|---|
| 1 | 1.9989 | 1.9986 | +0.0003 |
| 2 | 1.9971 | 1.9971 | 0.0000 |
| 3 | 1.9871 | 1.9897 | −0.0026 |
| 4 | 1.9758 | 1.9870 | −0.0112 |
| 5 | 1.9106 | 1.9510 | −0.0404 |
| 6 | 1.8766 | 1.9402 | −0.0636 |
| 7 | 0.1504 | 0.0756 | +0.0748 |
| 8 | 0.1034 | 0.0606 | +0.0428 |

**Orbital compositions:** Not available — SA retry script does not run Löwdin population analysis.

---

### rxn0896 — (14e, 9o)

**Convergence:** NO — 1-step SA-CASSCF not converged, stopped at 150 macro iterations (1854 JK, 414 micro). Results from non-converged orbitals.

**Energies and gap (non-converged):**

| Root | Energy (Ha) | ⟨S²⟩ |
|------|-------------|-------|
| Root 0 | −320.65217235 | 0.0000 |
| Root 1 | −320.61720515 | 0.0000 |
| Gap (Root1 − Root0) | +0.03496720 Ha | +951.6 meV |

**CASCI NOONs (proxy for final-iteration NOONs of killed single-state run):**
```
1.9999  1.9997  1.9997  1.9988  1.9982  1.9931  1.9821  0.0204  0.0080
```

**State-specific NOONs (state-specific CASCI on non-converged SA orbitals), sorted descending, 4 dp:**

| NO | Root 0 | Root 1 | in (0.02,1.98) R0 | in (0.02,1.98) R1 |
|----|--------|--------|-------------------|-------------------|
| 1 | 1.9996 | 1.9998 | | |
| 2 | 1.9991 | 1.9989 | | |
| 3 | 1.9985 | 1.9982 | | |
| 4 | 1.9966 | 1.9934 | | |
| 5 | 1.9946 | 1.9838 | ✓ | ✓ |
| 6 | 1.9711 | 1.9664 | ✓ | ✓ |
| 7 | 1.7502 | 1.1214 | ✓ | ✓ |
| 8 | 0.2559 | 0.8980 | ✓ | ✓ |
| 9 | 0.0344 | 0.0401 | ✓ | ✓ |

Root 0 NOONs in (0.02, 1.98): `1.9946  1.9711  1.7502  0.2559  0.0344`
Root 1 NOONs in (0.02, 1.98): `1.9838  1.9664  1.1214  0.8980  0.0401`

**Root 0 vs OptTS CASSCF NOONs (nat_occ, same (14e,9o)):**

| NO | Root 0 (DFT geom, non-conv) | OptTS geom | Δ |
|----|----------------------------|------------|---|
| 1 | 1.9996 | 1.9998 | −0.0002 |
| 2 | 1.9991 | 1.9997 | −0.0006 |
| 3 | 1.9985 | 1.9994 | −0.0009 |
| 4 | 1.9966 | 1.9993 | −0.0027 |
| 5 | 1.9946 | 1.9992 | −0.0046 |
| 6 | 1.9711 | 1.9756 | −0.0045 |
| 7 | 1.7502 | 1.3780 | +0.3722 |
| 8 | 0.2559 | 0.6232 | −0.3673 |
| 9 | 0.0344 | 0.0258 | +0.0086 |

**Orbital compositions:** Not available.

---

### rxn8837 — (18e, 11o)

**Convergence:** YES — 1-step SA-CASSCF converged in 110 macro iterations (1456 JK, 318 micro)

**Energies and gap:**

| Root | Energy (Ha) | ⟨S²⟩ |
|------|-------------|-------|
| Root 0 | −321.45971830 | 0.0000 |
| Root 1 | −321.36620544 | 0.0000 |
| Gap (Root1 − Root0) | +0.09351286 Ha | +2544.7 meV |

**CASCI NOONs (proxy for final-iteration NOONs of killed single-state run):**
```
1.9999  1.9999  1.9998  1.9997  1.9995  1.9990  1.9975  1.9964  1.9772  0.0266  0.0045
```

**State-specific NOONs (state-specific CASCI on SA orbitals), sorted descending, 4 dp:**

| NO | Root 0 | Root 1 | in (0.02,1.98) R0 | in (0.02,1.98) R1 |
|----|--------|--------|-------------------|-------------------|
| 1  | 1.9999 | 1.9997 | | |
| 2  | 1.9995 | 1.9990 | | |
| 3  | 1.9994 | 1.9983 | | |
| 4  | 1.9990 | 1.9975 | | |
| 5  | 1.9974 | 1.9971 | | |
| 6  | 1.9957 | 1.9964 | | |
| 7  | 1.9924 | 1.9852 | ✓ | ✓ |
| 8  | 1.9343 | 1.9467 | ✓ | ✓ |
| 9  | 1.6473 | 1.8091 | ✓ | ✓ |
| 10 | 0.3543 | 0.1875 | ✓ | ✓ |
| 11 | 0.0809 | 0.0836 | ✓ | ✓ |

Root 0 NOONs in (0.02, 1.98): `1.9924  1.9343  1.6473  0.3543  0.0809`
Root 1 NOONs in (0.02, 1.98): `1.9852  1.9467  1.8091  0.1875  0.0836`

**Root 0 vs OptTS CASSCF NOONs (nat_occ, same (18e,11o)):**

| NO | Root 0 (DFT geom) | OptTS geom | Δ |
|----|-------------------|------------|---|
| 1  | 1.9999 | 2.0000 | −0.0001 |
| 2  | 1.9995 | 2.0000 | −0.0005 |
| 3  | 1.9994 | 2.0000 | −0.0006 |
| 4  | 1.9990 | 2.0000 | −0.0010 |
| 5  | 1.9974 | 2.0000 | −0.0026 |
| 6  | 1.9957 | 1.9967 | −0.0010 |
| 7  | 1.9924 | 1.9872 | +0.0052 |
| 8  | 1.9343 | 1.9325 | +0.0019 |
| 9  | 1.6473 | 1.0897 | +0.5576 |
| 10 | 0.3543 | 0.9220 | −0.5677 |
| 11 | 0.0809 | 0.0730 | +0.0079 |

**Orbital compositions:** Not available.

---

## 26-Reaction Summary Table

NOONs in (0.02, 1.98) shown for Root 0 (SA cases) or converged state (single-state).
UNCONVERGED_AT_DFT_GEOMETRY: both single-state (killed, oscillating) and SA retry failed to converge.

| rxn | geom | converged | method | (n_e,n_o) | NOONs in (0.02, 1.98) | flags |
|-----|------|-----------|--------|-----------|----------------------|-------|
| rxn7949 | DFT-SP | True | single | (16e,10o) | 1.9385  1.9381  0.0674  0.0661 | |
| rxn8832 | DFT-SP | True | single | (16e,10o) | 1.9470  1.9352  0.0709  0.0589 | |
| rxn1320 | DFT-SP | True | single | (2e,2o) | 1.9135  0.0865 | |
| rxn4113 | DFT-SP | True | single | (16e,10o) | 1.9782  1.9111  0.1030  0.0264 | |
| rxn8885 | DFT-SP | True | single | (12e,9o) | 1.9679  1.9552  1.9313  0.0799  0.0402  0.0304 | |
| rxn7945 | DFT-SP | True | single | (14e,10o) | 1.9483  1.9337  1.9211  0.0841  0.0737  0.0616 | |
| rxn7937 | DFT-SP | True | single | (14e,10o) | 1.9461  1.9282  0.0855  0.0638 | |
| rxn6196 | DFT-SP | True | single | (14e,10o) | 1.9407  1.9335  1.9183  0.0916  0.0710  0.0587 | |
| rxn0346 | DFT-SP | True | single | (14e,9o) | 1.9558  1.6032  0.4064  0.0475 | |
| rxn1150 | DFT-SP | True | SA-2roots | (12e,8o) | Root0: 1.9758  1.9106  1.8766  0.1504  0.1034 | SA-conv@132 |
| rxn0896 | DFT-SP | False | SA-2roots | (14e,9o) | Root0: 1.9946  1.9711  1.7502  0.2559  0.0344 | UNCONVERGED_AT_DFT_GEOMETRY; SA stopped@150 |
| rxn4518 | DFT-SP | True | single | (14e,9o) | 1.9241  1.4631  0.5473  0.0716 | |
| rxn3107 | DFT-SP | True | single | (14e,8o) | 1.9352  0.0703 | |
| rxn8837 | DFT-SP | True | SA-2roots | (18e,11o) | Root0: 1.9924  1.9343  1.6473  0.3543  0.0809 | SA-conv@110 |
| rxn7060 | DFT-SP | True | single | (16e,11o) | 1.9756  1.9509  1.9404  0.0740  0.0665  0.0311 | |
| rxn5691 | DFT-SP | True | single | (16e,10o) | 1.9431  1.5572  0.4456  0.0590 | |
| rxn1283 | DFT-SP | True | single | (16e,9o) | 1.8900  0.1162 | |
| rxn8827 | DFT-SP | True | single | (16e,10o) | 1.9703  1.9307  0.0769  0.0290 | |
| rxn4522 | DFT-SP | True | single | (14e,9o) | 1.9411  1.5620  0.4510  0.0636 | |
| rxn7936 | DFT-SP | True | single | (18e,11o) | 1.9602  1.8592  0.1588  0.0422 | |
| rxn1147 | DFT-SP | True | single | (14e,8o) | 1.7941  0.2122 | |
| rxn0894 | DFT-SP | True | single | (12e,9o) | 1.9698  1.9067  1.5475  0.4635  0.0881  0.0340 | |
| rxn0101 | DFT-SP | True | single | (14e,9o) | 1.9569  1.9442  0.0594  0.0515 | |
| rxn10005 | DFT-SP | PENDING | PENDING | (20e,13o) | — | PENDING (job 10674827_3) |
| rxn10054 | DFT-SP | True | single | (16e,10o) | 1.9413  1.9403  0.0643  0.0631 | |
| rxn7957 | DFT-SP | True | single | (14e,9o) | 1.9266  1.7681  0.2400  0.0782 | |
