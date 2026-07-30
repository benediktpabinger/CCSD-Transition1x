# FOD Density Cube Analysis

## Overview

For all 23 MR-benchmark reactions, PySCF PBE/def2-SVP with Fermi smearing (T=5000 K) was
re-run at the ORCA NEB TS geometry. In addition to the scalar n_FOD already used for
screening, the calculation was extended to write the FOD density as a volumetric cube file:

```
ρ_FOD(r) = Σᵢ |nᵢ − n⁰ᵢ| |φᵢ(r)|²
```

where `nᵢ` is the Fermi-smeared MO occupation and `n⁰ᵢ` is the integer reference (2 for
occupied, 0 for virtual). The cube files were generated with an 80×80×80 grid
(~6.7 MB each, ~0.2 Å resolution). All cube files are in `~/fod_cube_results/` on the
cluster.

The analysis below makes no assumptions about electronic character. The geometry (R→P
displacement, bond length changes) is used to define the "reactive region", and the FOD
density is then compared to that region.

---

## Analysis Method

### Geometry: reactive region definition

Reactant (`reactant.xyz`) and product (`product.xyz`) geometries from the ORCA NEB are
Kabsch-aligned to remove rigid-body motion. Per-atom displacement is computed as
`|P_aligned[i] − R[i]|`. The reactive region is defined as the union of:

- Top 2–3 most-displaced atoms
- Both atoms of the bond with the largest length change (R→P)

A 2.5 Å sphere around this union captures the local bonding region.

### FOD integration

- **Total FOD**: `Σ_voxels ρ_FOD(r) × dV_bohr` — matches the scalar n_FOD from screening
  to within rounding.
- **Sphere FOD**: same sum restricted to voxels within 2.5 Å of any reactive atom.
- **Atom-local FOD**: each voxel is assigned to its nearest atom; FOD is summed per atom.
- **FOD maximum**: voxel with the highest density value; its position is reported relative
  to the reactive atoms from step 1.

---

## Results per Reaction

### Sphere integration (radius = 2.5 Å around reactive atoms)

| Reaction | n_FOD | FOD total | FOD sphere | Fraction | Biggest bond change |
|----------|------:|----------:|-----------:|---------:|---------------------|
| rxn7949  | 1.146 | 1.145 | 1.128 | **98%** | C–C Δ1.07 Å |
| rxn8832  | 1.000 | 0.999 | 0.974 | **98%** | C–C Δ1.08 Å |
| rxn1320  | 0.968 | 0.967 | 0.842 | 87% | C–H Δ1.80 Å |
| rxn4113  | 0.960 | 0.957 | 0.933 | **98%** | O–C Δ2.05 Å |
| rxn8885  | 0.923 | 0.921 | 0.783 | 85% | C–O Δ1.25 Å |
| rxn7945  | 0.903 | 0.901 | 0.884 | **98%** | C–N Δ2.10 Å |
| rxn7937  | 0.877 | 0.875 | 0.840 | 96% | C–N Δ2.04 Å |
| rxn6196  | 0.869 | 0.867 | 0.685 | 79% | C–C Δ2.65 Å |
| rxn0346  | 0.847 | 0.846 | 0.807 | 95% | C–H Δ1.69 Å |
| rxn1150  | 0.847 | 0.846 | 0.665 | 79% | N–H Δ2.08 Å |
| rxn0896  | 0.840 | 0.840 | 0.827 | **98%** | N–O Δ2.36 Å |
| rxn4518  | 0.833 | 0.833 | 0.811 | 97% | N–O Δ1.90 Å |
| rxn3107  | 0.801 | 0.800 | 0.773 | 97% | C–O Δ1.24 Å |
| rxn8837  | 0.798 | 0.795 | 0.778 | **98%** | N–C Δ2.11 Å |
| rxn7060  | 0.788 | 0.782 | 0.770 | **98%** | O–C Δ3.89 Å |
| rxn8827  | 0.760 | 0.756 | 0.742 | **98%** | N–C Δ2.44 Å |
| rxn4522  | 0.731 | 0.730 | 0.725 | **99%** | O–C Δ1.97 Å |
| rxn7936  | 0.727 | 0.726 | 0.643 | 89% | O–N Δ3.51 Å |
| rxn1147  | 0.725 | 0.725 | 0.720 | **99%** | C–C Δ1.77 Å |
| rxn0101  | 0.713 | 0.713 | 0.696 | **98%** | C–O Δ2.07 Å |
| rxn10005 | 0.695 | 0.694 | 0.664 | 96% | C–N Δ1.89 Å |
| rxn10054 | 0.695 | 0.694 | 0.672 | 97% | C–O Δ2.65 Å |
| rxn7957  | 0.684 | 0.683 | 0.654 | 96% | C–H Δ3.24 Å |

16 of 23 reactions capture ≥96% of total FOD within 2.5 Å of the geometrically reactive
atoms. Four reactions (rxn6196, rxn1150, rxn8885, rxn7936) have 79–89%, meaning a
non-trivial fraction of FOD density lies outside the immediate bonding region of the
most-displaced atoms.

---

### Atom-local FOD (top 4 atoms, nearest-voxel weighting)

| Reaction | #1 atom | FOD | #2 atom | FOD | #3 atom | FOD | #4 atom | FOD |
|----------|---------|----:|---------|----:|---------|----:|---------|----:|
| rxn7949  | C(3)  | 0.272 | O(0)  | 0.236 | C(2)  | 0.231 | N(6)  | 0.111 |
| rxn8832  | C(4)  | 0.269 | N(0)  | 0.218 | C(6)  | 0.150 | O(3)  | 0.113 |
| rxn1320  | C(2)  | 0.226 | C(1)  | 0.205 | N(5)  | 0.122 | O(0)  | 0.114 |
| rxn4113  | C(3)  | 0.245 | O(0)  | 0.234 | N(2)  | 0.208 | O(5)  | 0.101 |
| rxn8885  | N(6)  | 0.289 | C(1)  | 0.216 | O(2)  | 0.158 | C(4)  | 0.112 |
| rxn7945  | N(6)  | 0.270 | C(4)  | 0.215 | C(2)  | 0.166 | O(0)  | 0.061 |
| rxn7937  | N(6)  | 0.266 | C(4)  | 0.203 | C(3)  | 0.157 | O(0)  | 0.047 |
| rxn6196  | C(2)  | 0.333 | C(4)  | 0.162 | N(6)  | 0.154 | O(0)  | 0.045 |
| rxn0346  | C(5)  | 0.282 | N(1)  | 0.223 | O(4)  | 0.099 | O(0)  | 0.084 |
| rxn1150  | **N(3)** | **0.405** | O(5) | 0.209 | O(0) | 0.053 | C(2) | 0.043 |
| rxn0896  | O(5)  | 0.282 | N(2)  | 0.220 | C(1)  | 0.107 | C(4)  | 0.086 |
| rxn4518  | N(0)  | 0.296 | C(1)  | 0.194 | O(5)  | 0.115 | O(3)  | 0.057 |
| rxn3107  | N(5)  | 0.267 | C(2)  | 0.237 | O(3)  | 0.139 | C(4)  | 0.037 |
| rxn8837  | C(4)  | 0.203 | C(1)  | 0.174 | C(6)  | 0.104 | O(3)  | 0.085 |
| rxn7060  | C(1)  | 0.286 | N(6)  | 0.218 | C(3)  | 0.111 | O(0)  | 0.033 |
| rxn8827  | C(4)  | 0.192 | C(1)  | 0.139 | C(5)  | 0.124 | O(3)  | 0.093 |
| rxn4522  | N(0)  | 0.212 | C(1)  | 0.168 | O(3)  | 0.134 | O(5)  | 0.061 |
| rxn7936  | O(0)  | 0.274 | C(1)  | 0.146 | N(6)  | 0.118 | C(5)  | 0.084 |
| rxn1147  | C(2)  | 0.178 | C(1)  | 0.145 | O(5)  | 0.133 | C(4)  | 0.067 |
| rxn0101  | O(5)  | 0.186 | C(4)  | 0.180 | N(0)  | 0.126 | O(3)  | 0.096 |
| rxn10005 | O(3)  | 0.205 | N(2)  | 0.175 | C(4)  | 0.118 | N(6)  | 0.065 |
| rxn10054 | C(6)  | 0.204 | C(4)  | 0.177 | O(3)  | 0.123 | C(2)  | 0.043 |
| rxn7957  | N(6)  | 0.225 | C(1)  | 0.153 | O(0)  | 0.102 | C(4)  | 0.058 |

**Notes:**
- Top atom carries 20–48% of total FOD (units: electrons). Top 3 atoms together cover
  55–85%.
- H atoms occasionally appear but always rank below the heavy-atom top 3.
- rxn1150 has the most concentrated single-atom FOD: N(3) alone carries 48% of total.
- rxn6196: C(2) alone carries 38%.
- rxn7949 is the most evenly distributed: top 3 atoms (C, O, C) share nearly equal weight.

---

### FOD maximum position relative to displaced atoms

The voxel with the highest density value is reported. Its distance to the top-displaced
atoms (from the Kabsch analysis) and to the nearest atom of any kind is listed.

| Reaction | Most displaced atom | FOD max nearest atom | Dist max→nearest | Dist max→top-disp | Relationship |
|----------|--------------------:|---------------------:|-----------------:|------------------:|--------------|
| rxn7949  | H(8) Δ2.57 Å | C(1) | 1.49 Å | 2.96 Å (H) | max on C adjacent to bond, not on displaced H |
| rxn8832  | H(8) Δ1.34 Å | N(0) | 0.59 Å | 4.91 Å | max on N, 4.9 Å from displaced H |
| rxn1320  | H(6) Δ1.98 Å | C(2) | 0.32 Å | 2.68 Å (H) | max on C end of C–H bond being broken |
| rxn4113  | O(0) Δ2.66 Å | C(1) | 1.25 Å | 1.76 Å (O) | max between C and O, 1.76 Å from displaced O |
| rxn8885  | O(2) Δ1.51 Å | N(6) | 0.30 Å | 3.09 Å (O) | max on N, 3.1 Å from displaced O |
| rxn7945  | O(0) Δ1.58 Å | C(5) | 1.34 Å | 3.35 Å (O) | max on C not displaced; N(6) (2nd disp.) 1.79 Å |
| rxn7937  | N(6) Δ2.96 Å | C(5) | 1.45 Å | 1.92 Å (N) | max on C adjacent to N; N is 1.9 Å away |
| rxn6196  | C(2) Δ2.33 Å | C(2) | 1.08 Å | 1.08 Å | max on top-displaced atom |
| rxn0346  | H(9) Δ1.89 Å | N(1) | 0.25 Å | 2.95 Å (H) | max on N, 3.0 Å from displaced H |
| rxn1150  | H(10) Δ1.83 Å | N(3) | 1.04 Å | 1.15 Å (H) | max on N, H is 1.15 Å away (breaking N–H) |
| rxn0896  | O(5) Δ2.29 Å | O(3) | 1.74 Å | 2.20 Å (O) | max on O(3) not O(5); both in reactive region |
| rxn4518  | O(5) Δ1.66 Å | H(6) | 0.71 Å | 2.91 Å (O) | max on H near N; displaced O is 2.9 Å away |
| rxn3107  | C(4) Δ1.33 Å | N(5) | 0.50 Å | 0.66 Å (C) | max between N and C, both displaced |
| rxn8837  | N(0) Δ1.41 Å | C(1) | 0.31 Å | 1.15 Å (N) | max on C end of forming bond; N 1.15 Å away |
| rxn7060  | O(0) Δ4.81 Å | C(1) | 0.24 Å | 1.31 Å (O) | max on C end of breaking O–C; O travels 4.8 Å |
| rxn8827  | H(9) Δ1.77 Å | C(1) | 1.24 Å | 3.53 Å (H) | max on C/N region; displaced H is 3.5 Å away |
| rxn4522  | O(3) Δ2.32 Å | H(6) | 0.59 Å | 2.83 Å (O) | max on H near N; displaced O(3) is 2.8 Å |
| rxn7936  | O(0) Δ3.43 Å | H(7) | 1.84 Å | 2.41 Å (O) | max in H/C region; displaced O is 2.4 Å away |
| rxn1147  | H(9) Δ2.19 Å | O(5) | 0.56 Å | 3.76 Å (H) | max on O(5) which forms new bond; H travels far |
| rxn0101  | O(5) Δ2.74 Å | C(1) | 1.59 Å | 2.00 Å (O) | max on C adjacent to displaced O and N |
| rxn10005 | C(4) Δ1.73 Å | O(3) | 0.75 Å | 1.61 Å (C) | max on O of breaking C–O; C is 1.6 Å away |
| rxn10054 | O(3) Δ2.52 Å | O(3) | 0.17 Å | 0.17 Å | max essentially on top-displaced atom |
| rxn7957  | H(7) Δ2.49 Å | N(6) | 0.65 Å | 3.31 Å (H) | max on N; displaced H is 3.3 Å away |

---

## Summary of Spatial Relationships

### Sphere capture (2.5 Å around reactive atoms)

| Category | Count | Reactions |
|----------|------:|-----------|
| ≥96% captured | 16/23 | most reactions |
| 87–95% | 3/23 | rxn1320, rxn0346, rxn7936 |
| 79–85% | 4/23 | rxn6196, rxn1150, rxn8885, rxn8885 |

For the 4 low-capture reactions, FOD density extends into regions beyond the immediate
bonding atoms — either into adjacent atoms within the ring/chain, or into a second
independent site.

### FOD maximum vs. most-displaced atom

| Relationship | Count | Reactions |
|-------------|------:|-----------|
| Max on or within 1.2 Å of top-displaced heavy atom | 5 | rxn6196, rxn1150, rxn3107, rxn10054, rxn7957 |
| Max on other atom of the largest changing bond | 4 | rxn1320, rxn7060, rxn8837, rxn10005 |
| Max 1.2–2.5 Å from top-displaced | 5 | rxn4113, rxn7937, rxn8832 (to C), rxn7936, rxn0101 |
| Top-displaced is H; max on nearby heavy atom (>2.5 Å from H) | 9 | rxn7949, rxn8832, rxn0346, rxn4518, rxn8827, rxn4522, rxn1147, rxn7957, rxn8885 |

When H is the most-displaced atom (H-transfer or H abstraction), the FOD density maximum
is consistently on a heavy atom (C, N, or O) within the same bond-breaking/forming region,
not on the H itself. In every case in this dataset, the FOD maximum sits within 2.2 Å of
at least one atom involved in the largest bond-length change R→P.

### Size vs. FOD character

Across all 279 screened reactions (from `fod_ranking.json`):

- Pearson r (n_atoms vs. n_FOD) = **0.014** — essentially no correlation.
- The 14-atom molecules (largest in the dataset) have the lowest mean n_FOD (0.063).
- High-FOD reactions are 11–12 atoms because that is the dominant size class in
  Transition1x, not because size drives MR character.
- The top-26 benchmark reactions are all 10–12 atoms: 1× 10-atom, 12× 11-atom, 13× 12-atom.

---

## Files

| File | Contents |
|------|----------|
| `~/fod_cube_results/{rxn}_fod.cube` | FOD density cube, 80³ grid, PBE/def2-SVP T=5000K |
| `~/fod_cube_results/fod_cube_summary.json` | Per-reaction n_FOD scalar from cube integration |
| `fod_geo_analysis.json` | Full per-reaction output: displaced atoms, bond changes, sphere integrals, atom-local FOD, max positions |
| `pipeline/fod_geo_analysis.py` | Analysis script (Kabsch alignment, sphere integration, atom-local FOD) |
| `pipeline/fod_cube.py` | Cube generation script (PySCF RKS + cubegen) |

Cube files are visualizable in VESTA, VMD, or Avogadro. Recommended isosurface value:
0.001–0.005 e/Bohr³.
