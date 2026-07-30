# Chapter 2 — Notes

## Tishchenko M diagnostic — all 26 benchmark reactions

**Formula:** M = ½ [2 − n(MCDONO) + Σⱼ |n(SOMOⱼ) − 1| + n(MCUNO)] (Tishchenko, Zheng, Truhlar, JCTC 2008, 4, 1208, Eq. 1). MCDONO = most-changed doubly-occupied NO (smallest n among orbitals with n > 1.5 in dominant config); MCUNO = most-changed unoccupied NO (largest n among orbitals with n < 0.5); SOMOs = orbitals with 0.5 ≤ n ≤ 1.5 (singly occupied in dominant config). M = 0 for single-reference; M > 0.05 indicates significant MR character. No reaction exceeds M > 1.

Note: an earlier version of this table used M = ½ · Σᵢ min(nᵢ, 2 − nᵢ) summed over all active orbitals; that is a different (non-Tishchenko) quantity and has been replaced here.

**Source:** CASSCF(AVAS)+NEVPT2 TS natural orbital occupation numbers from `~/nevpt2_optts_results/{rxn}_avas/nevpt2_optts_results.json` (key: `ts.nat_occ`). For CASSCF-failed reactions (rxn5691, rxn1283, rxn0894), the JSON exists but the TS geometry is from a partially-optimised CASSCF trajectory, not a converged saddle point — M is computed but marked as unreliable. T1 = CCSD T1 diagnostic at the ORCA NEB TS geometry. N_FOD from Fermi-orbital-descriptor screening at T = 5000 K. SOMO column: number of natural orbitals with 0.5 ≤ n ≤ 1.5 (biradical character at TS). Table sorted by M descending.

```
rxn          M(TS)  SOMOs   N_FOD  FOD_rk      T1          CAS   DFT-geom  note
----------------------------------------------------------------------------------
rxn7060      0.433      2   0.788      15   0.0436   (16e,11o)        yes
rxn0896      0.403      2   0.840      11   0.0257    (14e,9o)      maybe
rxn10005     0.301      0   0.695      24   0.0190   (20e,13o)        yes
rxn8832      0.289      2   1.000       2   0.0610   (16e,10o)        yes
rxn7936      0.277      0   0.727      20   0.0275   (18e,11o)        yes
rxn7957      0.258      0   0.684      26   0.0360    (14e,9o)        yes
rxn0346      0.256      0   0.847       9   0.0345    (14e,9o)      maybe
rxn1147      0.248      0   0.725      21   0.0267    (14e,8o)      maybe
rxn8837      0.154      2   0.798      14   0.0361   (18e,11o)        yes
rxn6196      0.127      0   0.869       8   0.0156   (14e,10o)        yes
rxn4522      0.093      2   0.731      19   0.0315    (14e,9o)      maybe
rxn5691      0.162      2   0.778      16      —     (16e,10o)     failed  optts_unconverged
rxn1283      0.134      0   0.769      17      —      (16e,9o)     failed  optts_unconverged
rxn7945      0.081      0   0.903       6   0.0265   (14e,10o)        yes
rxn7949      0.066      0   1.146       1   0.0562   (16e,10o)        yes
rxn4113      0.077      0   0.960       4   0.0195   (16e,10o)        yes
rxn8885      0.078      0   0.923       5   0.0437    (12e,9o)        yes
rxn1320      0.078      0   0.968       3   0.0432     (2e,2o)        yes
rxn7937      0.075      0   0.877       7   0.0246   (14e,10o)        yes
rxn8827      0.075      0   0.760      18   0.0321   (16e,10o)        yes
rxn1150      0.068      0   0.847      10   0.0322    (12e,8o)        yes
rxn3107      0.068      0   0.801      13   0.0454    (14e,8o)        yes
rxn4518      0.060      2   0.833      12   0.0222    (14e,9o)      maybe
rxn10054     0.059      0   0.695      25   0.0260   (16e,10o)        yes
rxn0101      0.053      0   0.713      23   0.0187    (14e,9o)        yes
rxn0894      0.087      2   0.716      22      —      (12e,9o)     failed  optts_unconverged
```

SOMO pairs (biradical dominant configuration at TS):
- rxn7060: 1.3597, 0.6434 — strongly asymmetric, significant biradical
- rxn0896: 1.3780, 0.6232 — strongly asymmetric, significant biradical
- rxn8832: 1.2124, 0.7991 — moderately asymmetric biradical
- rxn8837: 1.0897, 0.9220 — near-symmetric singlet biradical
- rxn5691: 1.1055, 0.8950 — near-symmetric [UNCONVERGED]
- rxn4522: 1.0435, 0.9565 — nearly symmetric, M low because SOMOs close to 1.0
- rxn4518: 1.0125, 0.9907 — essentially exact singlet biradical, SOMOs at n≈1
- rxn0894: 1.0024, 0.9999 — true open-shell singlet geometry [UNCONVERGED]

T1 missing for rxn0894, rxn5691, rxn1283 (CCSD(T) not available). M for CASSCF-failed reactions is from the last available CASSCF density, not a converged saddle point.

**rxn1320 consistency check:** Backup directory `rxn1320_avas_backup_cas22` contains identical NOONs [1.9220, 0.0780] from the same geometry file. No alternative CAS output with different NOONs found.

**CAS(10,16) geom_A vs AVAS comparison — rxn7949 and rxn8832 (pending):**
When the CAS(10,16) single-points on geom_A complete, M_geomA will be added alongside M_AVAS. Current M_AVAS = 0.066 (rxn7949) and 0.289 (rxn8832).

---

## Computational details

### Reaction selection

The benchmark reactions were selected by screening all 279 reactions in the Transition1x test set for multireference (MR) character using the fractional occupation number weighted density (FOD) at T = 5000 K. FOD integrates the density of orbitals with fractional occupancies under Fermi smearing; a larger integrated FOD (n_FOD) indicates stronger static electron correlation. The top 26 reactions by n_FOD form the High MR benchmark used in the main table below. A separate 30-reaction set (top 10 / next 10 / next 10 by FOD = High / Mid / Low) was used for the Mid and Low MR tables. All reactions are from the Transition1x test set and were never used in training any of the models evaluated here.

### Models

| Label | Model | Training functional | Notes |
|-------|-------|---------------------|-------|
| T1x | Transition1x TS geometry | wB97X-D3/6-31G(d) | Explicitly optimised TS from Transition1x.h5, not a NEB image |
| MACE | MACE-T1x | wB97X-D3/6-31G(d) | Graph neural network trained on full Transition1x train set |
| Delta | MACE+delta fw2.0 | wB97M-V/def2-TZVP (correction only) | Frozen MACE backbone + NonLinearReadoutBlock MLP head trained to predict δ = E(wB97M-V) − E(wB97X-D3); head trained on 80 592 geometries from 4 997 Transition1x train reactions |
| UMA-S | Universal Model for Atoms (small) | Open Molecules 2025 | Meta/FAIRChem universal model, omol task |
| UMA-M | Universal Model for Atoms (medium) | Open Molecules 2025 | Larger variant of UMA-S |
| eSEN | eSEN-SM | Open Molecules 2025 | Meta equivariant network, same training corpus as UMA |

### NEB protocol

All models used the same protocol: 10-image band initialised from the Transition1x wB97X-D3 NEB path (reactant + 8 interior images + product), BFGS endpoint relaxation (fmax = 0.05 eV/Å), then plain NEB → CI-NEB (improved tangent), fmax = 0.05 eV/Å, max 500 steps each. The highest-energy image of the converged CI-NEB is taken as the model TS. The ORCA reference followed the same protocol at wB97M-V/def2-TZVP. wB97M-V was chosen as the reference functional because it includes non-local VV10 dispersion and is more accurate than the wB97X-D3/6-31G(d) training functional, providing a meaningful upper bound on DFT geometry quality.

### RMSD metric

Kabsch RMSD: centroids subtracted, optimal rotation found by SVD with determinant correction for improper rotations (reflections), RMSD computed over all atoms. Lower is better; 0.3 Å is used as a practical threshold separating acceptable from poor geometry agreement.

### DFT geom reasonable column

Indicates whether the ORCA wB97M-V NEB TS can be trusted as a reference for assessing MLIP geometry error. Classification is based on comparison to the CASSCF(AVAS)+NEVPT2 OptTS geometry (see `active_space_quality_analysis.md`):
- **yes** — ORCA and CASSCF locate the same saddle point (RMSD < 0.3 Å, consistent reaction coordinate). MLIP errors vs ORCA are interpretable as genuine geometry errors.
- **maybe** — moderate RMSD or ambiguous evidence; ORCA and CASSCF may be on slightly different parts of the surface.
- **—** — CASSCF OptTS did not converge (rxn5691, rxn1283, rxn0894). No reliable multireference geometry is available, so the DFT reference cannot be independently validated. These reactions are kept in the table because the RMSD vs ORCA is still a well-defined and internally consistent metric — it measures how well each MLIP reproduces the DFT NEB TS, regardless of whether that TS is the "true" saddle point. The CASSCF failure is itself diagnostic: it indicates particularly pathological MR character that makes even the CASSCF geometry optimisation ill-conditioned.

---

## TS geometry accuracy vs ORCA wB97M-V/def2-TZVP NEB reference

All-atom Kabsch RMSD (Å) between each model's NEB transition state and the ORCA wB97M-V/def2-TZVP reference, for all 26 top-FOD benchmark reactions. Delta = MACE+delta fw2.0. See Computational details above for model descriptions, NEB protocol, and column definitions.

```
rxn           T1x    MACE    Delta   UMA-S   UMA-M    eSEN   | rank | DFT geom
-----------------------------------------------------------------------------------
rxn7949      0.005   0.168   0.242   0.162   0.201   0.198   |  1   | yes
rxn8832      0.011   0.109   0.120   0.145   0.181   0.166   |  2   | yes
rxn1320      0.006   0.071   0.052   0.362   0.361   0.366   |  3   | yes
rxn4113      0.010   0.012   0.210   0.014   0.740   0.017   |  4   | yes
rxn8885      0.009   0.009   0.025   0.492   0.066   1.411   |  5   | yes
rxn7945      0.017   0.095   0.328   0.044   0.032   0.427   |  6   | yes
rxn7937      0.020   0.084   0.153   0.043   0.037   0.049   |  7   | yes
rxn6196      0.014   0.038   0.109   0.085   0.093   0.093   |  8   | yes
rxn0346      0.004   0.006   0.026   0.141   0.167   0.134   |  9   | maybe
rxn1150      0.017   0.016   0.077   0.009   0.007   0.007   | 10   | yes
rxn0896      0.005   0.098   0.116   0.020   0.012   0.019   | 11   | maybe
rxn4518      0.005   0.213   0.679   0.817   0.807   0.848   | 12   | maybe
rxn3107      0.006   0.167   0.140   0.027   0.028   0.035   | 13   | yes
rxn8837      0.006   0.196   0.259   2.707   1.812   1.439   | 14   | yes
rxn7060      0.014   0.077   0.086   0.044   0.013   0.054   | 15   | yes
rxn5691      0.005   0.007   0.027   0.301   0.207   0.297   | 16 (CASSCF failed) | —
rxn1283      0.010   0.018   0.049   0.301   0.261   0.315   | 17 (CASSCF failed) | —
rxn8827      0.009   0.087   0.407   0.211   0.236   0.262   | 18   | yes
rxn4522      0.005   0.414   0.261   0.317   0.311   0.312   | 19   | maybe
rxn7936      0.005   0.074   0.063   0.002   0.002   0.004   | 20   | yes
rxn1147      0.012   0.131   0.336   0.339   0.322   0.330   | 21   | maybe
rxn0894      0.005   2.820   0.594   0.530   2.366   2.482   | 22 (CASSCF failed) | —
rxn0101      0.006   0.110   0.171   0.074   0.056   0.058   | 23   | yes
rxn10005     0.014   0.040   0.050   0.004   0.003   0.004   | 24   | yes
rxn10054     0.007   0.007   0.027   0.022   0.015   0.014   | 25   | yes
rxn7957      0.012   0.117   0.164   0.206   0.214   0.205   | 26   | yes
-----------------------------------------------------------------------------------
mean         0.009   0.199   0.183   0.285   0.329   0.367
median       0.008   0.086   0.130   0.143   0.174   0.182
n(>0.3 Å)      0       2       5       9       7       9    (of 26)
--- DFT geom = yes (n=18) ---------------------------------------------------------
mean         0.010   0.082   0.149   0.259   0.228   0.267
median       0.010   0.081   0.130   0.059   0.061   0.076
n(>0.3 Å)      0       0       2       3       3       4
--- DFT geom = maybe (n=5) --------------------------------------------------------
mean         0.006   0.172   0.284   0.327   0.324   0.329
median       0.005   0.131   0.261   0.317   0.311   0.312
n(>0.3 Å)      0       1       2       3       3       3
--- CASSCF failed (n=3) -----------------------------------------------------------
mean         0.007   0.948   0.223   0.377   0.945   1.031
n(>0.3 Å)      0       1       1       3       1       2
```

Reactions with RMSD > 0.3 Å vs DFT reference:
- **T1x** (0): all < 0.020 Å — wB97X-D3/6-31G(d) vs wB97M-V/def2-TZVP, same saddle point found
- **MACE** (2): rxn4522, rxn0894
- **Delta** (5): rxn7945, rxn4518, rxn8827, rxn1147, rxn0894
- **UMA-S** (9): rxn1320, rxn8885, rxn4518, rxn8837, rxn5691, rxn1283, rxn4522, rxn1147, rxn0894
- **UMA-M** (7): rxn1320, rxn4113, rxn4518, rxn8837, rxn4522, rxn1147, rxn0894
- **eSEN** (9): rxn1320, rxn8885, rxn7945, rxn4518, rxn8837, rxn1283, rxn4522, rxn1147, rxn0894

---

## TS geometry accuracy — Mid and Low MR reactions (n=20)

Same metric as above (all-atom Kabsch RMSD vs ORCA wB97M-V/def2-TZVP NEB TS). Mid MR = FOD ranks 11–20 of the 30-reaction benchmark; Low MR = ranks 21–30. T1x from explicit `transition_state` subgroup in Transition1x.h5 (wB97X-D3/6-31G(d)).

```
rxn           T1x    MACE    Delta   UMA-S   UMA-M    eSEN   | group
---------------------------------------------------------------------------
rxn0896      0.005   0.098   0.116   0.020   0.012   0.019   | Mid
rxn1154      0.023   0.276   0.214   0.143   0.130   0.093   | Mid
rxn5690      0.086   0.060   0.102   0.166   0.168   0.178   | Mid
rxn4513      0.007   0.008   0.023   0.002   0.002   0.002   | Mid
rxn7955      0.005   0.007   0.015   0.002   0.001   0.001   | Mid
rxn4519      0.010   0.089   0.132   0.014   0.008   0.014   | Mid
rxn4500      0.008   0.357   0.478   0.001   0.001   0.001   | Mid
rxn2553      0.010   0.010   0.051   0.001   0.001   0.001   | Mid
rxn8829      0.005   0.006   0.009   0.003   0.002   0.002   | Mid
rxn1155      0.003   0.008   0.016   0.006   0.005   0.005   | Mid
---------------------------------------------------------------------------
rxn9246      0.009   0.009   0.026   0.004   0.005   0.006   | Low
rxn4498      0.008   0.026   0.038   0.003   0.002   0.003   | Low
rxn1061      0.006   0.018   0.040   0.016   0.019   0.021   | Low
rxn4003      0.004   0.040   0.060   0.019   0.019   0.022   | Low
rxn4004      0.004   0.030   0.079   0.009   0.009   0.016   | Low
rxn4063      0.003   0.003   0.099   0.000   0.000   0.001   | Low
rxn4114      0.003   0.003   0.005   0.001   0.000   0.001   | Low
rxn4060      0.003   0.015   0.066   0.012   0.009   0.011   | Low
rxn1961      0.003   0.003   0.018   0.000   0.001   0.001   | Low
rxn1962      0.006   0.006   0.096   0.002   0.002   0.002   | Low
---------------------------------------------------------------------------
mean Mid     0.016   0.092   0.116   0.036   0.033   0.032
mean Low     0.005   0.015   0.053   0.007   0.007   0.008
mean All 20  0.011   0.054   0.084   0.021   0.020   0.020
n(>0.3 Å)      0       1       1       0       0       0    (of 20)
```

Reactions with RMSD > 0.3 Å:
- **MACE** (1): rxn4500 = 0.357
- **Delta** (1): rxn4500 = 0.478

Notes:
- rxn5690 T1x = 0.086 Å — the only case where the wB97X-D3 and wB97M-V geometries differ noticeably; worth checking.
- rxn4500: MACE and Delta both fail (>0.3 Å) while UMA-S/M and eSEN are essentially perfect (0.001 Å). Isolated failure of the MACE backbone on this reaction.
- Low MR is nearly solved by all models: mean RMSD 0.003–0.053 Å, no outliers except Delta on individual reactions.

---

## Reactive-coordinate geometry analysis

**Subset:** 18 reactions from the 26-reaction High MR benchmark (FOD ranks 1–8, 10, 13–15, 18, 20, 23–26). Script: `~/_rxn_coord_full.py` on the DTU cluster.

**Bonding criterion (Step 1):** A pair (i, j) is bonded if distance < 1.3 × (cov_radius_i + cov_radius_j). Applied to both R and P; the union of bonded pairs from both is ranked by |d_P − d_R|. Top 2 pairs define the reactive atom set. Reactive set is 3 atoms for 17/18 reactions; 4 atoms for rxn10054 (two bonds sharing no atom).

**Bond-length errors (Step 2):** Computed directly from positions without alignment. Signed convention: model − reference.

**RMSD variants (Step 3):** (a) all-atom Kabsch, (b) heavy-atom Kabsch, (c) reactive-subset Kabsch aligned on the subset only.

---

### Step 1 — Reactive bonds (top 2 by |d_P − d_R|)

```
rxn          rank  bond_1              d_R      d_P    delta  |  bond_2              d_R      d_P    delta
----------------------------------------------------------------------------------------------------------
rxn7949         1  C3-C5           2.5384   1.4658  -1.0726  |  C4-C5           1.4425   2.4605  +1.0180
rxn8832         2  C1-C6           2.5583   1.4817  -1.0766  |  C1-C2           1.4687   2.4675  +0.9988
rxn1320         3  C2-H6           2.8880   1.0874  -1.8006  |  O0-H6           0.9634   2.5609  +1.5975
rxn4113         4  O0-C3           3.4760   1.4300  -2.0460  |  N2-C3           1.4497   3.4830  +2.0333
rxn8885         5  C1-O2           1.4152   2.6668  +1.2516  |  C1-N6           2.6360   1.5697  -1.0663
rxn7945         6  C2-N6           3.6123   1.5167  -2.0956  |  C2-C4           1.5214   2.3815  +0.8601
rxn7937         7  C3-N6           3.5736   1.5286  -2.0450  |  C3-C4           1.4966   2.4072  +0.9105
rxn6196         8  C2-C5           1.4685   4.1226  +2.6541  |  C2-H10          1.0925   3.4240  +2.3315
rxn1150        10  N3-H10          1.0069   3.0847  +2.0777  |  C2-N3           1.4687   3.2044  +1.7357
rxn3107        13  C2-O3           1.4063   2.6469  +1.2405  |  C2-N5           2.6161   1.5645  -1.0516
rxn8837        14  N0-C6           3.5763   1.4643  -2.1120  |  C4-C6           1.5065   2.3386  +0.8321
rxn7060        15  O0-C1           1.1902   5.0801  +3.8899  |  O0-C5           4.4813   1.1686  -3.3127
rxn8827        18  N0-C5           3.8904   1.4549  -2.4355  |  C4-C5           1.4844   2.5742  +1.0898
rxn7936        20  O0-N6           4.7223   1.2087  -3.5136  |  O0-C1           1.2002   2.8647  +1.6646
rxn0101        23  C4-O5           1.4274   3.4939  +2.0665  |  N0-C4           3.4732   1.4442  -2.0290
rxn10005       24  C4-N6           1.3885   3.2730  +1.8845  |  O3-C4           1.3546   3.2288  +1.8741
rxn10054       25  C2-O3           1.4423   4.0974  +2.6551  |  C4-C6           1.5065   2.5160  +1.0094
rxn7957        26  C1-H7           1.1075   4.3461  +3.2386  |  C5-H7           2.7028   1.0804  -1.6224
```

### Step 4 — Atom counts

```
rxn          rank  n_atoms  n_heavy  n_react
--------------------------------------------
rxn7949         1       12        7        3
rxn8832         2       12        7        3
rxn1320         3       11        6        3
rxn4113         4       11        6        3
rxn8885         5       12        7        3
rxn7945         6       12        7        3
rxn7937         7       12        7        3
rxn6196         8       12        7        3
rxn1150        10       11        6        3
rxn3107        13       11        6        3
rxn8837        14       12        7        3
rxn7060        15       12        7        3
rxn8827        18       12        7        3
rxn7936        20       12        7        3
rxn0101        23       11        6        3
rxn10005       24       10        7        3
rxn10054       25       12        7        4
rxn7957        26       12        7        3
```

---

### Step 2 — Bond-length errors at TS (signed, model − reference, Å)

#### Bond 1

```
rxn          rank       MACE      Delta      UMA-S      UMA-M       eSEN
------------------------------------------------------------------------
rxn7949         1    +0.1132    +0.1593    +0.3858    +0.4320    +0.4381
rxn8832         2    +0.1017    +0.0654    -0.0019    +0.0008    +0.0191
rxn1320         3    +0.0118    +0.0287    +0.6244    +0.6194    +0.6264
rxn4113         4    -0.0168    +0.6004    +0.0046    +1.2061    +0.0174
rxn8885         5    -0.0196    -0.0267    +1.2318    -0.0044    +3.4712
rxn7945         6    -0.0092    -0.0757    -0.0168    -0.0129    -0.6054
rxn7937         7    +0.0086    -0.0602    -0.0144    -0.0158    +0.0062
rxn6196         8    +0.0244    -0.0822    +0.0816    +0.1055    +0.1064
rxn1150        10    -0.0472    +0.0368    +0.0306    +0.0209    +0.0214
rxn3107        13    -0.0031    -0.0313    -0.0017    -0.0029    -0.0162
rxn8837        14    +0.0983    +0.1019    +4.2616    +2.2396    +2.0706
rxn7060        15    +0.0879    +0.0890    -0.0114    +0.0148    -0.0130
rxn8827        18    -0.0115    -0.3998    +0.0522    +0.0569    +0.0719
rxn7936        20    -0.0162    -0.0157    +0.0009    +0.0008    +0.0043
rxn0101        23    +0.1508    +0.1489    +0.1644    +0.1275    +0.1245
rxn10005       24    -0.0599    -0.0637    +0.0018    -0.0017    -0.0004
rxn10054       25    +0.0153    +0.0138    -0.0324    -0.0158    -0.0127
rxn7957        26    -0.1333    -0.1365    -0.4055    -0.4263    -0.4076
```

#### Bond 2

```
rxn          rank       MACE      Delta      UMA-S      UMA-M       eSEN
------------------------------------------------------------------------
rxn7949         1    -0.0251    -0.0192    -0.0239    -0.0248    -0.0246
rxn8832         2    -0.0416    -0.0436    -0.0177    -0.0157    -0.0149
rxn1320         3    -0.0022    -0.0015    -0.0264    -0.0263    -0.0258
rxn4113         4    -0.0219    +0.4118    +0.0291    +1.0871    +0.0488
rxn8885         5    +0.0143    +0.0084    +0.1002    +0.0093    +0.0893
rxn7945         6    +0.0174    +0.3417    +0.0110    -0.0007    +0.2359
rxn7937         7    -0.0131    +0.0540    +0.0145    +0.0068    +0.0255
rxn6196         8    +0.1012    -0.1973    +0.2407    +0.2732    +0.2721
rxn1150        10    -0.0358    +0.0111    +0.0201    +0.0149    +0.0063
rxn3107        13    +0.0278    +0.0090    +0.0170    +0.0159    +0.0179
rxn8837        14    +0.1674    +0.1849    -0.6330    -0.8277    -0.8328
rxn7060        15    -0.0357    -0.0372    -0.0085    -0.0076    -0.0118
rxn8827        18    -0.0449    +0.1873    +0.2274    +0.2309    +0.2339
rxn7936        20    +0.0432    +0.0353    +0.0031    +0.0042    +0.0037
rxn0101        23    -0.1424    -0.2113    -0.0605    -0.0422    -0.0497
rxn10005       24    +0.1038    +0.0803    +0.0005    +0.0036    +0.0018
rxn10054       25    -0.0027    -0.0001    +0.0474    +0.0212    +0.0235
rxn7957        26    +0.0084    +0.0112    +0.0184    +0.0383    +0.0212
```

#### Bond MAE (mean |error| over 2 bonds)

```
rxn          rank       MACE      Delta      UMA-S      UMA-M       eSEN
------------------------------------------------------------------------
rxn7949         1     0.0692     0.0892     0.2048     0.2284     0.2314
rxn8832         2     0.0716     0.0545     0.0098     0.0082     0.0170
rxn1320         3     0.0070     0.0151     0.3254     0.3228     0.3261
rxn4113         4     0.0194     0.5061     0.0168     1.1466     0.0331
rxn8885         5     0.0170     0.0176     0.6660     0.0068     1.7802
rxn7945         6     0.0133     0.2087     0.0139     0.0068     0.4206
rxn7937         7     0.0108     0.0571     0.0144     0.0113     0.0158
rxn6196         8     0.0628     0.1397     0.1612     0.1894     0.1892
rxn1150        10     0.0415     0.0240     0.0253     0.0179     0.0138
rxn3107        13     0.0154     0.0202     0.0094     0.0094     0.0170
rxn8837        14     0.1328     0.1434     2.4473     1.5336     1.4517
rxn7060        15     0.0618     0.0631     0.0100     0.0112     0.0124
rxn8827        18     0.0282     0.2936     0.1398     0.1439     0.1529
rxn7936        20     0.0297     0.0255     0.0020     0.0025     0.0040
rxn0101        23     0.1466     0.1801     0.1124     0.0849     0.0871
rxn10005       24     0.0819     0.0720     0.0012     0.0026     0.0011
rxn10054       25     0.0090     0.0070     0.0399     0.0185     0.0181
rxn7957        26     0.0708     0.0738     0.2120     0.2323     0.2144
------------------------------------------------------------------------
mean                  0.0494     0.1106     0.2451     0.2210     0.2770
median                0.0356     0.0675     0.0326     0.0182     0.0601
```

---

### Step 3a — All-atom Kabsch RMSD (Å)

```
rxn          rank       MACE      Delta      UMA-S      UMA-M       eSEN
------------------------------------------------------------------------
rxn7949         1      0.168      0.242      0.162      0.201      0.198
rxn8832         2      0.109      0.120      0.145      0.181      0.166
rxn1320         3      0.071      0.052      0.362      0.361      0.366
rxn4113         4      0.012      0.210      0.014      0.740      0.017
rxn8885         5      0.009      0.025      0.492      0.066      1.411
rxn7945         6      0.095      0.328      0.044      0.032      0.427
rxn7937         7      0.084      0.153      0.043      0.037      0.049
rxn6196         8      0.038      0.109      0.085      0.093      0.093
rxn1150        10      0.016      0.077      0.009      0.007      0.007
rxn3107        13      0.167      0.140      0.027      0.028      0.035
rxn8837        14      0.196      0.259      2.707      1.812      1.439
rxn7060        15      0.077      0.086      0.044      0.013      0.054
rxn8827        18      0.087      0.407      0.211      0.236      0.262
rxn7936        20      0.074      0.063      0.002      0.002      0.004
rxn0101        23      0.110      0.171      0.074      0.056      0.058
rxn10005       24      0.040      0.050      0.004      0.003      0.004
rxn10054       25      0.007      0.027      0.022      0.015      0.014
rxn7957        26      0.117      0.164      0.206      0.214      0.205
------------------------------------------------------------------------
mean                   0.082      0.149      0.259      0.228      0.267
median                 0.081      0.130      0.059      0.061      0.075
```

### Step 3b — Heavy-atom Kabsch RMSD (Å)

```
rxn          rank       MACE      Delta      UMA-S      UMA-M       eSEN
------------------------------------------------------------------------
rxn7949         1      0.113      0.156      0.136      0.162      0.166
rxn8832         2      0.066      0.076      0.078      0.077      0.082
rxn1320         3      0.050      0.032      0.143      0.146      0.148
rxn4113         4      0.015      0.216      0.010      0.482      0.016
rxn8885         5      0.010      0.023      0.454      0.040      1.403
rxn7945         6      0.094      0.276      0.028      0.017      0.296
rxn7937         7      0.071      0.122      0.035      0.030      0.040
rxn6196         8      0.034      0.080      0.068      0.073      0.073
rxn1150        10      0.016      0.049      0.008      0.005      0.003
rxn3107        13      0.096      0.077      0.017      0.021      0.032
rxn8837        14      0.146      0.170      1.720      0.914      1.099
rxn7060        15      0.092      0.093      0.039      0.012      0.033
rxn8827        18      0.045      0.450      0.119      0.128      0.144
rxn7936        20      0.069      0.060      0.002      0.002      0.005
rxn0101        23      0.091      0.121      0.073      0.056      0.055
rxn10005       24      0.040      0.041      0.002      0.003      0.005
rxn10054       25      0.007      0.014      0.019      0.009      0.009
rxn7957        26      0.117      0.150      0.213      0.219      0.213
------------------------------------------------------------------------
mean                   0.065      0.123      0.176      0.133      0.212
median                 0.068      0.086      0.054      0.048      0.064
```

### Step 3c — Reactive-subset Kabsch RMSD (Å, aligned on subset)

```
rxn          rank       MACE      Delta      UMA-S      UMA-M       eSEN
------------------------------------------------------------------------
rxn7949         1      0.057      0.081      0.184      0.206      0.210
rxn8832         2      0.056      0.045      0.009      0.008      0.013
rxn1320         3      0.018      0.017      0.269      0.266      0.269
rxn4113         4      0.013      0.275      0.012      0.600      0.020
rxn8885         5      0.011      0.027      0.521      0.012      1.696
rxn7945         6      0.009      0.185      0.009      0.005      0.281
rxn7937         7      0.012      0.036      0.009      0.008      0.011
rxn6196         8      0.043      0.081      0.102      0.115      0.115
rxn1150        10      0.020      0.015      0.013      0.009      0.009
rxn3107        13      0.015      0.016      0.010      0.010      0.012
rxn8837        14      0.091      0.131      1.840      0.985      0.891
rxn7060        15      0.037      0.039      0.007      0.007      0.012
rxn8827        18      0.035      0.213      0.102      0.102      0.104
rxn7936        20      0.023      0.018      0.001      0.002      0.003
rxn0101        23      0.106      0.131      0.086      0.065      0.066
rxn10005       24      0.051      0.044      0.001      0.002      0.001
rxn10054       25      0.006      0.007      0.022      0.010      0.010
rxn7957        26      0.055      0.056      0.168      0.176      0.169
------------------------------------------------------------------------
mean                   0.037      0.079      0.187      0.144      0.216
median                 0.029      0.044      0.017      0.011      0.043
```

---

### Step 5 — Failure characterization: rxn1320 and rxn8837

#### rxn1320 (rank 3) — H-transfer: C2-H6 breaking, O0-H6 forming

Reference TS position along reaction coordinate:
```
Bond        R        TS(ref)     P       delta(R->P)
C2-H6    2.8880     1.9812    1.0874     -1.8006
O0-H6    0.9634     0.9932    2.5609     +1.5975
```

Worst-atom displacement after global Kabsch alignment:
```
model    atom      disp(Å)    vector [x, y, z]
MACE     H7         0.1406    [+0.1285, -0.0188, -0.0541]
Delta    H7         0.1065    [+0.0931, -0.0227, -0.0466]
UMA-S    H6         0.7936    [-0.6903, +0.1540, +0.3599]
UMA-M    H6         0.7909    [-0.6907, +0.1516, +0.3541]
eSEN     H6         0.7988    [-0.6980, +0.1527, +0.3571]
```

Bond-length errors:
```
model    bond1(C2-H6)  bond2(O0-H6)    MAE
MACE       +0.0118        -0.0022      0.0070
Delta      +0.0287        -0.0015      0.0151
UMA-S      +0.6244        -0.0264      0.3254
UMA-M      +0.6194        -0.0263      0.3228
eSEN       +0.6264        -0.0258      0.3261
```

UMA-S/M/eSEN all displace H6 (the transferring hydrogen) ~0.79 Å in the same direction [-x, +y, +z]. C2-H6 bond ~0.62 Å too long — H6 placed too close to C2, not yet at the midpoint. O0-H6 near-correct (−0.026 Å). MACE and Delta: worst atom is spectator H7, displacement <0.14 Å, bond errors <0.03 Å.

#### rxn8837 (rank 14) — N-C forming (N0-C6), C-C breaking (C4-C6)

Reference TS position along reaction coordinate:
```
Bond        R        TS(ref)     P       delta(R->P)
N0-C6    3.5763     1.9120    1.4643     -2.1120
C4-C6    1.5065     2.1116    2.3386     +0.8321
```

Worst-atom displacement after global Kabsch alignment:
```
model    atom      disp(Å)    vector [x, y, z]
MACE     H8         0.3920    [+0.013, +0.277, +0.277]
Delta    H8         0.5755    [+0.115, +0.413, +0.383]
UMA-S    H8         8.1973    [+3.649, -1.242, -7.235]
UMA-M    H8         6.1069    [+3.970, -2.069, -4.154]
eSEN     H8         3.3915    [+1.901, -1.949, -2.023]
```

Bond-length errors:
```
model    bond1(N0-C6)  bond2(C4-C6)    MAE
MACE       +0.0983        +0.1674      0.1328
Delta      +0.1019        +0.1849      0.1434
UMA-S      +4.2616        -0.6330      2.4473
UMA-M      +2.2396        -0.8277      1.5336
eSEN       +2.0706        -0.8328      1.4517
```

All five models have H8 as the worst atom. UMA-S/M/eSEN displace H8 by 3.4–8.2 Å; N0-C6 error +2.1 to +4.3 Å — forming bond not formed (N0-C6 at 4.0–6.2 Å vs reference 1.91 Å); C4-C6 error −0.63 to −0.83 Å. MACE and Delta: H8 displaced <0.58 Å, N0-C6 error +0.10 Å, C4-C6 error +0.17 Å.
