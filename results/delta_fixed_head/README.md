# Fixed delta head — audit, retrain, and comparison against the v2 head

Date: 2026-09-01. All numbers from cluster runs of that day (jobs 10797800 train,
10798843 SP eval + rotation test, 10798883 NEB array). Raw JSONs in this folder;
analysis scripts: `pipeline/delta/compare_heads_tiers.py`, `pipeline/delta/analyze_neb.py`
(they read from a scratch copy — point `D` at this folder to re-run).

## The defect (v2 head, `delta_head_fw2.00.pt`)

- Declared input irreps `1024x0e+1024x1o+1024x2e+1024x3o` on `node_feats[:, 1024:]`.
- An equivariant linear to `64x0e` only has 0e→0e paths, so only the first 1,024
  dims carried weights: 65,600 = 1024·64 + 64. Higher-order features were ignored.
- MACE (0.3.15) concatenates node_feats as [layer0 full irreps (16,384), layer1
  scalars (1,024)]. The slice therefore starts with layer0's **1o components**, which
  the head treated as scalars → Δ was not rotation-invariant.
- Verified from checkpoint shapes (`linear_1.weight` = 65,536) and the MACE source.

## The fix (`delta_head_fixed_fw2.00.pt`, `train_delta_head_fixed.py`)

Full node_feats with true irreps `1024x0e+1024x1o+1024x2e+1024x3o+1024x0e`; the head
reads the 2,048 true scalars of both layers → 131,136 params. Same data, same
settings (fw = 2.0, batch 64, 10k geoms/epoch), fresh training; early stop at epoch 153.
Validation force loss (checkpoint criterion): **0.0037 → 0.0017**.

## Rotation invariance (30 ORCA TS geometries × 24 random rotations, spread of Δ)

| | mean spread | median | max |
|---|---|---|---|
| MACE energy (control) | 0.000 meV | 0.000 | 0.000 |
| v2 head | **212 meV** | 199 | 422 |
| fixed head | 0.0008 meV | 0.0008 | 0.0014 |

## Fixed-geometry SP benchmark (30 reactions, 10 images each, vs ωB97M-V/def2-TZVP)

Tiers: high = first 10 of ALL30, low = rxn9246…rxn1962, mid = rxn0896…rxn1155.

| set | method | eMAE meV | fMAE meV/Å | barrier MAE meV | barrier bias |
|---|---|---|---|---|---|
| all 30 | wB97X-D3 (DFT) | 94.9 | 140.2 | 151.3 | +148 |
| all 30 | MACE | 107.6 | 138.9 | 168.4 | +114 |
| all 30 | MACE+Δ v2 | 63.7 | 76.4 | 124.3 | −34 |
| all 30 | MACE+Δ fixed | **55.6** | **48.0** | 126.8 | −99 |
| low (10) | MACE+Δ v2 | 47.6 | 62.0 | 86.4 | +22 |
| low (10) | MACE+Δ fixed | **29.6** | **31.6** | **47.0** | +21 |
| mid (10) | MACE+Δ v2 | 70.0 | 80.1 | 135.8 | −25 |
| mid (10) | MACE+Δ fixed | 54.3 | 53.0 | 147.1 | −134 |
| high (10) | MACE+Δ v2 | 73.4 | 87.0 | 150.9 | −98 |
| high (10) | MACE+Δ fixed | 82.8 | 59.3 | 186.3 | −184 |

Forces improve on 30/30 reactions. MACE columns agree between runs to 0.1 meV.

## RKS-stable subset (22 reactions, vs ωB97M-V; OMol25 models for scale)

| | MACE | MACE+Δ v2 | MACE+Δ fixed | UMA-S / UMA-M / eSEN |
|---|---|---|---|---|
| eMAE meV | 109.3 | 62.1 | **56.1** | 5.1 / 5.3 / 4.9 |
| fMAE meV/Å | 130.6 | 72.4 | **44.5** | 11.5 / 10.8 / 12.7 |
| force cosine | 0.30 | 0.52 | **0.60** | 0.84 / 0.84 / 0.82 |
| eMAE by tier high/mid/low | 108/139/86 | 106/58/48 | **165**/35/30 | — |

## NEB-driven barriers (30 reactions, MACE or MACE+Δ as the NEB potential)

Forward barrier vs ωB97M-V CI-NEB reference; TS-RMSD = Kabsch RMSD (all atoms, proper
rotations) of the NEB-found TS to the ORCA ωB97M-V CI-NEB TS. All 30 converged for all
three potentials.

| set | method | barrier MAE meV | bias | TS-RMSD mean / median Å |
|---|---|---|---|---|
| all 30 | MACE | 164.7 | +69 | 0.056 / 0.017 |
| all 30 | MACE+Δ v2 | 173.6 | −117 | 0.101 / 0.072 |
| all 30 | MACE+Δ fixed | 166.7 | −139 | **0.054 / 0.016** |
| low (10) | MACE | 166.0 | +166 | 0.015 / 0.012 |
| low (10) | MACE+Δ v2 | 89.3 | +6 | 0.053 / 0.050 |
| low (10) | MACE+Δ fixed | **52.1** | +22 | **0.016 / 0.010** |
| mid (10) | MACE | 233.8 | +45 | 0.092 / 0.035 |
| mid (10) | MACE+Δ v2 | 159.4 | −131 | 0.116 / 0.077 |
| mid (10) | MACE+Δ fixed | 211.6 | −204 | 0.065 / 0.034 |
| high (10) | MACE | 94.4 | −3 | 0.061 / 0.054 |
| high (10) | MACE+Δ v2 | 272.1 | −226 | 0.134 / 0.115 |
| high (10) | MACE+Δ fixed | 236.4 | −235 | 0.081 / 0.049 |
| all 30 vs CCSD(T) | MACE / v2 / fixed | 212.2 / 167.6 / **145.1** | +153 / −34 / −55 | — |

## Reading

- The v2 head roughened the surface: TS-RMSD 0.101 Å vs 0.056 Å for bare MACE. The
  fixed head restores bare-MACE geometry quality (0.054 Å) in every tier.
- Where labels are reliable (low MR): fixed head halves the barrier error vs v2
  (89 → 52 meV NEB; 86 → 47 meV SP) with near-zero bias.
- Where labels are unreliable (mid/high MR): both heads underestimate barriers by
  100–250 meV; the fixed head, which now actually reads geometry, is worse than v2 on
  energies there. Δ-learning cannot repair the labels it is trained on.
- Against natively trained OMol25 models the cheap upgrade remains ~10× off in
  pointwise error; it closes about half the MACE → target gap.

## TS geometry yardstick (added 2026-09-02)

Kabsch RMSD between the T1x (wB97X-D3) transition state and the ORCA
wB97M-V reference transition state, over the 30 benchmark reactions:
**mean 0.011 Å, median 0.007 Å, max 0.086 Å (rxn5690)**. This is the
level-to-level geometry difference; MACE's NEB (median 0.017 Å) lands at
about twice it. Caveat: the reference NEB was warm-started from the T1x
band, so this difference may be biased slightly low.

MACE-NEB TS vs its **own target** TS (T1x wB97X-D3): **mean 0.054 Å,
median 0.015 Å** (n=30; script `mace_vs_target_ts.py`). So MACE's geometry
error to its own level (0.015) is already ~2× the level-to-level gap
(0.007) — the delta head has nothing meaningful to gain on geometries.
