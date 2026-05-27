# Thesis Project Context Document

**Author:** Benedikt Pabinger (s242862@student.dtu.dk)  
**Last updated:** 2026-05-25  
**Working directory:** Transition1x repository  
**Cluster:** DTU HPC, s242862@slid.fysik.dtu.dk (SLURM)

---

## 1. Project Narrative — What and Why

### 1.1 The Problem

Machine learning interatomic potentials (MLIPs) like MACE and PaiNN promise near-DFT accuracy at a fraction of the cost. But accuracy is only as good as the training data. The Transition1x dataset (T1x) was generated at the **wB97X-D3/6-31G(d)** level of theory — a computationally cheap but methodologically limited choice. Two limitations matter for this thesis:

1. **Basis set and functional incompleteness.** wB97X-D3/6-31G(d) systematically underestimates barriers compared to higher-level references. The correction (wB97M-V/def2-TZVP vs. wB97X-D3/6-31G(d)) is approximately −3 eV in total energy, with ~100–200 meV variation along reaction paths.

2. **Single-reference limitation.** Many transition states involve bond-breaking and bond-forming events with genuine multireference (MR) character — near-degenerate orbitals, partial biradical character. Single-reference methods (DFT, CCSD) are unreliable in these cases. T1x was generated entirely with single-reference DFT, so MLIPs trained on it inherit this limitation.

### 1.2 The Thesis Questions

1. How large is the DFT-level error (wB97X-D3 → wB97M-V) in the training data, and can a **delta learning model** correct it cheaply at inference time?
2. Which reactions in the T1x test set have genuine multireference character, and how do single-reference methods (including MACE) fail on them?
3. Can high-level multireference methods (NEVPT2/AVAS) quantify this failure and serve as a gold-standard benchmark?

### 1.3 The Approach (three parallel threads)

**Thread A — Delta Learning:** Train a correction head on top of a frozen MACE model to predict the DFT-level correction delta = E_wB97M-V − E_wB97X-D3. At inference: predicted E_wB97M = E_MACE(wB97X) + delta_head(geometry).

**Thread B — Multireference Benchmark:** Use FOD screening to identify the 10 most strongly correlated reactions in the T1x test set. Compute barriers at CCSD(T)/def2-TZVP and NEVPT2/AVAS/def2-TZVP. Compare with MACE and DFT predictions.

**Thread C — NEB Refinement:** Re-run NEB at the higher level (wB97M-V/def2-TZVP, ORCA) on the full val and test sets, providing higher-quality reference geometries and energies for benchmarking.

---

## 2. Dataset: Transition1x (T1x)

**Source:** Schreiner et al., *Scientific Data* 2022. Hosted on Zenodo / GitLab.

**Content:** ~10,000 elementary organic reactions (gas phase, H/C/N/O/F atoms only). Each reaction includes:
- Reactant, product, and transition state geometries
- Full NEB optimization trajectory (all intermediate images, not just endpoints)
- Energies and forces at wB97X-D3/6-31G(d) for every image

**Splits:**
| Split | Reactions | Approx. images |
|-------|-----------|----------------|
| Train | ~9,561 | ~10 million |
| Val | 225 | ~600,000 |
| Test | ~400+ | ~1 million |

**Python interface:**
```python
from transition1x import Dataloader
dl = Dataloader('data/Transition1x.h5', datasplit='test', only_final=True)
for mol in dl:
    ts_energy = mol['transition_state']['wB97x_6-31G(d).energy']  # eV
    forces    = mol['transition_state']['wB97x_6-31G(d).forces']  # eV/Å
```

**HDF5 structure:**
```
f[split][formula][rxn_id]['positions']              # (N_images, N_atoms, 3) Å
f[split][formula][rxn_id]['atomic_numbers']         # (N_atoms,)
f[split][formula][rxn_id]['wB97x_6-31G(d).energy'] # (N_images,) eV
f[split][formula][rxn_id]['wB97x_6-31G(d).forces'] # (N_images, N_atoms, 3) eV/Å
```

---

## 3. Thread C: NEB Refinement at wB97M-V/def2-TZVP

### 3.1 Goal

Re-run NEB for all val and test reactions at a higher level (wB97M-V/def2-TZVP via ORCA) to obtain:
- Better-optimized transition state geometries
- wB97M-V energies along the path (stored in `neb.db`)
- Higher-quality barrier heights as references

### 3.2 Implementation

**Software:** ORCA 5.0.4 via ASE's NEB interface  
**ORCA keywords:** `wB97M-V def2-TZVP def2/J RIJCOSX TightSCF`  
**Convergence criterion:** fmax < 0.05 eV/Å (CI-NEB)  
**Cluster:** SLURM array jobs, one reaction per task

**Output per reaction:**
```
orca_neb_val_results/{rxn}/
  neb.db          — ASE database, all images + wB97M-V energies
  converged       — empty marker file, written ONLY on true convergence
  transition_state.xyz
  fmaxs.json
```

### 3.3 Results

| Set | Total reactions | Converged | Failed |
|-----|----------------|-----------|--------|
| Val | 225 | 174 (77%) | 51 (23%) |
| Test | ~400 | 279+ | — |

The 51 failed val reactions have no `neb.db` — NEB did not converge. They are handled separately in the delta model validation (Group B, see Section 5.2).

---

## 4. ML Models: MACE and PaiNN

### 4.1 PaiNN

**Architecture:** Message-passing neural network with equivariant updates (Schütt et al. 2021). Standard baseline for comparison.  
**Training:** On T1x train split (wB97X-D3/6-31G(d) energies + forces).  
**Job scripts:** `pipeline/job_painn_train.sh`, `pipeline/train_painn.py`

### 4.2 MACE

**Architecture:** `ScaleShiftMACE` — equivariant message-passing with higher-order tensor products (Batatia et al. 2022). State of the art for molecular MLIPs.

**Trained model (on cluster):**
- TorchScript compiled: `~/mace_t1x_p10_compiled.model` (CUDA required)
- State dict: `~/mace_t1x_p10_run-123_epoch-362.pt`

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| `num_channels` | 1024 |
| `max_L` | 3 |
| `num_interactions` | 2 |
| `num_radial_basis` | 16 |
| `r_max` | 6.0 Å |
| `hidden_irreps` | `1024x0e + 1024x1o + 1024x2e + 1024x3o` (dim=16384) |
| Epochs | 362 (best checkpoint) |

**Training scale issue:** T1x train set has ~10M geometries. Standard MACE epoch = full dataset pass. A patch was applied to enable epoch-level subsampling (`--max_samples_per_epoch`, `p10` = 10% per epoch).

**MACE patch:** 3 lines added across 2 files. Local copies at `pipeline/mace/`:
- `arg_parser.py` — adds `--max_samples_per_epoch` CLI flag
- `run_train.py` — activates `RandomSampler` when flag is set
- `scripts_utils.py` — loader adjustment
- Deployed to cluster via `pipeline/deploy_mace_patches.py` (paramiko SFTP)

**MACE forward pass (relevant for delta head):**
```python
node_feats_concat = []
for interaction, product in zip(interactions, products):
    node_feats = interaction(node_feats, ...)
    node_feats = product(node_feats, ...)
    node_feats_concat.append(node_feats)
# interaction 0 → [N_atoms, 1024]   (scalars only: 1024x0e)
# interaction 1 → [N_atoms, 16384]  (full irreps: 1024x0e+1x1o+1x2e+1x3o)

node_feats_out = torch.cat(node_feats_concat, dim=-1)  # [N_atoms, 17408]
output["node_feats"] = node_feats_out
output["energy"] = ...  # wB97X-D3 prediction
```

**MACE barrier MAE (test set, full benchmark):**
| Metric | MACE |
|--------|------|
| Barrier energy MAE | ~217.8 meV (fwd), ~120.0 meV (rev) |

---

## 5. Thread A: Delta Learning

### 5.1 Concept

The delta model corrects MACE's predictions from the wB97X-D3 level to the wB97M-V level:

```
delta(geometry) = E_wB97M-V/def2-TZVP − E_wB97X-D3/6-31G(d)

At inference:
  E_wB97M_predicted = E_MACE(wB97X) + delta_head(node_feats)
```

The correction is approximately −3 eV in total energy (dominated by basis set improvement), with ~0.1–0.25 eV variation along reaction paths (geometry-dependent component the head must learn).

### 5.2 Training Data Pipeline

**Train set:** 500 reactions sampled from T1x train split (seed=42), 10 geometries each = 5,000 ORCA SPs.

- Sampling: uniform spacing along NEB trajectory (covers R, TS, P regions equally)
- wB97X-D3 energies + forces: read from T1x HDF5 (already there)
- wB97M-V/def2-TZVP energies + forces: computed via ORCA SP+EnGrad on cluster
- Output: `~/train_delta_sp/{rxn}/results.json`

**Val set:** All 225 val reactions split into two groups:

| Group | Count | Source | wB97X-D3 | wB97M-V | Forces |
|-------|-------|--------|----------|---------|--------|
| A (converged NEB) | 174 | `neb.db` full history | ORCA SP | stored in neb.db | Only wB97X-D3 |
| B (failed NEB) | 51 | T1x HDF5 | stored in T1x | ORCA SP | Both |

*Group A limitation:* wB97M-V forces are not stored in `neb.db`. Only delta energy (not forces) available for Group A. Group B has full delta energy + delta forces.

**Results.json format:**
```json
{
  "rxn": "rxn0006",
  "n_total": 87,
  "n_sampled": 10,
  "geometries": [
    {
      "geom_idx": 0,
      "e_wb97m_eV": -1234.567890,
      "e_wb97x_eV": -1231.234567,
      "delta_eV": -3.333323,
      "forces_wb97m_eV_per_ang": [[fx, fy, fz], ...],
      "forces_wb97x_eV_per_ang": [[fx, fy, fz], ...],
      "delta_forces_eV_per_ang": [[fx, fy, fz], ...]
    }
  ]
}
```

**Val delta statistics (plausibility check 2026-05-12):**
| | Group A (174) | Group B (51) |
|---|---|---|
| Delta mean | −3.03 ± 0.29 eV | −2.84 ± 0.15 eV |
| Within-rxn std (mean) | 0.089 eV | 0.093 eV |
| Within-rxn std (max) | 0.251 eV | 0.200 eV |

High within-reaction variance at strained intermediate geometries is physical (both wB97M and wB97X energies move coherently) — not SCF failure.

### 5.3 Delta Head Architecture

The delta head is a new `NonLinearReadoutBlock` (identical architecture to MACE's final readout, fresh weights) attached to the frozen MACE backbone:

```python
from e3nn import o3
from mace.modules.blocks import NonLinearReadoutBlock
import torch.nn.functional as F

hidden_irreps = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")  # dim=16384
MLP_irreps    = o3.Irreps("16x0e")

delta_head = NonLinearReadoutBlock(
    irreps_in  = hidden_irreps,   # [N_atoms, 16384] — last 16384 dims of node_feats_out
    MLP_irreps = MLP_irreps,      # intermediate: 16 scalars
    gate       = F.silu,
)
# Output: [N_atoms, 1] → summed to [N_structures, 1] per-structure delta energy
```

**Inference:**
```python
# Backbone frozen
for p in mace_model.parameters():
    p.requires_grad_(False)

batch['positions'].requires_grad_(True)
out = mace_model(batch, training=False, compute_force=False)

node_feats        = out['node_feats'][:, 1024:]           # [N, 16384]
e_wb97x           = out['energy']
per_atom_delta    = delta_head(node_feats)
per_struct_delta  = scatter_sum(per_atom_delta, batch.batch)

# Forces via autograd (MACE must run live — cannot cache node_feats)
delta_forces      = -torch.autograd.grad(per_struct_delta.sum(), batch['positions'])[0]

e_wb97m_predicted = e_wb97x + per_struct_delta
```

**Why forces require live MACE forward pass:** Forces = −d(delta_energy)/d(positions). This gradient flows through the delta head and MACE backbone back to atom positions. Cached node_feats are constants — autograd has nothing to differentiate through.

### 5.4 Training

| Setting | Value |
|---------|-------|
| Epochs | 200 |
| Batch size | 32 |
| Optimizer | Adam, lr=1e-3 |
| Scheduler | ReduceLROnPlateau (patience=10) |
| Loss | Huber(delta_energy) + Huber(delta_forces), δ=0.1 eV |
| Val set | Fixed 1024-geom sample from Group A (energy-only) |
| wandb run | "clean-star-4" (job 10368920) |
| Best checkpoint | `~/delta_head/delta_head.pt` |

**Training result:** Final val_e ≈ 13.9 meV (energy MAE on val sample).

### 5.5 Evaluation Results

#### Delta prediction on val set (20 reactions, 980 geometries):
| Method | MAE (meV) |
|--------|-----------|
| MACE alone (wB97X prediction vs wB97M truth) | ~2980 |
| MACE + delta head | ~154 |
| Improvement | −94% |

#### Barrier MAE on NEB paths (286 test reactions, 10 images each, `delta_head_eval_neb_results.json`):
| Metric | MACE | MACE + delta |
|--------|------|-------------|
| Energy MAE | 117.6 meV | 109.3 meV |
| Forward barrier MAE | 217.8 meV | 176.0 meV |
| Reverse barrier MAE | 120.0 meV | 190.6 meV |

**Key observation:** The delta head improves forward barrier MAE (−19%) but degrades reverse barrier MAE (+59%). The head systematically underpredicts |delta| by ~100–150 meV (positive bias — predicted correction smaller than true correction). Likely causes: training on energy-only (no force supervision on spatial variation), small MLP capacity (16x0e), or Huber δ=0.1 eV clipping large errors.

#### Known API pitfalls (cluster-specific):
1. `torch.serialization.add_safe_globals([slice])` must precede `from e3nn import o3`
2. TorchScript model needs `Dict[str, Tensor]`, not `Batch` — convert explicitly
3. `batch.keys` is a property (list), not a method — no `()`
4. Wrap validation forward in `with torch.enable_grad():` (forces need autograd even in eval mode)
5. Always use `python3 -u` in SLURM scripts (stdout buffering)

---

## 6. Thread B: Multireference Benchmark

### 6.1 FOD Screening

**Goal:** Identify reactions in the T1x test set with the strongest multireference character without running expensive CCSD on all 279 converged test reactions.

**Method:** FOD (Fractional Occupation number weighted Density, Grimme & Hansen 2015). Run DFT at high electronic temperature (T_el = 5000 K, Fermi-Dirac smearing). Measure fractional orbital occupations:

```
NFOD = Σᵢ |nᵢ − n⁰ᵢ|
```

Near-degenerate orbitals (occupation ≈ 1) contribute most. Thresholds: < 0.05 negligible, 0.05–0.5 mild, 0.5–1.5 significant, > 1.5 strongly MR.

**Implementation:** PySCF, PBE/def2-SVP, smearing via `pyscf.scf.addons.smearing_`, geometry = ORCA NEB TS (wB97M-V/def2-TZVP).

**Why FOD over T1 diagnostic:** FOD is a DFT single-point (10–20× cheaper than CCSD needed for T1). Screening 279 reactions with T1 would take days; FOD runs in one batch job.

### 6.2 Benchmark Design — 30 Reactions

The benchmark covers **30 reactions** from the T1x test split, grouped by MR character to allow method comparisons across the full MR spectrum:

| Group | Reactions | Selection |
|-------|-----------|-----------|
| High MR (top 10) | rxn7949, rxn8832, rxn1320, rxn4113, rxn8885, rxn7945, rxn7937, rxn6196, rxn0346, rxn1150 | Highest NFOD scores |
| Mid MR (middle 10) | rxn0896, rxn1154, rxn5690, rxn4513, rxn7955, rxn4519, rxn4500, rxn2553, rxn8829, rxn1155 | Intermediate NFOD |
| Low MR (bottom 10) | rxn9246, rxn4498, rxn1061, rxn4003, rxn4004, rxn4063, rxn4114, rxn4060, rxn1961, rxn1962 | Lowest NFOD scores |

Zero overlap with delta head training or validation data.

### 6.3 Level-of-Theory Ladder

All methods evaluated as single points on the same **wB97M-V/def2-TZVP geometries** (from ORCA NEB). Geometry and electronic-structure effects are cleanly separated.

| Step | Method | Geometries | Software |
|------|--------|-----------|----------|
| 1 | wB97X-D3/6-31G(d) | 10 NEB images per reaction (300 total) | ORCA |
| 2 | wB97M-V/def2-TZVP | 10 NEB images per reaction | stored in neb.db |
| 3 | CCSD(T)/def2-TZVP | R, TS, P only — all 30 reactions | PySCF (24 OMP threads) |
| 4 | NEVPT2/AVAS/def2-TZVP | R, TS, P only — all 30 reactions (reliability check) | PySCF |

### 6.4 CCSD(T) Single Points

RHF → RCCSD → CCSD(T) with PySCF on all 30 reactions. No active space needed. The triples correction (T) is larger at TS than R/P for MR systems; formally unreliable when MR is strong. Largest TS–R triples delta (−0.028 Ha): rxn7949 and rxn8832.

### 6.5 NEVPT2/AVAS

**Why AVAS (Automated Valence Active Space, Sayfutyarova et al. 2017):** Automates active space selection by projecting MOs onto target atomic orbital types. Reproducible, avoids human bias.

**Strategy:**
- AVAS active space defined **once at the TS** (where MR character is strongest)
- TS CASSCF MO coefficients projected to R and P via `mcscf.project_init_guess`
- This ensures R, TS, P share the same orbital space → barriers are physically meaningful

**AO targets:**
```
C 2pz, N 2p, O 2pz, F 2pz   (threshold = 0.2)
```
Using `2pz` (π-type) instead of full `2p` prevents selecting the σ-skeleton, which would make active spaces intractably large (32–34e, 22–26o → unfeasible). N uses all three p components because N lone pairs may not align with z.

**CASSCF convergence settings:**
```
max_cycle_macro = 1000
max_stepsize    = 0.05   (orbital rotation damping)
conv_tol        = 1e-8
```
Damping prevents oscillation between near-degenerate configurations.

**Reliability criterion — 5 flags:**
After CASSCF, compute natural orbital occupancies from 1-RDM. "Fractional" = 0.05 < n < 1.95. A reaction is **RED FLAG** if any of:

| Flag | Condition | Interpretation |
|------|-----------|----------------|
| `0 frac@R` | Zero fractional occupations at reactant | Active space idle at R |
| `0 frac@TS` | Zero fractional occupations at TS | Active space idle at TS |
| `0 frac@P` | Zero fractional occupations at product | Active space idle at P |
| `neg rev` | Reverse barrier < 0 meV | Physically impossible |
| `\|NEV-CCT\| > 300 meV` | NEVPT2 and CCSD(T) disagree by >300 meV | Secondary cross-check |

### 6.6 MR Benchmark Results

#### Full 30-reaction reliability table

All barriers in meV. Source: `full_benchmark_results.json`, `pipeline/_check_nevpt2_plausibility.py`.

| Reaction | Group | Active Space | frac R/TS/P | NEVPT2 fwd | CCSD(T) fwd | diff | Status |
|----------|-------|-------------|-------------|-----------|------------|------|--------|
| rxn7949  | High  | (16e,12o)   | 6/8/6       | 3254       | 3210        | +44  | Reliable |
| rxn8832  | High  | (18e,13o)   | 4/6/6       | 2540       | 2621        | −81  | Reliable |
| rxn1320  | High  | (16e,10o)   | 0/2/2       | 3147       | 3051        | +96  | Red flag — 0 frac@R |
| rxn4113  | High  | —           | —           | —          | 5346        | —    | Failed — CASSCF no conv. at P |
| rxn8885  | High  | (14e,11o)   | 6/6/4       | 3643       | 3564        | +79  | Reliable |
| rxn7945  | High  | (16e,12o)   | 6/6/6       | 3943       | 3923        | +20  | Reliable |
| rxn7937  | High  | (16e,12o)   | 6/6/4       | 3764       | 3858        | −94  | Reliable |
| rxn6196  | High  | (14e,12o)   | 6/6/8       | 4181       | 4282        | −101 | Reliable |
| rxn0346  | High  | (14e,10o)   | 2/2/2       | 3213       | 3336        | −123 | Reliable |
| rxn1150  | High  | (14e,10o)   | 1/4/4       | 3362       | 3460        | −98  | Red flag — 0 frac@R (borderline) |
| rxn0896  | Mid   | (16e,11o)   | 2/2/1       | 5145       | 5094        | +51  | Reliable |
| rxn1154  | Mid   | (14e,9o)    | 1/2/0       | 4295       | 3847        | +448 | Red flag — 0 frac@P; \|NEV-CCT\|=448meV |
| rxn5690  | Mid   | (18e,12o)   | 4/4/4       | 3274       | 3346        | −73  | Red flag — neg rev (−72meV) |
| rxn4513  | Mid   | (14e,9o)    | 2/0/0       | 1929       | 1936        | −7   | Red flag — 0 frac@TS; 0 frac@P |
| rxn7955  | Mid   | (18e,14o)   | 6/6/6       | 3100       | 3080        | +19  | Reliable |
| rxn4519  | Mid   | —           | —           | —          | 4903        | —    | Missing |
| rxn4500  | Mid   | —           | —           | —          | 4745        | —    | Missing |
| rxn2553  | Mid   | (12e,8o)    | 2/2/2       | 2009       | 2011        | −2   | Reliable |
| rxn8829  | Mid   | (16e,13o)   | 4/4/4       | 2994       | 2938        | +56  | Reliable |
| rxn1155  | Mid   | (14e,8o)    | 0/0/2       | 2453       | 2797        | −344 | Red flag — 0 frac@R; 0 frac@TS; \|NEV-CCT\|=344meV |
| rxn9246  | Low   | (14e,8o)    | 2/0/2       | 1433       | 1776        | −343 | Red flag — 0 frac@TS; \|NEV-CCT\|=343meV |
| rxn4498  | Low   | —           | —           | —          | 3268        | —    | Missing |
| rxn1061  | Low   | —           | —           | —          | 1144        | —    | Missing |
| rxn4003  | Low   | —           | —           | —          | 1998        | —    | Missing |
| rxn4004  | Low   | (18e,12o)   | 0/0/2       | 2185       | 2011        | +174 | Red flag — 0 frac@R; 0 frac@TS |
| rxn4063  | Low   | (18e,11o)   | 0/0/0       | 2031       | 1956        | +75  | Red flag — 0 frac@R/TS/P |
| rxn4114  | Low   | (14e,9o)    | 0/0/1       | 2434       | 2481        | −47  | Red flag — 0 frac@R; 0 frac@TS |
| rxn4060  | Low   | (20e,12o)   | 0/0/0       | 1688       | 1871        | −184 | Red flag — 0 frac@R/TS/P |
| rxn1961  | Low   | (14e,8o)    | 0/0/0       | 869        | 2269        | −1400| Red flag — 0 frac@R/TS/P; neg rev; \|NEV-CCT\|=1400meV |
| rxn1962  | Low   | (14e,8o)    | 0/0/0       | 2019       | 2411        | −393 | Red flag — 0 frac@R/TS/P; \|NEV-CCT\|=393meV |

**Summary by group:**

| Group | Reliable | Red flag | Missing/Failed | Total |
|-------|----------|----------|----------------|-------|
| High MR | 7 | 2 | 1 | 10 |
| Mid MR | 4 | 4 | 2 | 10 |
| Low MR | 0 | 7 | 3 | 10 |
| **All 30** | **11** | **13** | **6** | **30** |

**Red flag detail (rxn1320):** AVAS at TS selects orbitals with occupations 1.483/0.523 (bond-breaking pair). At reactant those orbitals are fully closed/empty (≥1.977). NEVPT2 adds no correlation at R but substantial at TS → forward barrier artificially biased. CCSD(T) is the reference for rxn1320 and rxn1150.

**Root cause of failures in low-MR reactions:** AVAS projects onto π-type AOs at the TS. For low-MR reactions the TS has no near-degenerate orbitals — all π orbitals are fully occupied or empty. The active space is physically idle everywhere: CASSCF converges to essentially HF, NEVPT2 adds negligible correlation. The method works only where genuine MR character exists.

**Mid-MR reactions:** Mixed. Four reliable (rxn0896, rxn7955, rxn2553, rxn8829) where moderate genuine MR character exists at all three geometries. Failures arise from TS-biased active spaces or marginally negative reverse barriers.

**Practical conclusion:** Use NEVPT2 barriers only for the 7 reliable top-10 reactions. For all others use CCSD(T) as the high-level reference.

**NEVPT2 vs. CCSD(T) spread:** −123 to +96 meV (top-10 reliable); −2 to +56 meV (mid-10 reliable). Consistent with NEVPT2 accuracy on moderately MR systems.

**MACE errors on MR reactions:** MACE forward barriers are systematically too high by 200–700 meV vs. CCSD(T) for high-MR reactions. The wB97X-D3 training data is itself unreliable for these reactions — the problem is in the training labels, not just model capacity. The delta head corrects the wB97X-D3→wB97M-V functional gap but cannot fix the SR→MR error.

---

## 7. Calculation Inventory

All calculation statuses tracked in `calculation_inventory.csv` (one row per reaction). Key columns:

| Column | Meaning |
|--------|---------|
| `rxn` | Reaction ID |
| `split` | train/val/test |
| `orca_neb_converged` | NEB converged (fmax < 0.05 eV/Å) |
| `orca_barrier_eV` | wB97M-V/def2-TZVP barrier from NEB |
| `t1x_barrier_eV` | wB97X-D3/6-31G(d) barrier from T1x NEB |
| `ccsdt_tz_compiled` | CCSD(T)/def2-TZ barriers compiled |
| `nevpt2_avas_fixed` | CASSCF converged; NEVPT2 completed |
| `nevpt2_avas_validated` | nat_occ balanced at R, TS, P |
| `nevpt2_avas_flag` | `ok` / `red_flag` / `failed` |
| `delta_sp` | wB97X-D3 + wB97M-V SPs done (test set) |
| `val_delta_sp` | Val Group A SPs done |
| `val_delta_sp_flip` | Val Group B SPs done |
| `mace_p10_ep291_meV` | MACE (p10, epoch 291) barrier prediction |

---

## 8. Pipeline Scripts Reference

All scripts in `pipeline/`. Shell scripts are SLURM jobs for the DTU cluster.

### NEB
| Script | Purpose |
|--------|---------|
| `orca_neb.py` | Run wB97M-V/def2-TZVP NEB on val set |
| `orca_neb_rerun.py` | Retry failed NEBs |
| `job_orca_neb.sh` | SLURM array for val NEB |

### MACE training
| Script | Purpose |
|--------|---------|
| `mace/run_train.py` | Patched MACE training CLI |
| `mace/arg_parser.py` | Adds --max_samples_per_epoch |
| `deploy_mace_patches.py` | SFTP upload to cluster |
| `job_mace_train.sh` | Main MACE training job |

### Delta learning
| Script | Purpose |
|--------|---------|
| `delta/sample_train_reactions.py` | Sample 500 train reactions |
| `delta/train_delta_sp.py` | wB97M-V ORCA SPs for train data |
| `delta/prepare_delta_data.py` | Build train.pt / val.pt caches |
| `delta/train_delta_head.py` | Train NonLinearReadoutBlock head |
| `delta/eval_delta_head.py` | Evaluate MACE vs MACE+delta |
| `delta/plot_delta_eval.py` | Generate plots |
| `val_delta_sp.py` | Val Group A SPs |
| `val_delta_sp_flip.py` | Val Group B SPs (flip approach) |

### MR Benchmark
| Script | Purpose |
|--------|---------|
| `screen_fod.py` | FOD single points at TS |
| `collect_fod.py` | Aggregate FOD, rank reactions |
| `mr_benchmark_setup.py` | Extract NEB geometries, gen ORCA inputs |
| `mr_benchmark_collect_sp.py` | Collect wB97X-D3 + wB97M-V barriers |
| `mr_benchmark_ccsdt.py` | CCSD(T)/def2-TZVP via PySCF |
| `mr_benchmark_nevpt2.py` | NEVPT2/AVAS via PySCF |
| `_analyze_full_benchmark.py` | Combine all levels into `full_benchmark_results.json` |

### Evaluation
| Script | Purpose |
|--------|---------|
| `eval_mace.py` | Evaluate MACE on test set |
| `eval_mace_delta.py` | Evaluate MACE + delta head |
| `compare_barriers.py` | Compare barriers across methods |
| `plot_barrier_errors.py` | Generate barrier error plots |

---

## 9. Key Results Summary

| Comparison | Method | Forward barrier MAE |
|------------|--------|-------------------|
| T1x reference | wB97X-D3/6-31G(d) | baseline |
| Higher DFT | wB97M-V/def2-TZVP | ~100–200 meV lower than wB97X-D3 |
| MLIP | MACE (p10, epoch 362) | ~217.8 meV vs. wB97M-V |
| MLIP + correction | MACE + delta head | ~176.0 meV vs. wB97M-V |
| Gold standard (SR) | CCSD(T)/def2-TZVP | reference for MR benchmark |
| Gold standard (MR) | NEVPT2/AVAS/def2-TZVP | ±123 meV vs. CCSD(T) (7 reliable) |

**MACE on MR reactions:** Errors of 200–700 meV vs. CCSD(T). Training data (wB97X-D3) is itself unreliable for these cases — the problem is in the training labels, not just the model.

**Delta head summary:** Reduces the DFT-level error (wB97X → wB97M) from ~3 eV total energy offset to ~154 meV MAE. Forward barrier improvement: −19%. Reverse barrier: degrades (+59%). Systematic underprediction bias (~130 meV) suggests force supervision and larger head capacity are needed.

---

## 10. Software Stack

| Component | Software | Notes |
|-----------|----------|-------|
| QC calculations | ORCA 5.0.4 | NEB, DFT SPs |
| QC calculations | PySCF | CCSD(T), NEVPT2, FOD |
| ML training | PyTorch + e3nn | MACE backbone |
| ML models | MACE (custom build) | Patched for T1x scale |
| ML models | PaiNN | Baseline |
| Geometry optimization | ASE | NEB interface |
| Cluster | DTU HPC SLURM | s242862@slid.fysik.dtu.dk |
| Experiment tracking | wandb | Delta head training |
| Data format | HDF5 (T1x), ASE db (NEB), JSON (results) | |

---

## 11. Open Questions / Future Work

1. **Retrain delta head with force supervision** (Val Group B has delta_forces). Expected to fix the systematic underprediction bias.
2. **Larger delta head** (64x0e MLP instead of 16x0e).
3. **Pre-cache AtomicData graphs** to speed up training (currently ~20 min/epoch).
4. **MR reactions: can delta head help?** The head corrects the DFT level, not the SR-to-MR error. For rxn7949/rxn8832, MACE errors are 700 meV vs. CCSD(T) — this is a training-data quality problem that delta learning cannot fix.
5. **Training data quality:** For MR reactions, wB97X-D3/6-31G(d) labels are unreliable. An MR-aware training strategy (or explicit exclusion from training) might improve MACE performance on MR reactions.
6. **FOD screening on train set:** Identify and potentially reweight or exclude strongly MR training reactions.

---

## 12. References

- Schreiner, M. et al. (2022). Transition1x — a dataset for building generalizable reactive machine learning potentials. *Scientific Data*, 9, 756.
- Batatia, I. et al. (2022). MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. *NeurIPS* 2022.
- Schütt, K. T. et al. (2021). Equivariant message passing for the prediction of tensorial properties and molecular spectra. *ICML* 2021.
- Grimme, S. & Hansen, A. (2015). A Practicable Real-Space Measure and Visualization of Static Electron-Correlation Effects. *Angew. Chem. Int. Ed.*, 54, 12308.
- Sayfutyarova, E. R. et al. (2017). Automated Construction of Molecular Active Spaces from Atomic Valence Orbitals. *J. Chem. Theory Comput.*, 13, 4063.
- Angeli, C. et al. (2001). N-electron valence state perturbation theory: a fast implementation of the strongly contracted variant. *Chem. Phys. Lett.*, 350, 297.
