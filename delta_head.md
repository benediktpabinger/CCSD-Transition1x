# Delta Learning Head

End-to-end documentation of the delta correction head trained on top of frozen MACE.

---

## 1. Motivation

MACE (trained on Transition1x) uses wB97X-D3/6-31G(d) as its reference level of theory — a cheap functional with a small basis set chosen for dataset generation speed. wB97M-V/def2-TZVP is a significantly better reference: it uses a larger basis (def2-TZVP vs 6-31G(d)) and a range-separated meta-GGA functional known to be more accurate for reaction barrier heights.

The systematic energy difference between the two levels is:

```
delta = E_wB97M-V/def2-TZVP − E_wB97X-D3/6-31G(d)
```

This difference is large in absolute terms (~3 eV per molecule) but relatively smooth across geometries of the same molecule (~0.09 eV within-reaction std). A lightweight correction head trained to predict `delta` can shift MACE predictions from wB97X-D3 level to wB97M-V level without retraining the full model.

**Why not retrain MACE from scratch?**
MACE training on T1x takes days on multiple GPUs. The delta head trains in minutes on a single GPU, and the MACE encoder — which already learned a rich representation of molecular geometry — is reused entirely.

**Scope of the correction:** The head corrects the DFT functional/basis gap. It does not correct for multireference (MR) character. For strongly MR reactions (rxn7949, rxn8832), the wB97X-D3 training labels are themselves unreliable — the delta head cannot fix errors in the training data.

---

## 2. Data

### Training set
- **500 reactions** randomly sampled from the T1x training split (`train_delta_rxns.txt`)
- **10 geometries per reaction** — uniformly spaced along the T1x NEB trajectory, ensuring reactant, TS, and product regions are always covered
- **wB97M-V/def2-TZVP** single-point energies and gradients computed with ORCA on the cluster (`pipeline/delta/train_delta_sp.py`, `pipeline/delta/job_train_delta_sp.sh`)
- **wB97X-D3/6-31G(d)** energies and forces read directly from the T1x HDF5 file
- **Target per geometry:** `delta_eV = E_wB97M-V − E_wB97X-D3`, `delta_forces = F_wB97M-V − F_wB97X-D3`

### Validation set
- **Group A (174 reactions):** converged NEB reactions from T1x val split. wB97M-V energies from `neb.db`; wB97X-D3 SPs + gradients computed with ORCA. Delta energy available; delta forces not available (wB97M-V forces not stored in `neb.db`).
- **Group B (51 reactions):** failed NEB reactions from T1x val split. wB97M-V SPs + gradients computed with ORCA on T1x geometries; wB97X-D3 read from T1x HDF5. Full delta energy and delta forces available.

### Benchmark set (evaluation only)
- **30 reactions** from the T1x test split — 10 high MR, 10 mid MR, 10 low MR (see `multireference_screening.md`)
- **Zero overlap** with training or validation sets (verified against all three lists)
- Single-point energies and forces computed for all four methods on the 10 final NEB images per reaction (`pipeline/delta/eval_benchmark_sp.py`)

---

## 3. Architecture

The delta head sits on top of the frozen MACE encoder.

```
Geometry
   |
   v
MACE encoder (frozen)
   |
   v
node_feats  [N_atoms x 17408]
   |
   | slice [:, 1024:]
   v
higher-order features  [N_atoms x 16384]
   |
   v
NonLinearReadoutBlock (MLP, SiLU activation)
   |
   v
per-atom delta  [N_atoms x 1]
   |
   | sum over atoms
   v
delta_total  [scalar, eV]
```

**Key parameters:**

| Parameter | Value |
|-----------|-------|
| MACE model | `mace_t1x_p10_compiled.model` |
| Delta head | `delta_head.pt` |
| `NODE_FEATS_OFFSET` | 1024 |
| `HIDDEN_IRREPS` | `"1024x0e + 1024x1o + 1024x2e + 1024x3o"` |
| `MLP_IRREPS` | `"16x0e"` |
| Input to head | `node_feats[:, 1024:]` — 16384-dim higher-order irreps |
| Output | per-atom scalar summed to total delta energy |

**Why `[:, 1024:]` and not all features?**
The first 1024 features are scalar (0e irreps). The remaining 16384 features are higher-order (1o, 2e, 3o) and encode directional/angular information about the local chemical environment. These are more sensitive to geometry changes and better suited for predicting a geometry-dependent correction.

---

## 4. Training

- MACE weights are **frozen** throughout — only the head is trained
- Loss: delta energy MSE (+ delta forces where available, Group B val data)
- Hardware: H100 or H200 GPU required (MACE compiled model uses TorchScript targeting sm_90a)
- Training script: `pipeline/delta/train_delta_head.py`
- SLURM job: `pipeline/delta/job_train_delta_head.sh`

---

## 5. Evaluation

### Method

All four methods evaluated as single points on the same 10 final NEB images per reaction (wB97M-V/def2-TZVP optimised geometries). Barriers computed as `max(E_relative)` over the 10-image profile. Results stored in `eval_benchmark_sp.json` and `full_benchmark_results.json`.

**Comparisons:**
- **Q1:** Does MACE track wB97X-D3 (its training target)?
  `e_mace_eV` vs `e_wb97x_eV`
- **Q2a:** Does MACE+delta track wB97M-V better than MACE alone?
  `e_mace_eV` vs `e_wb97m_eV`  and  `e_delta_eV` vs `e_wb97m_eV`

### Results (30-reaction benchmark)

**Energy — relative profiles (meV), all 30 reactions:**

| Method vs reference | Bias | eMAE | R² |
|---|---|---|---|
| MACE vs wB97X-D3 | — | ~55 | ~0.99 |
| MACE vs wB97M-V | +77 | 108 | 0.973 |
| MACE+delta vs wB97M-V | −5 | 106 | 0.967 |
| wB97X-D3 vs wB97M-V | — | ~95 | — |

**Forces (eV/Å):**

| Method vs reference | Cosine similarity | Force MAE |
|---|---|---|
| MACE vs wB97M-V | 0.324 | 139 |
| MACE+delta vs wB97M-V | 0.412 | 134 |

**Barrier heights (meV), forward, vs CCSD(T):**

| Method | MAE (all 30) | MAE (High MR) | MAE (Low MR) |
|---|---|---|---|
| wB97M-V (NEB) | — | — | — |
| wB97X-D3 (NEB) | ~95 | — | — |
| MACE | ~108 | ~350 | ~80 |
| MACE+delta | ~106 | — | — |

### Interpretation

**Energy bias:** MACE has a +77 meV systematic bias on forward barriers vs wB97M-V. The delta head removes this almost perfectly (−5 meV). This is the primary success.

**eMAE barely improves (108 → 106 meV):** The bias is gone but reaction-to-reaction variance is added — the head corrects by different amounts per reaction, sometimes overshooting.

**Force cosine similarity (0.324 → 0.412):** Delta meaningfully improves force directions toward wB97M-V. Important for NEB, which relies on forces to optimise the path.

**R² slightly worse (0.973 → 0.967):** Energy profile shape is marginally less smooth after correction.

**Pattern by MR group:**
- High MR / Mid MR: delta helps most — genuine functional gap exists and the head corrects it
- Low MR: delta overcorrects — the true gap is small, but the head still applies a correction, adding variance

### Limitations

1. **Does not fix MR errors.** For rxn7949 and rxn8832, MACE barriers are 500–700 meV above CCSD(T). This is a training-data quality problem (wB97X-D3 labels are wrong for strongly MR systems). The delta head corrects the wB97X-D3→wB97M-V gap, not the DFT→CCSD(T) gap.

2. **Force supervision is limited.** Only Group B (51 reactions) provides delta forces for training. Expanding force supervision would likely improve force MAE and cosine similarity further.

3. **Head capacity may be insufficient.** The MLP outputs `16x0e` — a small scalar readout. A larger head or one that also outputs force corrections explicitly might close more of the gap.

4. **Geometry dependence.** The delta is smooth within a reaction but varies between reactions (~0.09 eV std). The head must generalise this geometry-dependent correction to unseen reactions — which it does reasonably, but imperfectly.

---

## 6. Scripts

| File | Purpose |
|------|---------|
| `pipeline/delta/train_delta_sp.py` | Compute wB97M-V SPs on 500 training reactions |
| `pipeline/delta/job_train_delta_sp.sh` | SLURM job for training SPs |
| `pipeline/delta/prepare_delta_data.py` | Assemble delta targets from SP results + T1x HDF5 |
| `pipeline/delta/train_delta_head.py` | Train the delta head (frozen MACE + MLP) |
| `pipeline/delta/job_train_delta_head.sh` | SLURM job for training |
| `pipeline/delta/eval_benchmark_sp.py` | Compute all 4 methods on 30-reaction benchmark |
| `pipeline/delta/job_eval_benchmark_sp.sh` | SLURM job for benchmark eval |
| `pipeline/delta/analyze_full_benchmark.py` | Combine all results into `full_benchmark_results.json` |
| `pipeline/_analyze_benchmark_full.py` | Local analysis: energy/force metrics, barrier comparison |
| `pipeline/_check_nevpt2_plausibility.py` | NEVPT2 reliability check (bottom/middle 20 reactions) |

**Result files:**
- `eval_benchmark_sp.json` — per-image energies and forces for all 4 methods, all 30 reactions
- `full_benchmark_results.json` — per-reaction barrier heights, RMSD, NEVPT2 reliability flags

---

## 7. Key File Paths (cluster)

| Resource | Path |
|----------|------|
| MACE model | `~/mace_t1x_p10_compiled.model` |
| Delta head weights | `~/delta_head/delta_head.pt` |
| Training SP results | `~/delta_cache/` |
| Benchmark eval results | `~/delta_head/eval_benchmark_sp.json` |
| Full benchmark JSON | `~/delta_head/full_benchmark_results.json` |
