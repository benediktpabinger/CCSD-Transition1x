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

## 2. Data (Version 1)

### Training set
- **500 reactions** randomly sampled from the T1x training split (`train_delta_rxns.txt`)
- **10 geometries per reaction** — uniformly spaced along the T1x NEB trajectory
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

## 6. Version 2 — Scaled Training

### Motivation

Version 1 was a proof of concept: 500 reactions × 10 geometries = 5,000 training points. The v1 evaluation showed the head removes the systematic +77 meV energy bias and improves force directions, but eMAE barely improves (108 → 106 meV) and the head struggles on unseen reaction types. The root causes are data quantity and geometry coverage, not architecture — v2 addresses both.

T1x has 9,561 training reactions with ~950 geometries each (9+ million total). V1 used ~0.05% of available data.

---

### 6.1 Data — What to relabel

**Reactions: 5,000 (from 500)**

5,000 reactions are sampled randomly from the T1x training split using `sample_train_reactions.py --n-reactions 5000 --seed 42`. This uses 52% of available training reactions, giving broad coverage of chemical space. The same random seed is used for reproducibility.

Why 5,000 and not all 9,561? Each reaction requires 20 ORCA SP+gradient calculations. At ~10 min per geometry on 8 cores, 5,000 × 20 = 100,000 calculations takes ~3–4 days on the cluster. All 9,561 reactions would roughly double the compute cost with diminishing returns — 5,000 is a practical sweet spot.

**Geometries per reaction: 20 (from 10)**

Each reaction has ~950 geometries in the T1x HDF5 file representing the full NEB path from reactant to product. V1 sampled 10 uniformly — every ~95th frame. V2 uses 20 with stratified sampling (see below).

**Stratified 4-segment sampling**

V1 used `np.linspace(0, n-1, 10)` — purely uniform. This gave no guaranteed coverage of the transition state (TS) region, which is the most important geometry for barrier prediction.

V2 splits the path into 4 segments using the TS index (`argmax(energies_wb97x)`) as a landmark, and draws 5 points from each segment:

```
Energy
  |           TS
  |          /\
  |         /  \
  |        /    \
  |_______/      \________
  R                       P
  0      n//4   ts_idx   ts_idx+(n-ts_idx)//2   n-1
  |        |       |              |               |
  [  seg1  ][  seg2 ][   seg3    ][    seg4       ]
    5 pts    5 pts     5 pts          5 pts
```

- **Seg 1** (R → n//4): reactant valley
- **Seg 2** (n//4 → TS): approach to barrier
- **Seg 3** (TS → midpoint of downhill): immediate post-TS region
- **Seg 4** (midpoint → P): product valley

Because segs 2 and 3 bracket the TS and are shorter in index space but still get 5 points each, sampling is denser near the barrier. The TS index is always a segment boundary, so it is always a sampled point. Sampling is index-based (not path-length-weighted) for simplicity — sufficient given the ~950 uniformly spaced T1x frames.

**ORCA settings**

`wB97M-V/def2-TZVP` with `EnGrad` (energy + gradient), `RIJCOSX` density fitting, `TightSCF`. Key change from v1: `nprocs 8` (was 1) — uses all 8 CPUs allocated per SLURM job, reducing per-SP walltime from ~10 min to ~2–3 min. All 100,000 geometries have both energy and force labels (delta forces = F_wB97M-V − F_wB97X-D3).

---

### 6.2 Architecture changes

| Parameter | V1 | V2 | Reason |
|---|---|---|---|
| `MLP_IRREPS` | `16x0e` | `64x0e` | V1 head was severely capacity-limited — 16 scalar channels as a bottleneck after a 16384-dim input. 4× wider is justified by 20× more data and adds negligible compute overhead. |
| `batch_size` | 32 | 64 | More data means larger batches give a better gradient estimate per step. Fits comfortably in GPU memory. |

Everything else unchanged: `HIDDEN_IRREPS`, `NODE_FEATS_OFFSET`, frozen MACE backbone, SiLU activation.

---

### 6.3 Training — force weight sweep

The training loss is:

```
loss = loss_energy + force_weight × loss_forces
```

Both terms use Huber loss with `delta=0.1 eV`. In v1, `force_weight=1.0` was used without tuning. In v2, all 100,000 training points have force labels, making the force loss a stronger signal — but the right balance between energy and force supervision is not obvious a priori.

Three parallel training runs are submitted as a SLURM array (`job_train_delta_head.sh`, array 0–2):

| Run | `force_weight` | Output |
|---|---|---|
| 0 | 0.5 | `delta_head_fw0.50.pt` |
| 1 | 1.0 | `delta_head_fw1.00.pt` |
| 2 | 2.0 | `delta_head_fw2.00.pt` |

Each run trains for up to 200 epochs with `ReduceLROnPlateau` (patience=10, factor=0.5), stopping early when LR drops below 1e-6. All three are tracked in W&B (`transition1x-delta` project). The best checkpoint is selected by validation loss.

Why 0.5 / 1.0 / 2.0? These span a 4× range around the neutral default. `force_weight < 1` prioritises energies (barrier heights). `force_weight > 1` prioritises force directions (NEB convergence). The three runs cost ~3 × 2h = 6h GPU time in parallel — cheap relative to the ORCA data collection.

---

### 6.4 Submission notes

The DTU cluster (xeon24el8) limits array jobs to 1,000 tasks and 1,000 simultaneously submitted jobs. The 5,000 SP jobs are therefore split into 5 batches of 1,000, submitted sequentially as the queue clears:

```bash
sbatch --array=0-999 --export=OFFSET=0    pipeline/delta/job_train_delta_sp.sh  # rxn 1–1000
sbatch --array=0-999 --export=OFFSET=1000 pipeline/delta/job_train_delta_sp.sh  # rxn 1001–2000
sbatch --array=0-999 --export=OFFSET=2000 pipeline/delta/job_train_delta_sp.sh  # rxn 2001–3000
sbatch --array=0-999 --export=OFFSET=3000 pipeline/delta/job_train_delta_sp.sh  # rxn 3001–4000
sbatch --array=0-999 --export=OFFSET=4000 pipeline/delta/job_train_delta_sp.sh  # rxn 4001–5000
```

Each batch takes ~30–45 min wall time (1,000 reactions × 20 SPs × 2–3 min / 8 cores). The `OFFSET` variable shifts the reaction list index so each batch reads a different slice of `train_delta_rxns.txt`.

The skip logic in `train_delta_sp.py` checks `n_sampled >= args.n_images` — any reaction already computed with 20+ geometries is skipped, so batches can be safely resubmitted on failure.

---

### 6.5 Full v2 workflow

```
1. sample_train_reactions.py --n-reactions 5000 --seed 42
      → ~/ccsd_dataset/train_delta_rxns.txt

2. sbatch job_train_delta_sp.sh (×5 batches, ~3–4 days total)
      → ~/train_delta_sp/{rxn}/results.json  for all 5000 reactions

3. python prepare_delta_data.py
      → ~/delta_cache/train.pt   (~100k geometries)
      → ~/delta_cache/val.pt

4. sbatch job_train_delta_head.sh  (array 0–2, 3 parallel runs)
      → ~/delta_head/delta_head_fw0.50.pt
      → ~/delta_head/delta_head_fw1.00.pt
      → ~/delta_head/delta_head_fw2.00.pt

5. Compare W&B runs → select best force_weight → use that head
```

---

### 6.6 Version 2 — Training details (as implemented)

#### Data selection — 5000 reactions × 20 geoms

5000 reactions sampled from the T1x train split (`sample_train_reactions.py --n-reactions 5000 --seed 42`). Seed fixed for reproducibility.

20 geoms per reaction via **stratified 4-segment sampling** — the TS index (`argmax(wB97X energies)`) splits the path into 4 segments, 5 points drawn from each:

```
seg1: [0,       n//4)                          reactant valley
seg2: [n//4,    ts_idx)                        approach to barrier
seg3: [ts_idx,  ts_idx + (n-ts_idx)//2)       immediate post-TS
seg4: [ts_idx + (n-ts_idx)//2,  n-1]          product valley
```

The TS is always a segment boundary and is therefore always sampled. Segs 2 and 3 bracket the TS and are shorter in index space but still receive 5 points each → denser coverage near the barrier. Edge case: if `ts_idx < n//4`, seg 2 runs backward → 16–17 unique points instead of 20.

**Why stratified over uniform (v1):** uniform `linspace(0, n-1, 10)` gave no TS guarantee with only 10 points. With 20 points and 4 segments, the TS and both flanking regions are always represented.

**ORCA:** `wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad`, 8 nprocs, 4000 MB maxcore. All 20 geoms per reaction get energy + forces (unlike v1 where only energy was used for training). Scratch → `/home/scratch3/s242862` (not NFS — cluster admin requirement), `TMPDIR=/tmp`.

**Actual result:** 4997 reactions × ~20 geoms = **80,592 training geoms**. 3 reactions filtered by `prepare_delta_data.py` for having < 15 sampled geometries (artefact of batch 1 running before the scratch fix, where some jobs wrote partial output).

---

#### Data processing — `prepare_delta_data.py`

- Reads each `~/train_delta_sp/{rxn}/results.json`, loads positions from T1x HDF5, assembles `{rxn, geom_idx, positions, atomic_numbers, delta_eV, delta_forces}`.
- Filters reactions with `< 15` geoms (catches partial runs).
- Val A positions loaded from `neb.db` (NEB-optimised geometries, not T1x). Val B from T1x HDF5.
- Val A force labels: reads `delta_forces_eV_per_ang` from `results.json` if present (populated by `compute_val_a_forces.py` for the last 10 images). Entries without force labels store `delta_forces: None`.
- Outputs `~/delta_cache/train.pt` and `~/delta_cache/val.pt` — Python lists of dicts, loadable with `torch.load`.

---

#### Architecture change: `MLP_irreps` 16x0e → 64x0e

V1 bottlenecked a 16384-dim input through 16 scalar channels — severely capacity-limited. V2 uses 64 channels (4× wider). Justified by 20× more training data; adds negligible compute overhead at inference.

---

#### Training parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 200 (early stop when lr < 1e-6) |
| Batch size | 64 |
| Optimizer | Adam, lr=1e-3 |
| Scheduler | ReduceLROnPlateau(patience=10, factor=0.5) |
| Loss | Huber(delta_e) + force_weight × Huber(delta_f), δ=0.1 eV |
| Force weight | sweep: 0.5, 1.0, 2.0 (3 parallel runs) |
| W&B | project `transition1x-delta` — logs train_loss, train_e, train_f, val_loss, val_e, val_f, val_f_e, val_f_f, lr per epoch |

---

#### Validation design — two passes per epoch

Val A originally had no force labels (neb.db only stores energy). Val B had forces but only 51 reactions — insufficient diversity for reliable force quality assessment. To get meaningful force validation on NEB-like geometries, wB97M-V EnGrad was computed for the **last 10 NEB images of all 174 Val A reactions** (`compute_val_a_forces.py`, job 10485544). This yields ~1,740 force-labeled Val A geoms on converged MEP geometries — the most representative of actual inference conditions.

Two val passes each epoch:

```python
val_sample   = random.sample(val_data, 1024)                              # fixed at epoch 1
val_f_sample = [s for s in val_data if s['delta_forces'] is not None]    # ~2240 geoms
```

- `val_sample` → `val_loss` → drives `ReduceLROnPlateau`
- `val_f_sample` → `val_f_f` → drives **checkpoint saving**

**Why two separate val sets?** The LR scheduler needs a stable, broad signal to decide when the model has stopped improving — `val_loss` on the 1024-geom mixed sample provides this, covering diverse reactions and geometry types (energy-only and force-labeled alike). `val_f_sample` is more targeted (NEB-path geometries only) but noisier epoch-to-epoch; using it for the scheduler would cause premature LR reductions. The split keeps concerns separate: the scheduler asks "is the model still converging?" while checkpoint saving asks "is the model good for NEB forces?"

**Why checkpoint by force loss, not val_loss:** `val_loss` is dominated by the energy term both in magnitude and in the ratio of energy-only to force-labeled geoms in `val_sample`. The energy offset (~3 eV) converges quickly; force quality matters more for NEB. Saving by `val_f_f` on the dedicated force val set selects the checkpoint that best predicts force directions on NEB-path geometries.

---

#### `--resume` support

`train_delta_head.py` accepts `--resume <path.pt>` — loads the state dict and continues training from that checkpoint. Allows iterative fine-tuning if a first run converges but isn't good enough, without restarting from scratch.

---

## 7. Scripts

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
