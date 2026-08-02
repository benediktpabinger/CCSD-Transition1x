# Delta Learning Head

Method documentation for the delta correction head trained on top of frozen MACE:
motivation, architecture, data, and training.

**Results are not in this document.** All evaluation numbers — energies, forces, barriers, NEB
geometry, the comparison against OMol25 models — live in
[`delta_head_v2_metric_definitions.md`](delta_head_v2_metric_definitions.md), with machine-readable
values in `delta_head_v2_eval_numbers.json`.

Everything below describes the **v2 head** (`delta_head_fw2.00.pt`, `64x0e`, 5,000 reactions),
which is the head in use. Where v1 details are retained for context they are explicitly marked;
v1 results have been removed rather than carried forward, since two of their headline conclusions
are contradicted by v2. Git history holds the original text.

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

### Training set (v2)

- **5,000 reactions** sampled from the T1x training split
  (`sample_train_reactions.py --n-reactions 5000 --seed 42`); 4,997 survive filtering
- **20 geometries per reaction** via stratified 4-segment sampling around the TS — see §6.1
- **wB97M-V/def2-TZVP** energies **and gradients** for every geometry (ORCA EnGrad)
- **wB97X-D3/6-31G(d)** energies and forces read from the T1x HDF5 file
- **Target per geometry:** `delta_eV = E_wB97M-V − E_wB97X-D3`,
  `delta_forces = F_wB97M-V − F_wB97X-D3`
- **Actual total: 80,592 training geometries**, all carrying both energy and force labels

*v1 for context: 500 reactions × 10 uniformly spaced geometries = 5,000 points, energy-only
supervision except on val Group B. v2 is ~16× more data with full force supervision.*

### Validation set

- **Group A (174 reactions):** converged NEB reactions from the T1x val split. wB97M-V energies
  from `neb.db`; wB97X-D3 SPs + gradients from ORCA. Force labels were added for the last 10 NEB
  images of all 174 reactions by `compute_val_a_forces.py`, giving ~1,740 force-labelled geometries
  on converged MEP structures.
- **Group B (51 reactions):** failed NEB reactions from the T1x val split. wB97M-V SPs + gradients
  computed on T1x geometries; wB97X-D3 from T1x HDF5. Full delta energy and delta forces.
- **Totals:** 10,600 validation geometries, of which 2,240 carry force labels.

### Benchmark set (evaluation only)

- **30 reactions** from the T1x test split — 10 high MR, 10 mid MR, 10 low MR
  (see `multireference_screening.md`)
- **Zero overlap** with training or validation sets (verified against all three lists)
- A 22-reaction **RKS-stable subset** of these 30 is used for the comparison against OMol25 models;
  see the results document.

⚠ The training data is sampled from **T1x wB97X-D3 path geometries**, while evaluation is on
**ORCA wB97M-V path geometries**. Both are reaction-path structures, so the shift is mild and
arguably closer to deployment — but this is not an in-distribution evaluation.

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
| MACE model | `mace_t1x_p10_compiled.model` (frozen) |
| Delta head | `delta_head_fw2.00.pt` |
| `NODE_FEATS_OFFSET` | 1024 |
| `HIDDEN_IRREPS` | `"1024x0e + 1024x1o + 1024x2e + 1024x3o"` |
| `MLP_IRREPS` | `"64x0e"` *(v1 used `16x0e` — see §6.2)* |
| Input to head | `node_feats[:, 1024:]` — 16384-dim higher-order irreps |
| Output | per-atom scalar summed to total delta energy |
| Trainable parameters | 65,600 |

Forces are obtained by **analytic autograd through the head**, not finite differences, so the
corrected surface is conservative by construction:

```python
delta_f = -torch.autograd.grad(per_struct.sum(), positions)[0]
f_corrected = f_mace + delta_f
```

**Why `[:, 1024:]` and not all features?**
The first 1024 features are scalar (0e irreps). The remaining 16384 features are higher-order (1o, 2e, 3o) and encode directional/angular information about the local chemical environment. These are more sensitive to geometry changes and better suited for predicting a geometry-dependent correction.

---

## 4. Training — overview

- MACE weights are **frozen** throughout; only the head is trained
- Loss: `Huber(delta_e) + force_weight × Huber(delta_f)`, δ = 0.1 eV — every training geometry
  carries both labels in v2
- `force_weight` selected by sweep over {0.5, 1.0, 2.0}; **fw = 2.0 chosen** on lowest validation
  force loss (`val_f_f` = 0.0037 eV/Å). See §6.3.
- Hardware: H100 or H200 GPU required (MACE compiled model uses TorchScript targeting sm_90a)
- Training script: `pipeline/delta/train_delta_head.py`
- SLURM job: `pipeline/delta/job_train_delta_head.sh`

Full as-implemented detail — sampling, data processing, training parameters, the two-pass
validation design — is in §6.

---

## 5. Evaluation

**Results live in [`delta_head_v2_metric_definitions.md`](delta_head_v2_metric_definitions.md).**
That document carries every number together with its exact metric definition, the source line that
produced it, and its caveats — energies, forces, barriers against wB97M-V and CCSD(T), NEB-driven
geometry, MR-tier stratification, and the fixed-geometry comparison against the OMol25 models
(UMA-S, UMA-M, eSEN). Machine-readable values are in `delta_head_v2_eval_numbers.json`.

**Protocol, in brief.** All methods are evaluated as single points on the same 10 final images of
each converged ORCA wB97M-V/def2-TZVP CI-NEB band. Barriers are `max(E_relative)` over that
10-image profile. Energies are compared as **image-0-anchored relative profiles**, each method
anchored to its own first image, so constant offsets between levels of theory cancel.

The two questions it answers:

- **Q1** — does MACE track wB97X-D3, its own training target?
- **Q2** — does MACE+delta track wB97M-V better than MACE alone?

Raw per-image data: `eval_benchmark_sp_fw2_full.json` (30-reaction benchmark),
`eval_sp_rks_stable.json` (RKS-stable subset with OMol25 models).

### Scope limitations of the correction

These are properties of the approach and are not fixed by scaling:

1. **It does not fix multireference errors.** The head corrects the wB97X-D3 → wB97M-V functional
   and basis gap, not the DFT → CCSD(T) gap. Where the wB97X-D3 training labels are themselves
   unreliable — strongly MR reactions such as rxn7949 and rxn8832 — the head has nothing to learn
   from and cannot repair the underlying data. This is the single most important boundary on what
   delta-learning can deliver here.

2. **It corrects the systematic offset, not the reaction-to-reaction variance.** The head removes
   the bulk of MACE's systematic bias, but the residual scatter — over- and under-correcting by
   different amounts on different reactions — is not predictable from the features it sees.

3. **Geometry dependence.** The delta is smooth within a reaction but varies between reactions
   (~0.09 eV std). Generalising a geometry-dependent correction to unseen reactions is the core
   difficulty, and it is where the remaining error lives.

4. **Improving pointwise energies does not imply a better optimisation surface.** A learned
   correction can sharpen energies and forces at fixed geometries while roughening the landscape
   an optimiser has to traverse. See the NEB-driven results, and the practical consequence for how
   the head should be deployed.

---

## 6. Version 2 — Scaled Training

### Motivation

Version 1 was a proof of concept: 500 reactions × 10 geometries = 5,000 training points, with a
`16x0e` readout bottlenecking a 16384-dimensional input. It removed MACE's systematic energy bias
but left the error magnitude essentially unchanged, and force supervision was available on only 51
validation reactions. The limits were data quantity, geometry coverage, and head capacity — v2
addresses all three.

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

**Data generation and training**

| File | Purpose |
|------|---------|
| `pipeline/delta/sample_train_reactions.py` | Sample the 5,000 training reactions (seed 42) |
| `pipeline/delta/train_delta_sp.py` | wB97M-V EnGrad SPs on training reactions |
| `pipeline/delta/job_train_delta_sp.sh` | SLURM array job for training SPs (5 × 1,000) |
| `pipeline/delta/compute_val_a_forces.py` | Add force labels to val Group A NEB images |
| `pipeline/delta/prepare_delta_data.py` | Assemble delta targets from SP results + T1x HDF5 |
| `pipeline/delta/train_delta_head.py` | Train the delta head (frozen MACE + MLP) |
| `pipeline/delta/job_train_delta_head.sh` | SLURM job, array 0–2 for the force-weight sweep |

**Evaluation**

| File | Purpose |
|------|---------|
| `pipeline/delta/eval_benchmark_sp_fw2.py` | v2 fixed-geometry eval, 30-reaction benchmark |
| `pipeline/delta/job_eval_benchmark_sp_fw2.sh` | SLURM job for the above |
| `pipeline/delta/eval_sp_rks_stable.py` | 7-method comparison incl. UMA/eSEN, RKS-stable subset |
| `pipeline/delta/job_eval_sp_rks_stable.sh` | SLURM job for the above |
| `pipeline/mace_delta_neb.py` | NEB driven by MACE or MACE+delta |
| `pipeline/delta/analyze_full_benchmark.py` | Combine results into `full_benchmark_results.json` |
| `pipeline/_analyze_benchmark_full.py` | Local analysis: energy/force metrics, barrier comparison |

**Result files**

| File | Contents |
|------|----------|
| `eval_benchmark_sp_fw2_full.json` | v2 per-image energies + forces, 30 reactions × 10 images |
| `eval_sp_rks_stable.json` | 7 methods × 22 RKS-stable reactions, per-image energies + forces |
| `full_benchmark_results.json` | Per-reaction barriers (fixed-geometry and NEB-driven), all methods |
| `delta_head_v2_eval_numbers.json` | Aggregated metrics, keyed by step/method/metric |

*`eval_benchmark_sp.json` is the superseded v1 evaluation, retained only as a cross-check —
its bare-MACE columns match v2 to 0.05 meV, confirming the frozen base model is unchanged.*

---

## 8. Key File Paths (cluster)

| Resource | Path |
|----------|------|
| MACE model (frozen) | `~/mace_t1x_p10_compiled.model` |
| **Delta head weights (in use)** | `~/delta_head/delta_head_fw2.00.pt` |
| Sweep checkpoints | `~/delta_head/delta_head_fw0.50.pt`, `delta_head_fw1.00.pt` |
| v1 head (superseded) | `~/delta_head/delta_head.pt` |
| Training SP results | `~/train_delta_sp/{rxn}/results.json` |
| Prepared training cache | `~/delta_cache/train.pt`, `~/delta_cache/val.pt` |
| Benchmark eval (v2) | `~/delta_head/eval_benchmark_sp_fw2_full.json` |
| RKS-stable eval | `~/delta_head/eval_sp_rks_stable.json` |
| Full benchmark JSON | `~/delta_head/full_benchmark_results.json` |
| ORCA reference NEB bands | `~/orca_neb_results/{rxn}/neb.db` |
| wB97X-D3 EnGrad SPs | `~/mr_benchmark/orca_engrad/{rxn}/geom_NNNN/` |
| MACE+delta NEB (fw2) | `~/mace_delta_neb_results_fw2/{rxn}/` |
| OMol25 checkpoints | `~/checkpoints/uma-s-1p2.pt`, `uma-m-1p1.pt`, `esen_sm_conserving_all.pt` |
