# Delta Head — Status as of 2026-05-19

## What it is

A `NonLinearReadoutBlock` head trained on top of frozen MACE-T1x to predict:

    delta(geometry) = E_wB97M-V/def2-TZVP − E_wB97X-D3/6-31G(d)

At inference: `E_wB97M_predicted = E_wB97X_MACE + delta_head(node_feats)`

Input to head: `node_feats[:, 1024:]` — the 16384-dim output of MACE interaction 2.
Architecture: `NonLinearReadoutBlock(irreps_in="1024x0e+1024x1o+1024x2e+1024x3o", MLP_irreps="16x0e", gate=silu)`

---

## Training run 1 — completed (job 10368920, wandb "clean-star-4")

- **Data**: 500 train reactions × 10 geoms = 5,000 structures. Val: fixed 1024-geom sample from Group A (energy-only, no forces).
- **Config**: 200 epochs, batch 32, lr 1e-3, Huber δ=0.1, force_weight=1.0, patience=10.
- **Result**: Final val_e ≈ 0.0139 eV (best checkpoint). LR decayed to ~1e-5 by end.
- **Model saved**: `~/delta_head/delta_head.pt` on cluster.

## Evaluation results (job 10383303, 20 val reactions, 980 geoms)

| Method             | MAE (meV) | vs. MACE alone |
|--------------------|-----------|----------------|
| MACE alone         | ~2980     | baseline       |
| MACE + delta head  | ~154      | −94%           |

Plots saved locally at [delta_plots/](delta_plots/) and on cluster at `~/delta_head/plots/`:
- `scatter.png` — predicted vs true delta_eV: good correlation, slight positive bias
- `error_hist.png` — error distribution: right-skewed, mean=143 meV, median=126 meV
- `per_rxn_mae.png` — per-reaction MAE range: 57 meV (rxn1299) to 241 meV (rxn7544)
- `reaction_paths.png` — delta along NEB path for best 2 + worst 2 reactions

**Key observation**: The head systematically underpredicts |delta| by ~100–150 meV.
Errors are almost all positive (predicted correction smaller than true correction).

---

## Data status (as of 2026-05-19)

| Dataset     | Reactions | Results done | Forces available |
|-------------|-----------|--------------|------------------|
| Train       | 500       | 500 (100%)   | Yes (all)        |
| Val Group A | 174       | 174 (100%)   | No (neb.db only has energy) |
| Val Group B | 51        | 39           | 10 with forces   |

Val Group B was submitted as job 10367659 (array 0-9, 10 reactions). Status at last
check: 39/51 done, 10 with delta_forces. Not yet complete.

---

## What to do next (in priority order)

### 1. Retrain with force supervision on a proper val set
Once Val B finishes (need ~51 results with forces), retrain using:
- Train loss: Huber on delta_energy + Huber on delta_forces (weight 1.0)
- Val: use Group B reactions (forces available) for honest force validation
- Keep Group A as a held-out test set only

This is the main lever for improving the systematic underprediction bias — the force
loss anchors the PES shape, not just the total energy offset.

### 2. Fix the systematic positive bias
The head underpredicts |delta| by ~130 meV on average. Possible causes:
- Training only on energies (no force gradient signal on the spatial variation)
- Huber loss with δ=0.1 eV may be clipping the larger errors that carry the bias
- Head capacity may be insufficient (16x0e MLP — very small)

Candidates to try: increase MLP_irreps to "64x0e", lower huber_delta to 0.05, add force weight.

### 3. Speed up epoch time
Graph building (`AtomicData.from_config`) on CPU takes ~20 min/epoch at batch 32 for 5000 structures.
Fix: pre-cache `AtomicData` graphs as `.pt` files during `prepare_delta_data.py`, load directly during training.

### 4. Checkpoint resumption
Current training starts from scratch each run. Add `--resume` flag to `train_delta_head.py`
to load the best checkpoint and continue from there.

---

## Key API pitfalls (hard-won, do not forget)

1. **PyTorch 2.6 + e3nn**: `torch.serialization.add_safe_globals([slice])` MUST come before `from e3nn import o3` in every script — otherwise e3nn's `constants.pt` fails to load.

2. **TorchScript compiled model**: needs `Dict[str, Tensor]`, NOT a `Batch` object.
   Always convert: `batch_dict = {key: batch[key] for key in batch.keys}`

3. **`batch.keys` is a property** (list), not a method — no `()`.

4. **`torch.enable_grad()` inside `forward_delta`**: even during validation, forces need
   a live autograd graph. Wrap the entire forward body in `with torch.enable_grad():`.

5. **SLURM stdout buffering**: always use `python3 -u` in SLURM scripts.

6. **MACE model path**: `~/mace_t1x_p10_compiled.model` (TorchScript, CUDA required).
   NOT `mace_t1x_p10_run-123.model` — that file does not exist.

---

## File map

### Cluster
| Path | Description |
|------|-------------|
| `~/mace_t1x_p10_compiled.model` | TorchScript MACE (CUDA required) |
| `~/delta_head/delta_head.pt` | Best checkpoint from run 1 |
| `~/delta_cache/train.pt` | 5000-geom train set (list of dicts) |
| `~/delta_cache/val.pt` | ~8700-geom val pool (Groups A+B) |
| `~/delta_head/plots/` | PNG plots from eval run |
| `~/logs/` | SLURM logs (train_delta_head_*.log, eval_delta_*.log, plot_delta_*.log) |

### Local (pipeline/delta/)
| File | Status |
|------|--------|
| `prepare_delta_data.py` | Done |
| `train_delta_head.py` | Done — uses fixed val sample, wandb, ReduceLROnPlateau |
| `job_train_delta_head.sh` | Done |
| `eval_delta_head.py` | Done — prints MACE alone vs MACE+delta comparison |
| `job_eval_delta_head.sh` | Done |
| `plot_delta_eval.py` | Done — generates 4 PNGs to ~/delta_head/plots/ |
| `job_plot_delta_eval.sh` | Done |
