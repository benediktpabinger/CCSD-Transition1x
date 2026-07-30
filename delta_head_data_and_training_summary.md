# Delta head — data, architecture, training (quick reference)

> **Snapshot summary**, pulled directly from `pipeline/delta/train_delta_head.py`,
> `pipeline/delta/prepare_delta_data.py`, `pipeline/delta/train_delta_sp.py`, and
> the actual SLURM training logs (`logs/delta_head_v2_10502811_*.log`). For full
> background, motivation, and v1 vs v2 history, see `delta_head.md` — this file
> is the condensed, verified-against-logs version.

## What it predicts

A small correction head bolted onto a **frozen** MACE model
(`mace_t1x_p10_compiled.model`, r_max=6.0 Å, elements H/C/N/O/F):

```
delta(geometry) = E_wB97M-V/def2-TZVP - E_wB97X-D3/6-31G(d)
```

It only reads MACE's internal node features — it never sees positions directly;
delta forces are obtained via autograd through the head's energy output, not
predicted directly.

## Architecture

- `NonLinearReadoutBlock` (from MACE), reading the **second interaction layer's**
  node features: `HIDDEN_IRREPS = 1024x0e + 1024x1o + 1024x2e + 1024x3o`,
  sliced at `NODE_FEATS_OFFSET = 1024` (skips the first interaction's scalar-only
  output, keeps the higher-order 1o/2e/3o features which carry directional info)
- `MLP_IRREPS = 64x0e` → **65,600 trainable parameters**
- Per-atom energy contribution, summed to a per-structure delta

## Data selection

- **Train**: 5,000 reactions from the T1x train split (`sample_train_reactions.py
  --n-reactions 5000 --seed 42`), **stratified 4-segment sampling** around the TS
  per reaction:
  ```
  seg1: [0,        n//4)                      reactant valley
  seg2: [n//4,     ts_idx)                    approach to barrier
  seg3: [ts_idx,   ts_idx+(n-ts_idx)//2)       immediate post-TS
  seg4: [ts_idx+(n-ts_idx)//2,  n-1]           product valley
  ```
  5 points per segment (20 geoms/reaction target), TS index always a segment
  boundary so always sampled. wB97M-V references computed via fresh ORCA
  EnGrad single points; wB97X-D3 references read directly from the T1x HDF5.
- **Actual cached pool**: **80,592 training geometries** (4,997 reactions ×
  ~20 geoms; 3 reactions dropped for <15 sampled geoms, a partial-run artifact),
  **10,600 validation geometries**.
- **Validation composition**: 174 converged-NEB val reactions (Group A,
  energy-only labels — wB97M-V forces aren't stored in `neb.db`) + 51
  failed-NEB val reactions (Group B, full energy + force labels). Of the
  10,600 val geoms, **2,240 have force labels**.

## Training

- Loss: `Huber(delta_e, δ=0.1) + force_weight × Huber(delta_f, δ=0.1)`
- `force_weight ∈ {0.5, 1.0, 2.0}` swept as 3 parallel SLURM array tasks
- Adam, lr=1e-3, `ReduceLROnPlateau(patience=10, factor=0.5)`, batch size 64
- 10,000 geoms randomly resampled from the 80,592-geom pool **each epoch**
  (not the full pool every epoch)
- Two validation passes per epoch: a fixed 1,024-geom mixed sample drives the
  LR scheduler (`val_loss`); the 2,240-geom force-labeled subset drives
  **checkpoint saving** (best `val_f_f`) — decoupled because the LR scheduler
  needs a stable broad signal while checkpointing should optimize specifically
  for NEB-relevant force quality
- All three runs converged (LR decayed below 1e-6) around epoch 130-138

**Final validation losses (from actual training logs):**

| force_weight | val energy loss | val force loss | total val loss |
|---|---|---|---|
| 0.5 | 0.0109 | 0.0050 | 0.0135 |
| 1.0 | 0.0112 | 0.0039 | 0.0150 |
| 2.0 | 0.0112 | 0.0037 | 0.0186 |

fw=2.0 has the lowest force-loss component (why it was picked for NEB use,
where forces drive the geometry search) but the worst *energy* loss of the
three — a trade-off that foreshadowed the later finding that fw=2.0's
NEB-derived TS geometries are actually worse than bare MACE's (see
`mr_casscf_optts_status_2026_06_16.md` and the delta-ablation results from
2026-06-22).

## Checkpoints

| File | force_weight | MLP width |
|---|---|---|
| `delta_head.pt` | 1.0 (v1, no sweep) | 16x0e (old architecture) |
| `delta_head_fw0.50.pt` | 0.5 | 64x0e |
| `delta_head_fw1.00.pt` | 1.0 | 64x0e |
| `delta_head_fw2.00.pt` | 2.0 | 64x0e — used in the MACE+delta NEB benchmark |
