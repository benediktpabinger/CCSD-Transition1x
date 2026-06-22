# Session Summary — 2026-05-12

## What we were doing

Building a **delta correction head** on top of a frozen MACE model trained on Transition1x.
The delta head learns: delta(geometry) = E_wB97M-V/def2-TZVP − E_wB97X-D3/6-31G(d)
At inference: E_wB97M_predicted = E_wB97X_MACE + delta_head(node_feats)

---

## MACE model architecture (confirmed from state dict)

Model file: `~/mace_t1x_p10_run-123.model` (TorchScript, needs CUDA)
State dict: `~/mace_t1x_p10_run-123_epoch-362.pt`

- Class: `ScaleShiftMACE`
- `num_interactions`: 2
- `hidden_irreps`: `1024x0e + 1024x1o + 1024x2e + 1024x3o` (dim = 16384)
- First interaction outputs scalars only: `1024x0e` (dim = 1024)
- `node_feats_out = cat([1024-dim, 16384-dim]) = 17408-dim` total
- `r_max`: 6.0 Å, `num_radial_basis`: 16
- Trained with: `--num_channels=1024 --max_L=3 --num_interactions=2 --num_radial_basis=16`
- All hyperparameters came from CLI args to `mace_run_train` in SLURM job — no config files

### readouts:
- `readouts[0]`: `LinearReadoutBlock` — reads from interaction 0 output (1024-dim scalars)
- `readouts[1]`: `NonLinearReadoutBlock` — reads from interaction 1 output (16384-dim)
  - `linear_1`: [16384] → [16] (projects to `16x0e` scalars)
  - activation: silu
  - `linear_2`: [16] → [1] (per-atom energy)

### Forward pass (from models.py):
```python
node_feats_concat = []
for interaction, product in zip(interactions, products):
    node_feats = interaction(node_feats, ...)
    node_feats = product(node_feats, ...)
    node_feats_concat.append(node_feats)

node_feats_out = torch.cat(node_feats_concat, dim=-1)  # [N_atoms, 17408]
output["node_feats"] = node_feats_out
output["energy"] = ...  # wB97X-D3 prediction
```

---

## How MACE was patched for T1x training

Only 3 lines were added across 2 files. The patch adds `--max_samples_per_epoch` to enable
epoch-level subsampling via `RandomSampler` (T1x train set has ~10M geometries, too large for
full epochs).

**Local copies** (edited here, deployed to cluster):
- `pipeline/mace/arg_parser.py` → `~/.local/lib/python3.13/site-packages/mace/tools/arg_parser.py`
- `pipeline/mace/run_train.py` → `~/.local/lib/python3.13/site-packages/mace/cli/run_train.py`
- `pipeline/mace/scripts_utils.py` → same

**Deployed via**: `pipeline/deploy_mace_patches.py` (paramiko SFTP, overwrites installed package)

**Earlier approach** (still on cluster): `~/patch_mace.py` and `~/patch_mace2.py` — used
string find/replace directly on the installed files. Now superseded by the deploy script approach.

---

## Delta model training data pipeline

### Strategy decided:
- 500 training reactions (5.2% of 9,561 train reactions), randomly sampled with seed=42
- 10 geometries per reaction, **uniformly spaced** along T1x NEB trajectory
- wB97M-V/def2-TZVP SPs + **EnGrad** (forces) via ORCA
- wB97X-D3 energy + forces read from T1x HDF5 (already there)
- delta_energy = E_wB97M − E_wB97X
- delta_forces = F_wB97M − F_wB97X
- Total: 5,000 ORCA SPs

**Why uniform spacing (not random)?** T1x geometries are ordered along NEB trajectory.
Uniform spacing guarantees coverage of reactant, TS, and product regions. Random sampling
with only 10 points risks clustering (e.g. 7 geometries near reactant, none near TS).
TODO: benchmark uniform vs random once head is trained.

**Why 500 reactions × 10 geometries (not 100 reactions × 50)?**
Middle ground — more chemical diversity than 100 reactions, more within-reaction PES
coverage than 1 geometry per reaction over all 9,561.

### Files written:
- `pipeline/delta/sample_train_reactions.py` — samples 500 reactions, writes `~/ccsd_dataset/train_delta_rxns.txt`
- `pipeline/delta/train_delta_sp.py` — runs wB97M-V+EnGrad SPs, reads wB97X forces from T1x HDF5
- `pipeline/delta/job_train_delta_sp.sh` — SLURM array 0-499, 3h limit, 8 CPUs

### Val data (already computed, now being rerun with forces):
- **Group A** (174 converged NEB): wB97X-D3+EnGrad SPs on 50 NEB images. wB97M-V energy from neb.db.
  **Limitation**: wB97M-V forces NOT stored in neb.db. Only delta_energy available for Group A.
- **Group B** (51 failed NEB): wB97M-V+EnGrad SPs on 50 T1x geometries. wB97X forces from T1x HDF5.
  Full delta_energy + delta_forces available.

### SLURM jobs currently running (as of session end):
- **10360230** — 500 train reactions, rerun with EnGrad (3h limit). ~126 COMPLETED, ~66 RUNNING,
  rest pending. Estimated ~15-20 more hours to completion.
- **10360231** — 174 val Group A reactions, rerun with EnGrad (4h limit). Running.
- **10360232** — 51 val Group B reactions, 10h limit. **CANCELLED** to free up CPU quota for train.
  First ~20 tasks ran briefly before cancellation. Will need to be resubmitted after train completes.

### Results.json format (with forces):
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
Old results.json (energy only) are automatically detected and deleted for rerun.

---

## Delta head design (agreed, not yet implemented)

### Architecture:
```python
from e3nn import o3
from mace.modules.blocks import NonLinearReadoutBlock
import torch.nn.functional as F

hidden_irreps = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_irreps    = o3.Irreps("16x0e")

delta_head = NonLinearReadoutBlock(
    irreps_in  = hidden_irreps,
    MLP_irreps = MLP_irreps,
    gate       = F.silu,
)
# Input: node_feats[:, 1024:] — last 16384 dims from MACE
# Output: per-atom scalar → summed to per-structure delta_energy
```

### Inference pattern:
```python
# Freeze backbone
for p in mace_model.parameters():
    p.requires_grad_(False)

# Forward
batch['positions'].requires_grad_(True)
out = mace_model(batch, training=False, compute_force=False)

node_feats        = out['node_feats'][:, 1024:]
e_wb97x           = out['energy']
per_atom_delta    = delta_head(node_feats)
per_struct_delta  = scatter_sum(per_atom_delta, batch.batch)
delta_forces      = -torch.autograd.grad(per_struct_delta.sum(), batch['positions'])[0]
e_wb97m_predicted = e_wb97x + per_struct_delta
```

### Why no caching of node_feats:
Forces = −d(delta_energy)/d(positions). This gradient flows through the delta head
and MACE backbone back to positions. If node_feats are cached as fixed numbers,
autograd has nothing to differentiate through — so MACE must run live every training step.
Backbone weights are frozen (no weight updates), but the forward pass still runs.

### Training decisions (all agreed):
- **Loss**: Huber on delta_energy + Huber on delta_forces (only for structures with force targets)
- **Optimizer**: Adam, lr=1e-3
- **Scheduler**: ReduceLROnPlateau on val loss
- **wandb**: train loss, val loss, lr per epoch
- **No mean-shifting** of targets (kept simple for POC; head learns the ~-3 eV mean offset via bias terms)
- **Freeze backbone completely** (no fine-tuning of last interaction)
- **Save**: only `delta_head.state_dict()` to `~/delta_head/delta_head.pt` — original model untouched

---

## Scripts to write next (not yet written)

1. **`pipeline/delta/prepare_delta_data.py`**
   - Reads all `results.json` + loads positions from T1x HDF5 or neb.db
   - Saves `~/delta_cache/train.pt` and `~/delta_cache/val.pt`
   - Each .pt = list of dicts: {rxn, geom_idx, positions, atomic_numbers, delta_eV, delta_forces}
   - delta_forces = None for Group A val entries

2. **`pipeline/delta/train_delta_head.py`**
   - Loads train.pt / val.pt
   - Loads frozen MACE via MACECalculator
   - Builds MACE graphs using `AtomicData.from_config` each batch
   - Trains NonLinearReadoutBlock with Huber loss + forces + wandb + scheduler
   - Saves delta_head.pt

3. **`pipeline/delta/job_train_delta_head.sh`**
   - GPU node (h200), runs prepare then train sequentially

---

## File locations — cluster

| Path | Description |
|------|-------------|
| `~/mace_t1x_p10_run-123.model` | TorchScript compiled model (CUDA required) |
| `~/mace_t1x_p10_run-123_epoch-362.pt` | State dict checkpoint |
| `~/data/Transition1x.h5` | T1x dataset |
| `~/ccsd_dataset/train_delta_rxns.txt` | 500 sampled train reaction IDs |
| `~/ccsd_dataset/val_converged.txt` | 174 val Group A reaction IDs |
| `~/ccsd_dataset/val_failed.txt` | 51 val Group B reaction IDs |
| `~/train_delta_sp/{rxn}/results.json` | Train SP results (being rerun with forces) |
| `~/val_delta_sp/{rxn}/results.json` | Val Group A SP results (being rerun with forces) |
| `~/val_delta_sp_flip/{rxn}/results.json` | Val Group B SP results (cancelled, needs rerun) |
| `~/pipeline/delta/` | Delta pipeline scripts |
| `~/.local/lib/python3.13/site-packages/mace/` | Installed MACE package (patched) |

## File locations — local (pipeline/delta/)

| File | Status |
|------|--------|
| `sample_train_reactions.py` | Done, run on cluster |
| `train_delta_sp.py` | Done, running on cluster (job 10360230) |
| `job_train_delta_sp.sh` | Done |
| `prepare_delta_data.py` | NOT YET WRITTEN |
| `train_delta_head.py` | NOT YET WRITTEN |
| `job_train_delta_head.sh` | NOT YET WRITTEN |
| `README.md` | Written, up to date |

---

## Other completed work this session

### Plausibility check on val delta SPs (energy-only, before forces rerun)
- Group A (174 rxns): delta mean −3.03 ± 0.29 eV, within-rxn std mean 0.089 eV
- Group B (51 rxns): delta mean −2.84 ± 0.15 eV, within-rxn std mean 0.093 eV
- High std at strained geometries = physics, not SCF failure (both energies move coherently)
- All results documented in README.md

### calculation_inventory.csv updates
- Added `nevpt2_avas_flag` column: `ok` / `red_flag` / `failed` for 10 MR benchmark reactions
- Added `val_delta_sp` and `val_delta_sp_flip` columns
- Updated 10 MR benchmark rows with ccsdt/nevpt2 results
- Fixed stale `orca_neb_converged` values for 10 val reactions

### README.md
- Full "Calculation Inventory" column documentation added
- Val Delta SP Plausibility Check section added with results table + TODO for deeper investigation
- Updated delta SP column descriptions to note forces and Group A limitation

### multireference_screening.md
- Full results section added: 7 reliable, 2 red_flag (rxn1320, rxn1150), 1 failed (rxn4113)
- nat_occ reliability criterion: ≥1 fractional occupation (0.05<n<1.95) at R, TS, AND P

### Prompt for architecture discussion (separate chat)
- Written to `delta_architecture_prompt.md` — full context for understanding the delta head
  architecture in a new Claude conversation without clogging this chat.
