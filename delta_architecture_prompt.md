# Delta Head Architecture — Context for New Chat

I am building a delta correction head on top of a frozen MACE model trained on the
Transition1x dataset. I want to understand the full architecture and how the pieces
connect before writing the training code.

---

## The MACE model

Trained on Transition1x wB97X-D3/6-31G(d) energies and forces.
Model file on cluster: `~/mace_t1x_p10_run-123.model` (TorchScript compiled, needs CUDA)
State dict checkpoint: `~/mace_t1x_p10_run-123_epoch-362.pt`

Architecture (inferred from state dict key shapes):
- Class: `ScaleShiftMACE` (from `mace.modules.models`)
- `num_interactions`: 2
- `hidden_irreps`: `1024x0e + 1024x1o + 1024x2e + 1024x3o` (dim = 16384)
- First interaction outputs only scalars: `1024x0e` (dim = 1024)
- `r_max`: 6.0 Å, `num_radial_basis`: 16
- Trained with `--num_channels=1024 --max_L=3 --num_interactions=2 --num_radial_basis=16`

### MACE forward pass (relevant excerpt from models.py)

```python
# Inside ScaleShiftMACE.forward():
node_feats_concat: List[torch.Tensor] = []

for i, (interaction, product, readout) in enumerate(zip(...)):
    node_feats = interaction(node_feats, ...)
    node_feats = product(node_feats, ...)
    node_feats_concat.append(node_feats)   # interaction 0: [N, 1024], interaction 1: [N, 16384]

# Output node features = concatenation of all interactions
node_feats_out = torch.cat(node_feats_concat, dim=-1)  # [N_atoms, 17408]

output["node_feats"] = node_feats_out   # this is what we hook into
output["energy"] = ...                  # wB97X-D3 energy prediction
```

### MACE readout blocks (from blocks.py)

```python
# readouts[0] — after interaction 0 — uses 1024-dim scalars only
class LinearReadoutBlock(torch.nn.Module):
    # linear: weight shape [1024]
    # input: node_feats_concat[0] = [N, 1024]

# readouts[1] — after interaction 1 — uses full 16384-dim irreps tensor
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in, MLP_irreps, gate, irrep_out='0e'):
        self.linear_1 = Linear(irreps_in, MLP_irreps)    # [N, 16384] → [N, 16]  (16x0e)
        self.non_linearity = Activation(MLP_irreps, [silu])
        self.linear_2 = Linear(MLP_irreps, irrep_out)    # [N, 16] → [N, 1]

    def forward(self, x):
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)   # [N_atoms, 1] — per-atom scalar energy
```

The model sums per-atom energies to get per-structure energy.

---

## The delta head

Goal: predict delta(geometry) = E_wB97M-V/def2-TZVP − E_wB97X-D3/6-31G(d)

The delta head is a **new** `NonLinearReadoutBlock` with the same architecture as
`readouts[1]` but with fresh randomly-initialized weights.

- Input: `node_feats_out[:, 1024:]` — the last 16384 dims of MACE's node_feats
  (these are from interaction 1, the full `1024x0e + 1024x1o + 1024x2e + 1024x3o`)
- Output: per-atom scalar, scattered (summed) to per-structure delta energy
- Forces: −d(delta_energy)/d(positions) via autograd

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
```

### At inference

```python
# Freeze backbone
for p in mace_model.parameters():
    p.requires_grad_(False)

# Forward pass
batch['positions'].requires_grad_(True)
out = mace_model(batch, training=False, compute_force=False)

node_feats   = out['node_feats'][:, 1024:]          # [N_atoms, 16384]
e_wb97x      = out['energy']                         # [N_structures] — MACE prediction

per_atom_delta    = delta_head(node_feats)            # [N_atoms, 1]
per_struct_delta  = scatter_sum(per_atom_delta, batch.batch)  # [N_structures, 1]

delta_forces = -torch.autograd.grad(
    per_struct_delta.sum(), batch['positions']
)[0]                                                  # [N_atoms, 3]

e_wb97m_predicted = e_wb97x + per_struct_delta        # final prediction
```

---

## File locations on cluster

| Path | Description |
|------|-------------|
| `~/.local/lib/python3.13/site-packages/mace/modules/models.py` | ScaleShiftMACE — forward pass |
| `~/.local/lib/python3.13/site-packages/mace/modules/blocks.py` | NonLinearReadoutBlock definition |
| `~/.local/lib/python3.13/site-packages/mace/cli/run_train.py` | Training loop (patched) |
| `~/mace_t1x_p10_run-123.model` | TorchScript compiled model (CUDA required) |
| `~/mace_t1x_p10_run-123_epoch-362.pt` | State dict checkpoint |
| `~/train_delta_sp/{rxn}/results.json` | 500 train reactions × 10 geoms, delta energy + forces |
| `~/val_delta_sp/{rxn}/results.json` | 174 val Group A × 50 geoms, delta energy only |
| `~/val_delta_sp_flip/{rxn}/results.json` | 51 val Group B × 50 geoms, delta energy + forces |
| `~/data/Transition1x.h5` | T1x dataset — positions, atomic_numbers, wB97X forces |
| `~/ccsd_dataset/train_delta_rxns.txt` | List of 500 sampled training reaction IDs |

## Local project files (on my PC)

| Path | Description |
|------|-------------|
| `pipeline/mace/run_train.py` | Patched MACE training script |
| `pipeline/mace/arg_parser.py` | Patched arg parser (adds --max_samples_per_epoch) |
| `pipeline/deploy_mace_patches.py` | Uploads patches to cluster via SFTP |
| `pipeline/delta/train_delta_sp.py` | Runs wB97M-V ORCA SPs for training data |
| `pipeline/delta/README.md` | Delta pipeline documentation |

---

## What I want to understand / build

I need to write three scripts (to go in `pipeline/delta/`):

1. **`prepare_delta_data.py`** — reads all `results.json` + loads positions from `Transition1x.h5`,
   saves `~/delta_cache/train.pt` and `~/delta_cache/val.pt` as lists of dicts with
   `{positions, atomic_numbers, delta_eV, delta_forces_eV_per_ang}`

2. **`train_delta_head.py`** — loads the cached data, runs the frozen MACE backbone
   each training step, attaches the delta head, trains with Huber loss on energy +
   forces, logs to wandb, saves best head weights to `~/delta_head/delta_head.pt`.
   Validation on val set (Group A: energy only; Group B: energy + forces).

3. **`job_train_delta_head.sh`** — SLURM GPU job that runs `prepare_delta_data.py`
   then `train_delta_head.py` sequentially.

Key constraints:
- MACE backbone weights are completely frozen (no gradient updates)
- Forces computed via autograd: requires `batch['positions'].requires_grad_(True)`
  before the MACE forward pass
- Only delta_head weights are optimized
- Loss = Huber(delta_energy) + w * Huber(delta_forces), w=1.0 initially
- Scheduler: ReduceLROnPlateau on val loss
- wandb logging: train loss, val loss, lr per epoch
- Save: only `delta_head.state_dict()`, not the full model
