# MACE+delta NEB Benchmark — Intermediate Results

> **Status: preliminary.** UMA-m NEB (job 10499578) and rxn1150 retry (job 10499577) are still running.
> Delta head training will continue — these results reflect v2 fw=1.0 only.
> This document will be rewritten once all comparisons are complete and a later head checkpoint is available.

---

## What this is

NEB run using the v2 MACE+delta head as the ASE calculator, benchmarked against the 30-reaction multireference benchmark.
The calculator wraps frozen MACE (wB97X-D3/6-31G*) + a delta correction head trained to predict δ = E(wB97M-V/def2-TZVP) − E(wB97X-D3/6-31G*), giving a wB97M-V-level PES at MACE inference speed.

The key question: does the delta-corrected PES find transition state geometries closer to the wB97M-V/ORCA reference than foundation models (eSEN, UMA)?

---

## Delta head (v2)

| Parameter | Value |
|-----------|-------|
| Backbone | `mace_t1x_p10_compiled.model` (frozen) |
| Head architecture | `NonLinearReadoutBlock`, MLP_irreps=64x0e |
| Training data | 80,592 geoms from 4,997 Transition1x train reactions |
| Force labels | All train geoms (wB97M-V EnGrad) + 2,240 val force geoms |
| Force weight | 1.0 (selected from sweep 0.5/1.0/2.0 by val force loss) |
| Epochs | 200, batch=64, lr=1e-3, 10k geoms/epoch subsampling |
| Val force loss | 0.0042 eV/Å (best checkpoint) |

See [delta_head.md](delta_head.md) for full training details.

---

## Benchmark setup

30 reactions from the Transition1x test set, split by multireference character (FOD rank):

- **High MR** (n=10): rxn7949, rxn8832, rxn1320, rxn4113, rxn8885, rxn7945, rxn7937, rxn6196, rxn0346, rxn1150
- **Mid MR** (n=10): rxn0896, rxn1154, rxn5690, rxn4513, rxn7955, rxn4519, rxn4500, rxn2553, rxn8829, rxn1155
- **Low MR** (n=10): rxn9246, rxn4498, rxn1061, rxn4003, rxn4004, rxn4063, rxn4114, rxn4060, rxn1961, rxn1962

**NEB protocol** (all methods):
- Starting band: 10 images from Transition1x.h5 wB97X-D3 NEB (R + last 8 interior + P)
- Endpoint relaxation: BFGS, fmax=0.05 eV/Å
- Plain NEB → CI-NEB (improvedtangent), fmax=0.05 eV/Å, max 500 steps each

**Reference**: ORCA wB97M-V/def2-TZVP NEB (same protocol, see `orca_neb.py`)

---

## Results — TS geometry (RMSD vs ORCA reference)

Kabsch RMSD on centroid-aligned transition state geometries (Å). Lower = closer to the wB97M-V reference saddle point.

```
rxn        MR     ORCA-eSEN   ORCA-UMA-s  ORCA-MACEd  fmax(MACEd)
rxn7949    High       0.198       0.162       0.098      0.134 !
rxn8832    High       0.166       0.145       0.107      0.075
rxn1320    High       0.366       0.362       0.046      0.056
rxn4113    High       0.017       0.014       0.020      0.079
rxn8885    High       1.411       0.492       0.028      0.058
rxn7945    High       0.427       0.044       0.325      0.052
rxn7937    High       0.049       0.043       0.178      0.112 !
rxn6196    High       0.093       0.085       0.222      0.135 !
rxn0346    High       0.134       0.141       0.027      0.051
rxn1150    High       0.007       0.009       0.284      2.214 !! (did not converge — retrying)

rxn9246    Low        0.006       0.004       0.042      0.063
rxn4498    Low        0.003       0.003       0.026      0.055
rxn1061    Low        0.021       0.016       0.057      0.163 !
rxn4003    Low        0.022       0.019       0.068      0.052
rxn4004    Low        0.016       0.009       0.064      0.075
rxn4063    Low        0.001       0.000       0.012      0.054
rxn4114    Low        0.001       0.001       0.013      0.055
rxn4060    Low        0.011       0.012       0.091      0.106 !
rxn1961    Low        0.001       0.000       0.022      0.038
rxn1962    Low        0.002       0.002       0.057      0.055

rxn0896    Mid        0.019       0.020       0.157      0.057
rxn1154    Mid        0.093       0.143       0.188      0.077
rxn5690    Mid        0.178       0.166       0.081      0.062
rxn4513    Mid        0.002       0.002       0.022      0.068
rxn7955    Mid        0.001       0.002       0.047      0.056
rxn4519    Mid        0.014       0.014       0.115      0.052
rxn4500    Mid        0.001       0.001       0.429      0.065
rxn2553    Mid        0.001       0.001       0.089      0.052
rxn8829    Mid        0.002       0.003       0.027      0.055
rxn1155    Mid        0.005       0.006       0.020      0.060
```

`!` = fmax > 0.10 eV/Å (CI-NEB not fully converged, TS geometry less reliable)

### Mean RMSD by MR class

| | High MR | Mid MR | Low MR |
|--|---------|--------|--------|
| eSEN | 0.287 | 0.032 | 0.008 |
| UMA-s | 0.150 | 0.036 | 0.007 |
| **MACEd-NEB** | **0.134** | 0.117 | 0.045 |
| UMA-m | *pending* | *pending* | *pending* |

**Key finding (preliminary):** MACEd-NEB finds TS geometries closest to the ORCA reference on High MR reactions, where the wB97M-V correction matters most. eSEN and UMA-s are significantly better on Mid and Low MR, where their base PES is already accurate. This motivates a routing approach: use MACEd for High MR, UMA for everything else.

Notable case: **rxn8885** — eSEN lands 1.41 Å from the ORCA saddle (wrong saddle point entirely), UMA-s 0.49 Å, MACEd 0.028 Å.

---

## Known issues

- **rxn1150** (High MR): CI-NEB stalled at fmax=2.2 eV/Å within 500 steps. Retrying with 1500 steps (job 10499577). May reflect a particularly rough delta-corrected PES for this reaction.
- **4 reactions with fmax > 0.10 Å**: rxn7949, rxn7937, rxn6196 (High MR) and rxn1061, rxn4060 (Low MR). Their TS geometries are the best found within the step limit but may not be true saddle points.
- **neb.db accumulation bug** (fixed): ASE's `ase.db.connect` appends to existing databases. The v2 run initially appended to v1 results, making barrier extraction wrong. Fixed in `mace_delta_neb.py` by deleting stale `neb.db` at startup. Collection script now hardcodes `n_images=10` for robustness.

---

## What's next

- [ ] UMA-m results (job 10499578) — add column to RMSD table
- [ ] rxn1150 converged result
- [ ] Continue delta head training (more data, longer training, sweep architectures)
- [ ] Full barrier comparison table once all methods complete
- [ ] Extend benchmark beyond 30 reactions

---

## Scripts

| Script | Purpose |
|--------|---------|
| `pipeline/mace_delta_neb.py` | MACE+delta ASE calculator + NEB pipeline |
| `pipeline/job_mace_delta_neb.sh` | SLURM array (30 reactions, h200 partition) |
| `pipeline/uma_neb.py` | UMA-s NEB (sm3090el8) |
| `pipeline/uma_m_neb.py` | UMA-m NEB (sm3090el8) |
| `pipeline/job_uma_m_neb.sh` | SLURM array for UMA-m |
| `pipeline/_collect_mace_delta_barriers.py` | Extract barriers from neb.db, update full_benchmark_results.json |
| `pipeline/_ts_rmsd_all_methods.py` | Kabsch RMSD comparison across all methods |
