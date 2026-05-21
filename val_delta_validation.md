# Val Delta Validation Data

## Goal

The delta model learns to correct wB97X-D3/6-31G(d) energies toward wB97M-V/def2-TZVP:

```
delta(geometry) = E_wB97M-V(geometry) - E_wB97X-D3(geometry)
```

To validate it, we need ground-truth deltas at geometries the model has not seen during training. All 225 val reactions in the Transition1x val split are used.

---

## Val Set Composition

| Status | Count | Source |
|--------|-------|--------|
| NEB converged | 174 | `orca_neb_val_results/{rxn}/neb.db` |
| NEB failed | 51 | `data/Transition1x.h5` (T1x geometries) |
| **Total** | **225** | |

### Convergence verification

Convergence is determined by the presence of a `converged` marker file at `orca_neb_val_results/{rxn}/converged`. This file is written by `orca_neb_rerun.py` only when the CI-NEB optimizer returns `converged=True`, i.e., fmax < 0.05 eV/Å. It is not written on partial convergence or failure.

The 174 converged reactions were confirmed by counting marker files directly:
```bash
find ~/orca_neb_val_results/ -name "converged" | wc -l  # → 174
```

All 225 reaction directories exist (`ls ~/orca_neb_val_results/ | wc -l → 225`), so the 51 without a marker genuinely failed — they did not simply fail to start.

The val NEB was run at **ωB97M-V/def2-TZVP** (same level as the test set), using ORCA 5.0.4 via ASE (`wB97M-V def2-TZVP def2/J RIJCOSX TightSCF`). The energies stored in `neb.db` are at this level of theory.

---

## Group A: 174 Converged NEB Reactions

**Script:** `pipeline/val_delta_sp.py`  
**Job:** `pipeline/job_val_delta_sp.sh` (SLURM array 0–173, job 10347722)  
**Output:** `~/val_delta_sp/{rxn}/results.json`

### What it computes

For each reaction:
1. Opens `orca_neb_val_results/{rxn}/neb.db`
2. Reads all rows (~2780 per reaction — the full NEB optimization history)
3. Selects 50 geometries uniformly spaced across the full history
4. Runs ORCA **wB97X-D3/6-31G(d)** SP on each geometry
5. Reads the **wB97M-V/def2-TZVP** energy already stored in `neb.db`
6. Saves delta = wB97M-V − wB97X-D3

### Why 50 images from the full history

The val `neb.db` stores every image from every NEB optimization iteration (~278 iterations × 10 images = ~2780 rows). The full history contains geometries at many stages of path optimization — from the initial linear interpolation through to the converged MEP. This gives diverse PES coverage.

**Why not just the last 10 images (final MEP):**
The delta model will be applied to arbitrary geometries during inference, not just final MEPs. Validating only on the final 10 images would test performance on a narrow slice of the PES and give an overly optimistic picture of generalization. The training data (T1x) contains geometries from all stages of NEB optimization, so validation should reflect the same distribution.

**Why not all ~2780 images:**
Running wB97X-D3 SPs on all 2780 images × 174 reactions = ~483,000 ORCA calculations is too expensive. 50 uniformly sampled images per reaction (8,700 total) gives sufficient statistical coverage at reasonable cost. Each reaction takes ~1–2 hours with 8 threads.

**Sampling method:** `np.linspace(0, n_total-1, 50)` — evenly spaced indices over the full history, rounded to integers. This ensures coverage from the earliest to the latest iteration.

### ORCA keywords

```
! wB97X-D3 6-31G(d) TightSCF
%pal nprocs 8 end
%maxcore 4000
```

Consistent with the wB97X-D3/6-31G(d) level used for the MR benchmark SPs (`mr_benchmark_setup.py`).

---

## Group B: 51 Failed NEB Reactions (Flip Approach)

**Script:** `pipeline/val_delta_sp_flip.py`  
**Job:** `pipeline/job_val_delta_sp_flip.sh` (SLURM array 0–50, job 10347920)  
**Output:** `~/val_delta_sp_flip/{rxn}/results.json`

### Why the flip approach

The 51 failed reactions have no `neb.db` — the NEB did not converge, so there are no wB97M-V geometries or energies. However, these reactions exist in the T1x dataset, which stores geometries and **wB97X-D3/6-31G(d)** energies from the original NEB run that generated Transition1x.

The flip approach uses T1x geometries as the common reference:
- **wB97X-D3 energy**: read directly from T1x (`wB97x_6-31G(d).energy`, in eV)
- **wB97M-V energy**: computed via ORCA SP on the same geometry

The delta is still E_wB97M-V − E_wB97X-D3 at the same geometry — identical quantity to Group A, just computed in the opposite direction.

### Why this is unbiased

Both approaches compute the delta at the same geometry using both levels of theory. Neither introduces a geometry selection bias:
- Group A: wB97M-V-optimized geometries with wB97X-D3 SPs on top
- Group B: wB97X-D3-optimized geometries with wB97M-V SPs on top

The delta model predicts a geometry-dependent correction. The geometry origin (which optimizer produced it) does not affect the validity of the delta label — as long as both energies are evaluated at the same structure.

**Selection bias concern:** Using only the 174 converged NEB reactions would bias the val set toward reactions that are "easy" for NEB (well-behaved PES, no problematic barriers). Including the 51 failed reactions via the flip approach removes this bias.

### What it computes

For each reaction:
1. Loads geometries and wB97X-D3 energies from `data/Transition1x.h5` (val split)
2. Selects 50 geometries uniformly from the full T1x history
3. Runs ORCA **wB97M-V/def2-TZVP** SP on each
4. Saves delta = wB97M-V_SP − wB97X-D3_T1x

### T1x HDF5 structure

```
f['val'][formula][rxn_id]['positions']             # (N_images, N_atoms, 3) Å
f['val'][formula][rxn_id]['atomic_numbers']        # (N_atoms,)
f['val'][formula][rxn_id]['wB97x_6-31G(d).energy'] # (N_images,) eV
```

### ORCA keywords

```
! wB97M-V def2-TZVP def2/J RIJCOSX TightSCF
%pal nprocs 8 end
%maxcore 4000
%scf maxiter 200 end
```

Identical to the keywords used in `orca_neb.py` for the val NEB run, ensuring consistency.

---

## Summary

| Group | Reactions | Geometries/rxn | Total SPs | SP level | Ref energy source |
|-------|-----------|----------------|-----------|----------|-------------------|
| A (converged NEB) | 174 | 50 from neb.db | 8,700 | wB97X-D3/6-31G(d) | neb.db (wB97M-V) |
| B (failed NEB) | 51 | 50 from T1x | 2,550 | wB97M-V/def2-TZVP | T1x (wB97X-D3) |
| **Total** | **225** | **50** | **11,250** | | |

---

## Scripts

| File | Purpose |
|------|---------|
| `pipeline/val_delta_sp.py` | Group A: wB97X-D3 SPs on NEB geometries |
| `pipeline/job_val_delta_sp.sh` | SLURM array for Group A (174 reactions) |
| `pipeline/val_delta_sp_flip.py` | Group B: wB97M-V SPs on T1x geometries |
| `pipeline/job_val_delta_sp_flip.sh` | SLURM array for Group B (51 reactions) |

Reaction lists:
- `ccsd_dataset/val_converged.txt` — 174 converged reactions (ordered as in val_reactions.txt)
- `ccsd_dataset/val_failed.txt` — 51 failed reactions
