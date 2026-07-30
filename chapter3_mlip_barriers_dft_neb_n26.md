# Chapter 3 Notes — MLIP Barriers on Fixed DFT NEB Geometries (n=26)

## What this is

For each reaction in the MR benchmark, four MLIPs (UMA-S, UMA-M, bare MACE,
MACE+delta) are normally evaluated by running their **own independent NEB**
on the wB97x starting band — each lands on its own optimized TS, which may
sit at a different geometry than the DFT (ORCA wB97M-V/def2-TZVP) TS. That
conflates two different questions: "does this MLIP's PES have the right
barrier height" and "can this MLIP's optimizer find the right saddle point."

This note documents a separate, complementary calculation: **freeze the
geometry** at the ORCA wB97M-V/def2-TZVP NEB's own converged
reactant/TS/product structures, and evaluate every method as a single point
on those exact geometries. This isolates level-of-theory error from
geometry error — same rationale as the existing CCSD(T)/NEVPT2
single-point-on-fixed-geometry benchmark (`multireference_screening.md`
Step 2), just extended to the MLIPs and to more reactions.

## Reaction set (n=26)

23 reactions from the existing MR benchmark (10 High(orig) + 13 next-HIGH,
see `active_space_quality_analysis.md` / `mace_delta_neb_benchmark.md` Part
3), plus 3 reactions previously marked "CASSCF OptTS did not converge" and
excluded (rxn5691, rxn1283, rxn0894). On inspection, all three actually have
complete, physically sane `nevpt2_optts_results.json` output on the cluster
(clean natural-orbital occupations, no intruder-like values) — the "failed"
status in `mace_delta_neb_benchmark.md` (`26 attempted; 23 converged`) and
`mr_casscf_optts_status_2026_06_16.md` appears stale; that file explicitly
warns it is temporary and to trust the JSON provenance over the manually
maintained tables. They were folded back in here rather than left out.

Full list:
```
rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn7945 rxn7937 rxn6196 rxn0346 rxn1150
rxn0896 rxn7060 rxn8827 rxn1147 rxn10005
rxn4518 rxn3107 rxn4522 rxn7936 rxn0101 rxn10054 rxn7957 rxn8837
rxn5691 rxn1283 rxn0894
```

## How each column was computed

**DFT** — the ORCA wB97M-V/def2-TZVP NEB's *own* forward barrier, i.e. no
recomputation: for the 23 original reactions this is `orca_fwd_meV` from
`barrier_comparison_optts.json` (already derived from each reaction's
converged NEB path). For the 3 recovered reactions it was extracted
directly from `orca_neb_results/<rxn>/neb.db`: read all rows via `ase.db`,
take the last 10 (the final converged image set), forward barrier =
`E[argmax] - E[0]`.

**CCSD(T)** — RHF → RCCSD → CCSD(T)/def2-TZVP via PySCF, single points on
`orca_neb_results/<rxn>/{reactant,transition_state,product}.xyz` — the
*unmodified* DFT NEB geometries, not the separately-optimized CASSCF OptTS
geometry used elsewhere in the benchmark. Script: `pipeline/mr_benchmark_ccsdt.py`
(no `--geom-dir` override → defaults to `orca_neb_results/<rxn>`). Run as
SLURM CPU array jobs on `xeon24el8` (24 cores, 120GB, one reaction per
task): `pipeline/job_mr_ccsdt_nexthigh.sh` (the 12 next-HIGH reactions that
didn't already have this) and `pipeline/job_mr_ccsdt_extra3.sh` (the 3
recovered reactions). The original 10 High(orig) + rxn0896 already had
CCSD(T) at this geometry from earlier benchmark work.

**UMA-S / UMA-M / MACE bare / MACE+delta** — single-point energy at the
same three fixed DFT geometries, using each method's normal ASE calculator
(`fairchem` FAIRChemCalculator for UMA-S/UMA-M, the custom
`MACEDeltaCalculator` from `pipeline/mace_delta_neb.py` for bare/delta MACE,
delta head `delta_head_fw2.00.pt`). New scripts, since no fixed-geometry
single-point path existed for the MLIPs before this:
- `pipeline/mr_benchmark_mlip_sp.py` — single-reaction CLI, mirrors the
  CCSD(T) script's interface (`<rxn> --method {uma_s,uma_m,esen,mace_bare,mace_delta}`)
- `pipeline/mr_benchmark_mlip_sp_batch.py` — loads the model once and loops
  over many reactions, so a GPU job doesn't pay model-load cost per
  reaction
- `pipeline/job_mr_mlip_sp_dftneb.sh` / `job_mr_mlip_sp_dftneb_extra3.sh` —
  SLURM GPU array jobs on `h200`, one task per method

Each single point writes `{rxn}_{method}_sp_dftneb.json` (barrier_fwd_meV,
barrier_rev_meV, and per-geometry energies) to
`~/mr_benchmark/results/` on the cluster, plus a `summary_{method}_sp_dftneb.json`
collecting all reactions for that method.

## Results — forward barriers (meV)

| rxn | DFT | CCSD(T) | UMA-s | UMA-m | MACE bare | MACE+Δ |
|---|---|---|---|---|---|---|
| rxn7949 | 3956 | 3210 | 3396 | 3398 | 3925 | 3728 |
| rxn8832 | 3207 | 2621 | 2768 | 2772 | 2968 | 2845 |
| rxn1320 | 3407 | 3051 | 3072 | 3076 | 3393 | 3190 |
| rxn4113 | 5438 | 5346 | 5406 | 5408 | 5532 | 5228 |
| rxn8885 | 3607 | 3564 | 3540 | 3547 | 3660 | 3850 |
| rxn7945 | 3900 | 3923 | 3879 | 3888 | 3885 | 3777 |
| rxn7937 | 3829 | 3858 | 3820 | 3818 | 3886 | 3852 |
| rxn6196 | 4255 | 4282 | 4246 | 4238 | 4377 | 4236 |
| rxn0346 | 3550 | 3336 | 3400 | 3394 | 3663 | 3467 |
| rxn1150 | 3601 | 3460 | 3607 | 3604 | 3674 | 3602 |
| rxn0896 | 5222 | 5094 | 5208 | 5206 | 4708 | 4728 |
| rxn7060 | 6164 | 5957 | 6138 | 6132 | 5338 | 5221 |
| rxn8827 | 3862 | 3561 | 3822 | 3816 | 3805 | 3440 |
| rxn1147 | 4194 | 3980 | 4074 | 4068 | 4037 | 3819 |
| rxn10005 | 3722 | 3552 | 3719 | 3723 | 3782 | 3547 |
| rxn4518 | 5441 | 4643 | 4796 | 4795 | 5256 | 5090 |
| rxn3107 | 4140 | 4083 | 4072 | 4071 | 4292 | 4218 |
| rxn4522 | 5695 | 5331 | 5519 | 5518 | 5445 | 4977 |
| rxn7936 | 5812 | 5711 | 5793 | 5795 | 5396 | 5692 |
| rxn0101 | 2772 | 2765 | 2736 | 2739 | 3114 | 2856 |
| rxn10054 | 1841 | 1704 | 1837 | 1837 | 2034 | 1800 |
| rxn7957 | 3953 | 3728 | 3852 | 3842 | 3858 | 3692 |
| rxn8837 | 4225 | 3777 | 3914 | 3920 | 3975 | 3685 |
| rxn5691 | 3126 | 2782 | 2974 | 2962 | 3339 | 3088 |
| rxn1283 | 5363 | 5143 | 5301 | 5300 | 5526 | 5417 |
| rxn0894 | 4934 | 4442 | 4736 | 4739 | 4997 | 4755 |

## MAE (meV), n=26

| Method | vs DFT | vs CCSD(T) |
|---|---|---|
| UMA-s | 138.6 | 118.7 |
| UMA-m | 139.2 | 117.3 |
| MACE bare | 182.6 | **295.2** |
| MACE+Δ | 245.5 | 204.1 |

**Reading:** UMA-S/UMA-M are closest to both references when evaluated on
the true DFT saddle-point geometry. MACE bare tracks the DFT barrier
reasonably (182.6 MAE) but is furthest from CCSD(T) (295.2) — consistent
with it being trained on wB97X-D3 labels. MACE+delta moves *toward*
CCSD(T) relative to bare MACE (295.2 → 204.1) but *away* from DFT (182.6 →
245.5), which makes sense: the delta head is trained to approximate
wB97M-V, a level closer to (but not identical to) CCSD(T)/NEVPT2 than plain
wB97X-D3 is.

On the 18-reaction subset that excludes the weakest/most divergent
reactions (rxn0346, rxn0896, rxn1147, rxn4518, rxn4522, rxn5691, rxn1283,
rxn0894), the same ranking holds with tighter errors across the board (UMA-s
115.9/101.6, UMA-m 115.4/101.1, MACE bare 171.7/260.2, MACE+Δ 227.2/164.1
meV vs DFT/CCSD(T) respectively).

## Caveats

- rxn8837's UMA-S/UMA-M *own-NEB* runs fragment (see
  `mace_delta_neb_benchmark.md` Part 3) — irrelevant here since this
  analysis uses the fixed DFT geometry, not each method's own NEB path, so
  rxn8837 has valid entries in this table despite that.
- rxn5691/rxn1283/rxn0894 are not yet reflected in the reliability
  classification tables elsewhere in the repo (`active_space_quality_analysis.md`,
  `mace_delta_neb_benchmark.md` Part 3) — those still list them as failed/excluded.
  If they're kept in the benchmark going forward, those docs should be
  updated to avoid the same "23 converged" claim being repeated.
- Raw per-reaction/per-method JSON files live in
  `~/mr_benchmark/results/` on the cluster
  (`{rxn}_ccsdt.json`, `{rxn}_{method}_sp_dftneb.json`,
  `summary_{method}_sp_dftneb.json`); not yet copied into this repo.
