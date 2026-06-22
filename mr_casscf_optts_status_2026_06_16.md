# MR CASSCF OptTS — Status as of 2026-06-16

> **TEMPORARY DOCUMENTATION.** This file exists so we don't lose track of what
> changed in this session. It should be folded into a proper writeup (or
> deleted) once the CASSCF convergence situation is settled.

## What this pipeline does

`pipeline/mr_casscf_optts.py` reoptimizes the transition state for the 10
High-MR benchmark reactions at CASSCF(AVAS)/NEVPT2, starting from the ORCA
wB97M-V NEB TS:

1. RHF + AVAS at the ORCA NEB TS → auto-selects (nelecas, ncas)
2. CASSCF at the TS with AVAS MOs
3. `geometric` eigenvector-following OptTS
4. CASSCF+NEVPT2 at the optimized TS (MOs projected from step 2)
5. CASSCF+NEVPT2 at ORCA's R and P, MOs projected from step 4 (same active
   space and orbital frame throughout → barriers are comparable)

Output: `~/nevpt2_optts_results/{rxn}_avas/nevpt2_optts_results.json` +
`ts_casscf_opt.xyz`.

## History so far

1. First run (threshold=0.2, default): 4/10 converged (rxn7949, rxn4113,
   rxn6196, rxn1150 — actually rxn1150 stalled, not a clean convergence;
   effectively rxn7949/rxn4113/rxn6196 succeeded cleanly). 6/10 failed —
   mostly CASSCF not converging because AVAS selected very large active
   spaces (up to 16e/12o).
2. Retry (threshold=0.4, job 10506706): raised the AVAS threshold to shrink
   active spaces. Only 3/10 converged this time (rxn7949, rxn6196, rxn0346).
   rxn4113 *failed* in this retry (gradient non-convergence during OptTS) —
   but because the script only writes output files after each step
   succeeds, the failed retry never overwrote rxn4113's original
   threshold=0.2 result, which was still sitting on disk and still valid.
   So we actually had **4 usable results**: rxn7949, rxn6196, rxn0346,
   rxn4113 — recovered by re-discovering the untouched directory.
3. Diagnosis: failure logs showed active orbitals with natural occupations
   like 0.22, 0.09, 0.05, 0.03 — i.e. AVAS was including marginal/intruder
   orbitals (weak projection onto the target AOs, but not actually part of
   the correlation problem) that cause CASSCF macro-iteration oscillation
   instead of contributing real multireference character.

## What changed today

Added two fixes to `mr_casscf_optts.py`:

- **`prune_active_space()`**: right after AVAS selects (ncas, nelecas, mo),
  runs a cheap CASCI (diagonalization only, no orbital optimization) to get
  natural orbital occupations in the active space. Any active orbital with
  occupation > 1.98 (move to core, nelecas -= 2) or < 0.02 (move to virtual)
  is dropped before the expensive CASSCF orbital optimization starts.
  Disable with `--no-prune` if needed for debugging.
- **`run_casscf()` solver fallback**: if PySCF's default `mc1step`
  (augmented-Hessian, one-step) solver doesn't converge, automatically
  retries with `mc.mc2step()` (two-step solver — slower but more robust to
  exactly this kind of near-degenerate oscillation) before giving up.

## Action taken

- Backed up the 4 existing successful result directories (copied, not
  moved) to `~/nevpt2_optts_results/{rxn}_avas_backup_preprune/` for
  rxn7949, rxn6196, rxn0346, rxn4113 — so we don't lose them if the rerun
  changes or breaks something.
- Resubmitted **all 10** High-MR reactions (not just the 6 that were
  failing) with the new pipeline, for methodological consistency — same
  reasoning as the earlier "rerun everyone with the same parameters"
  decision. Job **10515621**, array 0-9, `xeon24el8`, same REACTIONS order
  as before: `rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn7945 rxn7937
  rxn6196 rxn0346 rxn1150`.

## Bug found in the pruning fix, and the follow-up fix

Job 10515621's pruning logic was too aggressive: it dropped any active
orbital with occupation outside a fixed [0.02, 1.98] window, with no floor
on how many orbitals could be removed.

- **Task 3 (rxn4113)**: AVAS at threshold=0.4 selected a genuinely
  degenerate space — (16e, 8o), i.e. nelecas = 2×ncas exactly, every
  orbital occupation = 2.0. Pruning correctly identified all 8 as
  "near-doubly-occupied" and dropped them, collapsing to (0e, 0o), which
  crashed PySCF's sanity check. This one really is a bad AVAS selection at
  this threshold — not a pruning bug per se.
- **Task 8 (rxn0346)**: this reaction converged fine in the *original*
  (unpruned) pipeline. AVAS selected (14e, 9o) with occupations
  `[1.992, 1.997, 1.997, 1.995, 1.995, 1.997, 1.987, 0.022, 0.018]` — every
  orbital sits just outside the [0.02, 1.98] window, but this is weak,
  *distributed* multireference character, not intruder orbitals. Pruning
  stripped the entire space to (0e, 0o) and crashed — a real regression,
  since this reaction wasn't broken before.

**Fix**: `prune_active_space()` now takes a `min_ncas=2` floor. If pruning
would leave fewer than `min_ncas` orbitals, it's skipped entirely and the
original unpruned AVAS space is used instead (logged via `prune_info` in
the output JSON, see below). This recovers rxn0346's previously-working
behavior while still allowing pruning to fire for reactions where only
*some* orbitals are marginal.

**Also added**: every `nevpt2_optts_results.json` now records its own
provenance — `avas_threshold`, `avas_ncas`/`avas_nelecas` (pre-pruning),
`prune_enabled`, `prune_info` (whether pruning fired, what was dropped, or
why it was skipped), final `ncas`/`nelecas`, and `solver_used_ts` /
`solver_used_ts_opt` (`mc1step` or `mc2step`). This was added specifically
so we don't have to rely on memory of which run/settings produced which
reaction's result — check the JSON itself.

Resubmitted just the 2 failed tasks (not the whole array) as job
**10518016**, `--array=3,8` → rxn4113, rxn0346.

## Update 2026-06-22: the CAS(2,2) problem and the no-prune rerun

Job 10515621's pruning (even with the `min_ncas=2` floor) turned out to be
over-aggressive for several reactions: it pruned down to a bare CAS(2,2) —
scientifically too small to call a multireference gold standard — for 6
reactions (rxn7949, rxn8832, rxn1320, rxn8885, rxn7945, rxn7937) and to
CAS(4,4) for 2 more (rxn6196, rxn1150).

**Action**: backed up the CAS(2,2) results to `*_avas_backup_cas22` and
resubmitted those 6 with `--no-prune` (full AVAS-selected active space,
mc1step→mc2step fallback only) as job **10525513**, array 0-5.

**Result**: 5/6 converged with proper active spaces; 1 failed.

| task | rxn | outcome | elapsed |
|---|---|---|---|
| 0 | rxn7949 | COMPLETED → (16e,10o) | 4h27m |
| 1 | rxn8832 | COMPLETED → (16e,10o) | 1d12h09m |
| 2 | rxn1320 | **FAILED** | 10h02m |
| 3 | rxn8885 | COMPLETED → (12e,9o), mc2step | 21h52m |
| 4 | rxn7945 | COMPLETED → (14e,10o) | 9h53m |
| 5 | rxn7937 | COMPLETED → (14e,10o) | 10h04m |

**rxn1320's failure mode is different from the others**: CASSCF converges
fine for the initial TS, but the `geometric` eigenvector-following OptTS
crashes mid-optimization with `RuntimeError: Nuclear gradients ... not
converged` — the CASSCF *scanner* (used to get energy+gradient at each new
geometry along the optimization path) fails to converge at some
intermediate geometry, not at the start. mc2step fallback is implemented in
`run_casscf()` for the initial calculation but isn't wired into the scanner
used during OptTS, so it can't help here. Still only has the old CAS(2,2)
result; excluded from all MR-Optimized comparisons.

Then resubmitted rxn6196 and rxn1150 (the CAS(4,4) cases) with `--no-prune`
too, for consistency — backed up their old results to `*_avas_backup_cas44`
and submitted as job **10536404**, array 0-1. Still running as of
2026-06-22.

## Status tracker — which settings produced each reaction's current result

| rxn | Current result from | active space | avas_threshold | prune | solver | Notes |
|---|---|---|---|---|---|---|
| rxn7949 | job 10525513 (task 0, no-prune) | (16e,10o) | 0.4 | off | mc1step | valid |
| rxn8832 | job 10525513 (task 1, no-prune) | (16e,10o) | 0.4 | off | mc1step | valid |
| rxn1320 | job 10515621 (task 2) — **still CAS(2,2)** | (2e,2o) | ? | on (old) | ? | job 10525513 task 2 **FAILED** (CASSCF diverges mid-OptTS, see above). Excluded from MR-Optimized comparisons. |
| rxn4113 | original run (pre-2026-06-17), provenance patched manually | (16e,10o) | 0.2 | off | mc1step | threshold=0.4 yields degenerate (16e,8o) AVAS space → 0.2 required |
| rxn8885 | job 10525513 (task 3, no-prune) | (12e,9o) | 0.4 | off | mc2step | valid |
| rxn7945 | job 10525513 (task 4, no-prune) | (14e,10o) | 0.4 | off | mc1step | valid |
| rxn7937 | job 10525513 (task 5, no-prune) | (14e,10o) | 0.4 | off | mc1step | valid |
| rxn6196 | job 10515621 (task 7, pruned) — **resubmission pending** | (4e,4o) | ? | on (old) | ? | resubmitted no-prune as job **10536404** task 0; old result backed up to `*_avas_backup_cas44` |
| rxn0346 | job 10518016 (task 8, fixed pruning) | (14e,9o) | 0.4 | skipped (min_ncas floor) | mc1step | valid |
| rxn1150 | job 10515621 (task 9, pruned) — **resubmission pending** | (4e,4o) | ? | on (old) | ? | resubmitted no-prune as job **10536404** task 1; old result backed up to `*_avas_backup_cas44`; suspicious result — all 4 MLIP methods give near-identical RMSD (0.144-0.145 Å) vs this reference, suggesting the (4e,4o) "optimized" TS barely moved from the ORCA NEB starting point |

**7/10 reactions now have valid, scientifically meaningful active spaces**
(rxn7949, rxn8832, rxn4113, rxn8885, rxn7945, rxn7937, rxn0346). These are
the set used in the current MR-Optimized RMSD comparison
(`mr_optimized_rmsd_all10.png`):

| Method | Mean RMSD vs CASSCF+NEVPT2 OptTS (Å), n=7 |
|---|---|
| ORCA wB97M-V | 0.117 |
| UMA-S | 0.144 |
| UMA-M | 0.192 |
| MACE+delta fw2 | 0.211 |
| eSEN | 0.326 |

**Don't trust this table blindly** — always cross-check against the
`avas_threshold`/`prune_info`/`solver_used_*` fields actually written in
each reaction's `nevpt2_optts_results.json`, since that's the authoritative
record, not this manually-maintained table.

## What to check when job 10536404 finishes

- If rxn6196/rxn1150 converge to proper active spaces (not CAS(2,2)/(0,0)),
  add them to the valid set and refresh `mr_optimized_rmsd_all10.png`
  (n=7 → n=9).
- rxn1320 remains the one true holdout. If it's worth fixing: try a looser
  CASSCF `conv_tol`/`conv_tol_grad` during the OptTS scan, a smaller
  `max_stepsize`, or capping `maxsteps` and accepting a partially-converged
  saddle point rather than none at all.

## Related files

| File | Purpose |
|------|---------|
| `pipeline/mr_casscf_optts.py` | Main script (pruning + mc2step fallback, `--no-prune` flag) |
| `pipeline/job_casscf_optts_mr_retry.sh` | SLURM array, all 10 reactions, threshold=0.4, pruning on (job 10515621) |
| `pipeline/job_casscf_optts_no_prune.sh` | SLURM array, 6 CAS(2,2) reactions, `--no-prune` (job 10525513) |
| `pipeline/job_casscf_optts_noprune2.sh` | SLURM array, rxn6196 + rxn1150, `--no-prune` (job 10536404) |
| `pipeline/_backup_and_resubmit_mr.py` | One-off: backed up old results, resubmitted job 10515621 |
| `pipeline/_backup_and_resubmit_noprune.py` | One-off: backed up CAS(2,2) results, resubmitted job 10525513 |
| `pipeline/_collect_mr_results.py` / `_collect_mr_results_remote.py` | RMSD vs MR-Optimized TS for all valid reactions, generates `mr_optimized_rmsd_all10.png` |
| `benchmark_plots/mr_optimized_rmsd_all10.png` | Current plot, n=7 valid reactions |
