# Transition1x relabelling project

MSc thesis (DTU Energy, Atomic Scale Materials Modelling, supervisor Arghya Bhowmik), submitted 2026-09-21, graded 12.
Two questions: RQ1, lift a MACE model trained on Transition1x to wB97M-V/def2-TZVP with a delta head trained on <1 % relabelled data;
RQ2, test the OMol25 models (UMA-S, UMA-M, eSEN) at broken-symmetry transition states. A paper from RQ2 is in preparation.

## Where things are

- `docs/overleaf_v2/`: the submitted thesis text (LaTeX), `REVIEW_LOG.md` with every number check. `docs/overleaf/` is the older mirror.
- `results/`: the small CSV/JSON summaries every thesis number comes from. `pipeline/check_results_section.py` asserts the
  thesis numbers against them and must stay at 0 Beanstandungen.
- `pipeline/`: all scripts. `pipeline/delta/`: delta-head training and evaluation. `pipeline/plot_*.py`: thesis figures.
- `Pictures/`, `figures/`: thesis figures. `docs/defence/`: defence slides and their drawing scripts.
- `data/Transition1x.h5`: the dataset, not in git (6.5 GB). Fields `wB97x_6-31G(d).energy/.forces`.

## Facts that are easy to get wrong

- Transition1x is wB97X/6-31G(d) WITHOUT D3 (Chai & Head-Gordon 2008). ORCA's `wB97X-D3` is a different functional.
  The final Chapter 3 single points are the no-D3 rerun: `results/delta_fixed_head/eval_benchmark_sp_fixed_full_nod3.json`.
- Thesis vocabulary: closed-shell / broken-symmetry (groups), stable / unstable (restricted solution), restricted and
  unrestricted surface, "level of theory" = functional/basis only, "OMol25 protocol" = level + unrestricted + route to the
  BS solution. Never "ground-state surface". British spelling.
- Benchmark reactions are from the Transition1x TEST split; any fine-tuning uses the TRAIN split only.

## Cluster (Niflheim, DTU)

- User `bepab`, home `/home/energy/bepab`, login node `thul.fysik.dtu.dk` (`slid` rejects sessions). Old student account
  `s242862` is closed; its whole home was copied here, so old `/home/energy/s242862/...` paths exist under the new home with
  the name changed. Many scripts still hard-code the old path.
- Login nodes are not reachable from outside the DTU network (SSH times out); from elsewhere use the DTU VPN first.
- Do not open many SSH sessions in a row: a dozen rapid connections trip a port-22 ban on all login nodes for hours.
- Login nodes enforce a hard limit of 20 min CPU time per process. Anything heavier than a few seconds goes into a SLURM job.
- SLURM: max 20 concurrent jobs. Jobs must request whole nodes: `--ntasks 24` on xeon24el8/xeon24el9, 40 on xeon40el8/xeon40el9;
  do not mix partitions with different core counts. GPU partitions: `h200el9` (used for the delta-head evaluation), `a100`.
  `h200` without suffix is invalid. `#SBATCH --output=~/logs/...` writes to a literal `~/logs` directory; use `$HOME/logs`.
- Environment: `source /etc/profile; module load Python/3.13.5-GCCcore-14.3.0` (fairchem-core 2.20.0, ase 3.28, torch 2.8).
  ORCA: `ORCA/5.0.4-gompi-2023a`. Node: `module load nodejs/22.17.1-GCCcore-14.3.0`.
- Old working folders next to the repo clone: `~/pipeline` (stale copies of the repo scripts), `~/mr_benchmark`
  (ORCA single points; `orca_engrad_nod3` is the final run), `~/delta_head`, `~/data/transition1x_train.db`,
  `~/autocas_*`, `~/band_*` (multireference and NEB experiments). The repo clone is `~/Transition1x`.
- Claude Code on the cluster: `~/.npm-global/bin/claude`, started under Remote Control in a tmux session named `claude`.

## Conventions

- Every number stated in the thesis or a paper must be asserted by the checker script against a results file.
- Commit with the attribution line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and push to
  `origin rks-stability-bs-omol25`. Pull before starting work on another machine.
- Files are CRLF on the Windows copy; the cluster clone uses LF. Do not reformat line endings in commits.
