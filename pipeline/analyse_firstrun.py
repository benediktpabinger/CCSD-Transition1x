"""
Analyse first-run NEB trajectories to determine whether reactions that
converged in the rerun were actually still making progress in the first run
(i.e. would have converged with more steps) or were genuinely stuck.

For each converged reaction:
  1. Parse neb.log timestamps to find the time gap that marks the rerun start
  2. Split fmaxs.json into first-run and rerun portions at that index
  3. Classify the first-run ending trajectory:
       still_decreasing  – fmax was still falling when the run hit the step limit
       plateau           – fmax levelled off (oscillation < 0.01 eV/Å)
       oscillating       – fmax jumping around with no trend
       crash             – no steps recorded
  4. Report: how many reactions just needed more steps vs needed the restart

Usage:
    python analyse_firstrun.py --results ~/orca_neb_results --split test
    python analyse_firstrun.py --results ~/orca_neb_val_results --split val
    python analyse_firstrun.py --results ~/orca_neb_results ~/orca_neb_val_results
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_neb_log_runs(neb_log):
    """
    Parse neb.log and return a list of runs, where each run is a list of fmax values.
    Each run starts with a header line containing 'Step     Time'.
    The log may contain multiple runs appended together (original + reruns).
    """
    runs = []
    current = []
    with open(neb_log) as f:
        for line in f:
            if 'Step' in line and 'Time' in line and 'fmax' in line:
                # New run header — save previous run if non-empty
                if current:
                    runs.append(current)
                current = []
            elif 'NEBOptimizer' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        fmax = float(parts[-1])
                        current.append(fmax)
                    except ValueError:
                        pass
    if current:
        runs.append(current)
    return runs


def classify_trajectory(fmaxs, window=10):
    """Classify the tail of a fmax trajectory."""
    if not fmaxs:
        return 'crash'
    if len(fmaxs) < 3:
        return 'too_few_steps'
    tail = fmaxs[-window:] if len(fmaxs) >= window else fmaxs
    slope = (tail[-1] - tail[0]) / len(tail)
    osc   = max(tail) - min(tail)
    if slope < -0.003 and osc < 0.05:
        return 'still_decreasing'
    elif osc < 0.015:
        return 'plateau'
    else:
        return 'oscillating'


# ── main ─────────────────────────────────────────────────────────────────────

def analyse(results_dirs):
    records = []

    for results_dir in results_dirs:
        results_dir = os.path.expanduser(results_dir)
        if not os.path.isdir(results_dir):
            print(f"WARNING: not found: {results_dir}")
            continue

        for rxn in sorted(os.listdir(results_dir)):
            rxn_dir = os.path.join(results_dir, rxn)
            if not os.path.isdir(rxn_dir):
                continue

            converged    = os.path.exists(os.path.join(rxn_dir, 'converged'))
            neb_log      = os.path.join(rxn_dir, 'neb.log')
            fmaxs_path   = os.path.join(rxn_dir, 'fmaxs.json')

            if not os.path.exists(fmaxs_path) and not os.path.exists(neb_log):
                continue

            # Parse neb.log into runs regardless of fmaxs.json
            runs = []
            if os.path.exists(neb_log):
                runs = parse_neb_log_runs(neb_log)

            if os.path.exists(fmaxs_path):
                fmaxs = json.load(open(fmaxs_path))
            else:
                # Reconstruct from neb.log directly
                fmaxs = [f for run in runs for f in run]

            if not fmaxs and not runs:
                records.append(dict(rxn=rxn, converged=converged,
                                    first_steps=0, rerun_steps=0,
                                    first_final_fmax=None, traj_class='crash',
                                    had_rerun=False))
                continue

            # Split into runs using neb.log header sections
            had_rerun = False
            first_fmaxs = fmaxs
            rerun_fmaxs = []

            if len(runs) >= 2:
                had_rerun = True
                first_fmaxs = runs[0]
                rerun_fmaxs = [f for run in runs[1:] for f in run]
            elif len(runs) == 1:
                first_fmaxs = runs[0]

            traj_class     = classify_trajectory(first_fmaxs)
            first_final    = first_fmaxs[-1] if first_fmaxs else None

            records.append(dict(
                rxn            = rxn,
                converged      = converged,
                first_steps    = len(first_fmaxs),
                rerun_steps    = len(rerun_fmaxs),
                first_final_fmax = first_final,
                traj_class     = traj_class,
                had_rerun      = had_rerun,
            ))

    return records


def report(records):
    total      = len(records)
    converged  = [r for r in records if r['converged']]
    failed     = [r for r in records if not r['converged']]
    had_rerun  = [r for r in records if r['had_rerun']]
    no_rerun   = [r for r in records if not r['had_rerun']]

    print(f"Total reactions analysed: {total}")
    print(f"  Converged:   {len(converged)}")
    print(f"  Failed:      {len(failed)}")
    print(f"  Had rerun:   {len(had_rerun)}")
    print(f"  No rerun:    {len(no_rerun)}")
    print()

    # Converged reactions: did they need the rerun?
    conv_no_rerun  = [r for r in converged if not r['had_rerun']]
    conv_had_rerun = [r for r in converged if r['had_rerun']]
    print(f"=== CONVERGED REACTIONS ({len(converged)}) ===")
    print(f"  Converged in first run (no rerun needed): {len(conv_no_rerun)}")
    print(f"  Converged only after rerun:               {len(conv_had_rerun)}")
    print()

    # For reactions that needed a rerun and converged: what was first-run doing?
    print(f"=== FIRST-RUN TRAJECTORY for reactions that needed rerun ({len(conv_had_rerun)}) ===")
    traj_counts = defaultdict(list)
    for r in conv_had_rerun:
        traj_counts[r['traj_class']].append(r)

    for cls, rxns in sorted(traj_counts.items()):
        fmaxs_end = [r['first_final_fmax'] for r in rxns if r['first_final_fmax'] is not None]
        steps     = [r['first_steps'] for r in rxns]
        print(f"  {cls}: {len(rxns)} reactions")
        if fmaxs_end:
            print(f"    first-run final fmax: {min(fmaxs_end):.3f} – {max(fmaxs_end):.3f} eV/Å  (mean {np.mean(fmaxs_end):.3f})")
        if steps:
            print(f"    first-run steps:      {min(steps)} – {max(steps)}  (mean {np.mean(steps):.0f})")
        # List the still_decreasing ones — these are the ones that just needed more steps
        if cls == 'still_decreasing':
            print(f"    *** These likely would have converged with more steps ***")
            for r in sorted(rxns, key=lambda x: x['first_final_fmax'] or 9):
                print(f"      {r['rxn']}: fmax={r['first_final_fmax']:.4f} eV/Å, steps={r['first_steps']}")
        print()

    # Failed reactions: first-run trajectory
    print(f"=== FIRST-RUN TRAJECTORY for STILL FAILED reactions ({len(failed)}) ===")
    traj_counts_f = defaultdict(list)
    for r in failed:
        traj_counts_f[r['traj_class']].append(r)

    for cls, rxns in sorted(traj_counts_f.items()):
        fmaxs_end = [r['first_final_fmax'] for r in rxns if r['first_final_fmax'] is not None]
        steps     = [r['first_steps'] for r in rxns]
        print(f"  {cls}: {len(rxns)} reactions")
        if fmaxs_end:
            print(f"    first-run final fmax: {min(fmaxs_end):.3f} – {max(fmaxs_end):.3f} eV/Å")
        if steps:
            print(f"    first-run steps:      {min(steps)} – {max(steps)}  (mean {np.mean(steps):.0f})")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+',
                        default=['/home/energy/s242862/orca_neb_results',
                                 '/home/energy/s242862/orca_neb_val_results'])
    args = parser.parse_args()

    records = analyse(args.results)
    report(records)


if __name__ == '__main__':
    main()
