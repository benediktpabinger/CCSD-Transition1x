"""
Check progress of CCSD NEB array job.

Usage:
    python check_status.py \
        --results-dir ~/ccsd_pyscf_results \
        --reaction-list ~/test_reactions.txt
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir',   required=True)
parser.add_argument('--reaction-list', required=True)
args = parser.parse_args()

with open(args.reaction_list) as f:
    reactions = [line.strip() for line in f if line.strip()]

converged, running, failed, pending = [], [], [], []

for rxn in reactions:
    out_dir = os.path.join(args.results_dir, rxn)
    if not os.path.exists(out_dir):
        pending.append(rxn)
    elif os.path.exists(os.path.join(out_dir, 'converged')):
        converged.append(rxn)
    elif os.path.exists(os.path.join(out_dir, 'neb.log')):
        running.append(rxn)
    else:
        failed.append(rxn)

total = len(reactions)
print(f"Total:     {total}")
print(f"Converged: {len(converged):>4}  ({100*len(converged)/total:.1f}%)")
print(f"Running:   {len(running):>4}  ({100*len(running)/total:.1f}%)")
print(f"Pending:   {len(pending):>4}  ({100*len(pending)/total:.1f}%)")
print(f"Failed:    {len(failed):>4}  ({100*len(failed)/total:.1f}%)")

if failed:
    print(f"\nFailed reactions:")
    for r in failed:
        print(f"  {r}")
