"""
Merge all partial CCSD result files from a SLURM array job into one final JSON.
Usage:
    python merge_results.py --output-dir ccsd_rxn1961
"""
import json
import glob
import os
from argparse import ArgumentParser
import numpy as np


def main(args):
    pattern = os.path.join(args.output_dir, 'results_*.json')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No result files found in {args.output_dir}/")
        print("Expected files matching: results_XXXX_YYYY.json")
        return

    print(f"Found {len(files)} partial result files:")
    for f in files:
        print(f"  {f}")

    # Merge and sort by config_index
    all_results = []
    for f in files:
        with open(f) as fh:
            all_results.extend(json.load(fh))

    all_results.sort(key=lambda r: r['config_index'])

    # Summary
    total = len(all_results)
    successful = sum(1 for r in all_results if r['success'])
    failed = total - successful

    print(f"\nTotal configs: {total}")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")

    if successful > 0:
        diffs = [r['energy_diff'] for r in all_results if r.get('success') and 'energy_diff' in r]
        if diffs:
            print(f"\nEnergy comparison (CCSD vs wB97x):")
            print(f"  Mean absolute difference: {np.mean(diffs):.4f} eV")
            print(f"  Max absolute difference:  {np.max(diffs):.4f} eV")
            print(f"  Min absolute difference:  {np.min(diffs):.4f} eV")
            print(f"  Std deviation:            {np.std(diffs):.4f} eV")

    # Save merged file
    output_file = os.path.join(args.output_dir, 'ccsd_results_final.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nMerged results saved to: {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description='Merge partial CCSD result files from SLURM array job')
    parser.add_argument('--output-dir', required=True, help='Directory with results_*.json files')
    args = parser.parse_args()
    main(args)
