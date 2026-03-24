"""Generate test_reactions.txt for SLURM array job."""
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--h5file', required=True)
parser.add_argument('--split', default='test')
parser.add_argument('--output', default='test_reactions.txt')
args = parser.parse_args()

with h5py.File(args.h5file, 'r') as f:
    split = f[args.split]
    reactions = [rxn for formula in split for rxn in split[formula]]

with open(args.output, 'w') as out:
    for r in reactions:
        out.write(r + '\n')

print(f"Written {len(reactions)} reactions to {args.output}")
