"""
Sample training reactions for delta model data collection.

Randomly samples N reactions from the T1x training split and writes
their IDs to ~/ccsd_dataset/train_delta_rxns.txt.

Usage:
    python sample_train_reactions.py [--n-reactions 500] [--seed 42]
"""
import argparse
import os
import random

import h5py

HOME    = '/home/energy/s242862'
H5FILE  = f'{HOME}/data/Transition1x.h5'
OUT_DIR = f'{HOME}/ccsd_dataset'
OUT_TXT = f'{OUT_DIR}/train_delta_rxns.txt'


def main(args):
    print(f'Loading train split from {H5FILE}...')
    rxns = []
    with h5py.File(H5FILE, 'r') as f:
        train = f['train']
        for formula in train:
            for rxn in train[formula]:
                rxns.append(rxn)

    rxns.sort()
    n_total = len(rxns)
    n_sample = min(args.n_reactions, n_total)

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(rxns, n_sample))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_TXT, 'w') as f:
        for rxn in sampled:
            f.write(rxn + '\n')

    print(f'Train split: {n_total} reactions')
    print(f'Sampled {n_sample} with seed {args.seed}')
    print(f'Written to {OUT_TXT}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-reactions', type=int, default=500)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()
    main(args)
