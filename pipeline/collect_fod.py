"""
Collect FOD screening results and rank by NFOD.

Usage:
    python collect_fod.py --out-dir ~/fod_results --top 10
"""
import argparse
import json
import os


def main(args):
    results = []
    errors  = []

    for fname in sorted(os.listdir(args.out_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(args.out_dir, fname)) as f:
            d = json.load(f)
        if 'error' in d:
            errors.append(d)
        elif 'nfod' in d:
            results.append(d)

    results.sort(key=lambda r: r['nfod'], reverse=True)

    print(f'Completed: {len(results)}  Errors: {len(errors)}\n')
    print(f'{"Rank":<5}  {"rxn":<12}  {"NFOD":>8}  {"n_atoms":>8}  {"n_elec":>8}')
    print('-' * 50)
    for i, r in enumerate(results[:args.top], 1):
        print(f'{i:<5}  {r["rxn"]:<12}  {r["nfod"]:>8.4f}  {r["n_atoms"]:>8}  {r["n_elec"]:>8}')

    if errors:
        print(f'\nFailed reactions ({len(errors)}):')
        for e in errors:
            print(f'  {e["rxn"]}: {e["error"]}')

    out_path = os.path.join(args.out_dir, 'fod_ranking.json')
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'errors': errors}, f, indent=2)
    print(f'\nSaved ranking: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='/home/energy/s242862/fod_results')
    parser.add_argument('--top',     type=int, default=10)
    args = parser.parse_args()
    main(args)
