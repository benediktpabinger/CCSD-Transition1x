"""
Batch driver for mr_benchmark_mlip_sp.py: loads one MLIP calculator once and
loops over multiple reactions, so a GPU job doesn't pay model-load cost once
per reaction. Writes the same per-reaction JSON files as the single-reaction
CLI (`{rxn}_{method}_sp_dftneb.json` in mr_benchmark/results/).

Usage:
    python mr_benchmark_mlip_sp_batch.py --method uma_s \
        --rxns rxn7949,rxn8832,rxn1320,rxn4113,rxn8885,rxn7945,rxn7937,rxn6196,rxn0346,rxn1150,rxn0896
"""
import argparse
import json
import os

from mr_benchmark_mlip_sp import METHODS, OUT_DIR, make_calc, run_barrier


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    rxns = args.rxns.split(',')
    calc = make_calc(args.method, device=args.device)

    summary = []
    for rxn in rxns:
        out_path = f'{OUT_DIR}/{rxn}_{args.method}_sp_dftneb.json'
        if os.path.exists(out_path) and not args.overwrite:
            print(f'{rxn} {args.method}: already done, skipping ({out_path})')
            continue
        try:
            results = run_barrier(rxn, args.method, calc)
        except Exception as e:
            print(f'{rxn} {args.method}: ERROR: {e}', flush=True)
            results = {'rxn': rxn, 'method': args.method, 'error': str(e)}
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved: {out_path}')
        summary.append(results)

    summary_path = f'{OUT_DIR}/summary_{args.method}_sp_dftneb.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary: {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',    required=True, choices=METHODS)
    parser.add_argument('--rxns',      required=True, help='comma-separated reaction IDs')
    parser.add_argument('--device',    default='cuda')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    main(args)
