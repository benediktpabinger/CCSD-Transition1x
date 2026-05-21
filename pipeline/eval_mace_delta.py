"""
Evaluate MACE p10 ep291 on delta-experiment geometries.

For each reaction (rxn0103, rxn1319, rxn7952) we have 90 ORCA NEB geometries
with both wB97X-D3/6-31G(d) (training functional) and wB97M-V/def2-TZVP
(higher-level reference) single-point energies.

Outputs per-geometry energy profiles for all three methods and
RMSE(MACE, wB97X-D3) vs RMSE(MACE, wB97M-V) to show which functional MACE tracks.

Usage:
    python eval_mace_delta.py \
        --model  ~/mace_sweep_results/mace_t1x_p10_epoch291.model \
        --delta-dir ~/delta_experiment/results \
        --output    ~/delta_experiment/mace_eval.json
"""
import argparse
import json
import os

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def ref_to_reactant(vals):
    return [v - vals[0] for v in vals]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))


def eval_reaction(rxn, delta_dir, geom_dir, calc):
    path = os.path.join(delta_dir, f'{rxn}.json')
    with open(path) as f:
        data = json.load(f)

    geoms = data['geometries']
    e_t1x_abs    = []
    e_silver_abs = []
    e_mace_abs   = []

    for g in geoms:
        xyz_path = os.path.join(geom_dir, rxn, f'geom_{g["geom_idx"]:03d}.xyz')
        atoms = read(xyz_path)
        atoms.calc = calc
        e_mace_abs.append(atoms.get_potential_energy())
        e_t1x_abs.append(g['e_t1x_eV'])
        e_silver_abs.append(g['e_silver_eV'])
        if (g['geom_idx'] + 1) % 10 == 0:
            print(f"  {rxn}  {g['geom_idx']+1}/{len(geoms)}", flush=True)

    # Relative energies (meV, referenced to reactant)
    de_t1x    = [v * 1000 for v in ref_to_reactant(e_t1x_abs)]
    de_silver = [v * 1000 for v in ref_to_reactant(e_silver_abs)]
    de_mace   = [v * 1000 for v in ref_to_reactant(e_mace_abs)]

    rmse_vs_t1x    = rmse(de_mace, de_t1x)
    rmse_vs_silver = rmse(de_mace, de_silver)

    ts_idx = int(np.argmax(de_t1x))

    per_geom = [
        {
            'geom_idx':    g['geom_idx'],
            'de_t1x_meV':    round(de_t1x[i],    2),
            'de_silver_meV': round(de_silver[i],  2),
            'de_mace_meV':   round(de_mace[i],    2),
        }
        for i, g in enumerate(geoms)
    ]

    return {
        'n_geometries':       len(geoms),
        'ts_idx':             ts_idx,
        'rmse_vs_t1x_meV':    round(rmse_vs_t1x,    1),
        'rmse_vs_silver_meV': round(rmse_vs_silver,  1),
        'barrier_t1x_meV':    round(de_t1x[ts_idx],    0),
        'barrier_silver_meV': round(de_silver[ts_idx],  0),
        'barrier_mace_meV':   round(de_mace[ts_idx],   0),
        'geometries':         per_geom,
    }


def main(args):
    print(f"Loading MACE model: {args.model}")
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    calc = MACECalculator(model_paths=args.model, device=device, default_dtype='float32')

    rxns = args.rxns.split(',')
    results = {}

    for rxn in rxns:
        print(f"\n{rxn} ...", flush=True)
        r = eval_reaction(rxn, args.delta_dir, args.geom_dir, calc)
        results[rxn] = r
        print(f"  RMSE vs wB97X-D3  : {r['rmse_vs_t1x_meV']:.1f} meV")
        print(f"  RMSE vs wB97M-V   : {r['rmse_vs_silver_meV']:.1f} meV")
        print(f"  Barrier (meV)     : wB97X-D3={r['barrier_t1x_meV']:.0f}  "
              f"wB97M-V={r['barrier_silver_meV']:.0f}  MACE={r['barrier_mace_meV']:.0f}")

    print("\n=== Summary ===")
    print(f"{'rxn':<12}  {'RMSE_t1x':>10}  {'RMSE_wB97M':>10}  "
          f"{'B_t1x':>8}  {'B_wB97M':>8}  {'B_mace':>8}  meV")
    for rxn, r in results.items():
        print(f"{rxn:<12}  {r['rmse_vs_t1x_meV']:>10.1f}  {r['rmse_vs_silver_meV']:>10.1f}  "
              f"{r['barrier_t1x_meV']:>8.0f}  {r['barrier_silver_meV']:>8.0f}  "
              f"{r['barrier_mace_meV']:>8.0f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     default='/home/energy/s242862/mace_t1x_p10_epoch291.model')
    parser.add_argument('--delta-dir', default='/home/energy/s242862/delta_experiment/results')
    parser.add_argument('--geom-dir',  default='/home/energy/s242862/delta_experiment/geometries')
    parser.add_argument('--output',    default='/home/energy/s242862/delta_experiment/mace_eval.json')
    parser.add_argument('--rxns',      default='rxn0103,rxn1319,rxn7952')
    args = parser.parse_args()
    main(args)
