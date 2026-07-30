"""
MR benchmark: MLIP single points on the ORCA (DFT) NEB geometries.

Same rationale as mr_benchmark_ccsdt.py / mr_benchmark_nevpt2.py: evaluate a
method as a single point on the fixed wB97M-V/def2-TZVP ORCA NEB
reactant/TS/product geometries, isolating level-of-theory differences from
geometry differences. Here the "method" is one of the five MLIPs (UMA-S,
UMA-M, eSEN, bare MACE, MACE+delta) instead of CCSD(T)/NEVPT2 -- these
normally run their own independent NEB and land on their own TS geometry;
this script instead asks "what barrier does this MLIP predict for the DFT
TS/R/P structures, without re-optimizing anything".

Usage:
    python mr_benchmark_mlip_sp.py <rxn> --method uma_s
    python mr_benchmark_mlip_sp.py <rxn> --method uma_m
    python mr_benchmark_mlip_sp.py <rxn> --method esen
    python mr_benchmark_mlip_sp.py <rxn> --method mace_bare
    python mr_benchmark_mlip_sp.py <rxn> --method mace_delta
"""
import argparse
import json
import os
import sys

from ase.io import read

HOME    = '/home/energy/s242862'
NEB_DIR = f'{HOME}/orca_neb_results'
OUT_DIR = f'{HOME}/mr_benchmark/results'

CHECKPOINTS = {
    'uma_s': f'{HOME}/checkpoints/uma-s-1p2.pt',
    'uma_m': f'{HOME}/checkpoints/uma-m-1p1.pt',
    'esen':  f'{HOME}/checkpoints/esen_sm_conserving_all.pt',
}

DELTA_HEAD_PATH = f'{HOME}/delta_head/delta_head_fw2.00.pt'

METHODS = ['uma_s', 'uma_m', 'esen', 'mace_bare', 'mace_delta']


def make_calc(method, device='cuda'):
    if method in ('uma_s', 'uma_m'):
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        predict_unit = pretrained_mlip.load_predict_unit(CHECKPOINTS[method], device=device)
        return FAIRChemCalculator(predict_unit, task_name='omol')
    if method == 'esen':
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        predict_unit = pretrained_mlip.load_predict_unit(CHECKPOINTS['esen'], device=device)
        return FAIRChemCalculator(predict_unit)
    if method in ('mace_bare', 'mace_delta'):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from mace_delta_neb import MACEDeltaCalculator, load_models
        model, delta_head, z_table, r_max = load_models(
            device, head_path=DELTA_HEAD_PATH, no_delta=(method == 'mace_bare'))
        return MACEDeltaCalculator(model, delta_head, z_table, r_max, device)
    raise ValueError(f'Unknown method: {method}')


def single_point_energy_eV(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    return float(atoms.get_potential_energy())


def run_barrier(rxn, method, calc, geom_dir=None):
    """Compute R/TS/P single-point energies and forward/reverse barriers
    for `method` on the ORCA NEB geometries of `rxn`. Returns the results
    dict; does not write to disk (caller decides where)."""
    neb_rxn = geom_dir or f'{NEB_DIR}/{rxn}'
    geometries = {
        'reactant':         f'{neb_rxn}/reactant.xyz',
        'transition_state': f'{neb_rxn}/transition_state.xyz',
        'product':          f'{neb_rxn}/product.xyz',
    }

    results = {'rxn': rxn, 'method': method, 'geometry_source': neb_rxn, 'energies_eV': {}}
    for label, xyz_path in geometries.items():
        atoms = read(xyz_path)
        e = single_point_energy_eV(atoms, calc)
        results['energies_eV'][label] = round(e, 6)
        print(f'{rxn} {method} {label}: {e:.6f} eV', flush=True)

    fwd = (results['energies_eV']['transition_state'] - results['energies_eV']['reactant']) * 1000
    rev = (results['energies_eV']['transition_state'] - results['energies_eV']['product']) * 1000
    results['barrier_fwd_meV'] = round(fwd, 1)
    results['barrier_rev_meV'] = round(rev, 1)
    print(f'{rxn} {method}: fwd={fwd:.1f} meV  rev={rev:.1f} meV')
    return results


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    calc = make_calc(args.method, device=args.device)
    results = run_barrier(args.rxn, args.method, calc, geom_dir=args.geom_dir)

    out_path = f'{OUT_DIR}/{args.rxn}_{args.method}_sp_dftneb.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--method',   required=True, choices=METHODS)
    parser.add_argument('--geom-dir', default=None,
                         help='Directory with reactant.xyz/transition_state.xyz/product.xyz '
                              '(default: orca_neb_results/<rxn>)')
    parser.add_argument('--device',   default='cuda')
    args = parser.parse_args()
    main(args)
