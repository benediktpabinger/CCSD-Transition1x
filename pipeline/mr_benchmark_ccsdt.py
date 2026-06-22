"""
MR benchmark Step 3: CCSD(T)/def2-TZVP single points on R, TS, P.
Geometries: ORCA NEB wB97M-V/def2-TZVP optimised stationary points by default,
or any other NEB result directory via --geom-dir (e.g. UMA-S or MACE+delta
TS/R/P geometries, to test geometry quality independent of the PES used to
find them).

Usage:
    python mr_benchmark_ccsdt.py <rxn> [--n-threads 24]
    python mr_benchmark_ccsdt.py <rxn> --geom-dir ~/uma_neb_results/<rxn> --tag uma_s
"""
import argparse
import json
import os

from ase.io import read
from pyscf import gto, scf, cc

HOME    = '/home/energy/s242862'
NEB_DIR = f'{HOME}/orca_neb_results'
OUT_DIR = f'{HOME}/mr_benchmark/results'

HA_TO_EV = 27.2114


def atoms_to_mol(atoms, basis):
    return gto.M(
        atom=[(s, tuple(p)) for s, p in
              zip(atoms.get_chemical_symbols(), atoms.get_positions())],
        basis=basis,
        unit='Angstrom',
        verbose=0,
    )


def run_ccsdt(atoms, basis, n_threads):
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    mol = atoms_to_mol(atoms, basis)

    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF not converged')

    mycc = cc.RCCSD(mf)
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError('CCSD not converged')

    et = float(mycc.ccsd_t())
    e_ccsdt = float(mycc.e_tot) + et

    return {
        'e_hf_Ha':    round(float(mf.e_tot),  8),
        'e_ccsd_Ha':  round(float(mycc.e_tot), 8),
        'e_t_Ha':     round(et,                8),
        'e_ccsdt_Ha': round(e_ccsdt,           8),
        'e_ccsdt_eV': round(e_ccsdt * HA_TO_EV, 6),
    }


def main(args):
    rxn     = args.rxn
    neb_rxn = args.geom_dir or f'{NEB_DIR}/{rxn}'
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix  = f'_{args.tag}' if args.tag else ''

    geometries = {
        'reactant':         f'{neb_rxn}/reactant.xyz',
        'transition_state': f'{neb_rxn}/transition_state.xyz',
        'product':          f'{neb_rxn}/product.xyz',
    }

    results = {'rxn': rxn, 'basis': args.basis, 'geometries': {}}

    for label, xyz_path in geometries.items():
        print(f'{rxn} {label} ...', flush=True)
        atoms = read(xyz_path)
        try:
            res = run_ccsdt(atoms, args.basis, args.n_threads)
            results['geometries'][label] = res
            print(f'  e_ccsdt = {res["e_ccsdt_eV"]:.4f} eV', flush=True)
        except RuntimeError as e:
            results['geometries'][label] = {'error': str(e)}
            print(f'  ERROR: {e}', flush=True)

    g = results['geometries']
    if all('e_ccsdt_eV' in g[k] for k in ['reactant', 'transition_state', 'product']):
        fwd = (g['transition_state']['e_ccsdt_eV'] - g['reactant']['e_ccsdt_eV']) * 1000
        rev = (g['transition_state']['e_ccsdt_eV'] - g['product']['e_ccsdt_eV']) * 1000
        results['barrier_fwd_meV'] = round(fwd, 1)
        results['barrier_rev_meV'] = round(rev, 1)
        print(f'{rxn}: fwd={fwd:.0f} meV  rev={rev:.0f} meV')

    out_path = f'{OUT_DIR}/{rxn}_ccsdt{suffix}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--basis',     default='def2-TZVP')
    parser.add_argument('--n-threads', type=int, default=24)
    parser.add_argument('--geom-dir',  default=None,
                        help='Directory with reactant.xyz/transition_state.xyz/product.xyz '
                             '(default: orca_neb_results/<rxn>)')
    parser.add_argument('--tag',       default=None,
                        help='Suffix for output filename, e.g. "uma_s" -> <rxn>_ccsdt_uma_s.json')
    args = parser.parse_args()
    main(args)
