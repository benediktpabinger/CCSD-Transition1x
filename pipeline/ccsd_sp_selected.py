"""
CCSD(T) single-point energies on CURATOR-selected configs.

Reads geometries from selected_configs.xyz and DFT energies from
selected_configs.json, runs CCSD(T) on each config, computes
delta = E_CCSD(T) - E_DFT, saves results.

Usage (single image, for SLURM array):
    python ccsd_sp_selected.py --round-dir ~/curator_results/round1 \
        --basis cc-pVDZ --image-idx 3 --n-threads 8

Gather all results:
    python ccsd_sp_selected.py --round-dir ~/curator_results/round1 \
        --basis cc-pVDZ --gather
"""
import argparse
import json
import os
import sys

from ase.io import read as ase_read
from ase import units
from pyscf import gto, scf, cc


def atoms_to_mol(atoms, basis, verbose=2):
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    mol = gto.M(
        atom=[(s, tuple(p)) for s, p in zip(symbols, positions)],
        basis=basis,
        unit='Angstrom',
        verbose=verbose,
    )
    return mol


def run_ccsdt(atoms, basis, n_threads, verbose=2):
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    mol = atoms_to_mol(atoms, basis, verbose)
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge")
    e_ccsd = float(mycc.e_tot)
    et = float(mycc.ccsd_t())
    e_ccsdt = e_ccsd + et
    return {
        'e_hf_Ha':    float(mf.e_tot),
        'e_ccsd_Ha':  e_ccsd,
        'e_t_Ha':     et,
        'e_ccsdt_Ha': e_ccsdt,
    }


def run_single(args):
    round_dir = args.round_dir
    xyz_path  = os.path.join(round_dir, 'selected_configs.xyz')
    json_path = os.path.join(round_dir, 'selected_configs.json')

    configs = ase_read(xyz_path, index=':')
    with open(json_path) as f:
        metadata = json.load(f)

    i = args.image_idx
    if i >= len(configs):
        print(f"ERROR: image_idx {i} out of range ({len(configs)} configs)")
        sys.exit(1)

    out_json = os.path.join(round_dir, f'ccsdt_sp_{i}.json')
    if os.path.exists(out_json) and not args.force:
        print(f"Already done: {out_json}. Use --force to recompute.")
        sys.exit(0)

    atoms = configs[i]
    meta  = metadata[i]
    print(f"\n=== Config {i}: {meta['symbols']} ({meta['n_atoms']} atoms) ===")

    try:
        res = run_ccsdt(atoms, args.basis, args.n_threads)
        res['image_idx']  = i
        res['symbols']    = meta['symbols']
        res['n_atoms']    = meta['n_atoms']
        res['e_dft_eV']   = meta['e_dft_eV']
        res['e_ccsdt_eV'] = res['e_ccsdt_Ha'] * units.Hartree
        res['delta_eV']   = res['e_ccsdt_eV'] - meta['e_dft_eV']
        print(f"  E_CCSD(T) = {res['e_ccsdt_Ha']:.8f} Ha")
        print(f"  E_DFT     = {meta['e_dft_eV']:.6f} eV")
        print(f"  delta     = {res['delta_eV']:+.6f} eV")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        res = {'image_idx': i, 'symbols': meta['symbols'], 'error': str(exc)}

    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Saved to {out_json}")


def gather(args):
    round_dir = args.round_dir
    json_path = os.path.join(round_dir, 'selected_configs.json')
    with open(json_path) as f:
        metadata = json.load(f)
    n = len(metadata)

    results = []
    for i in range(n):
        sp_json = os.path.join(round_dir, f'ccsdt_sp_{i}.json')
        if not os.path.exists(sp_json):
            print(f"  WARNING: missing ccsdt_sp_{i}.json")
            results.append({'image_idx': i, 'error': 'missing'})
            continue
        with open(sp_json) as f:
            results.append(json.load(f))

    out_json = os.path.join(round_dir, 'ccsdt_singlepoints.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    valid = [r for r in results if 'delta_eV' in r]
    print(f"  {len(valid)}/{n} configs completed")
    for r in valid:
        print(f"  [{r['image_idx']}] {r['symbols']:12s}  delta = {r['delta_eV']:+.4f} eV")
    print(f"{'='*55}")
    print(f"Saved to {out_json}")


def main(args):
    if args.gather:
        gather(args)
    elif args.image_idx is not None:
        run_single(args)
    else:
        for i in range(args.n_configs):
            args.image_idx = i
            run_single(args)
        args.image_idx = None
        gather(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CCSD(T) single-points on CURATOR-selected configs')
    parser.add_argument('--round-dir',  required=True, help='curator_results/roundN directory')
    parser.add_argument('--basis',      default='cc-pVDZ')
    parser.add_argument('--n-configs',  type=int, default=10)
    parser.add_argument('--n-threads',  type=int, default=8)
    parser.add_argument('--image-idx',  type=int, default=None)
    parser.add_argument('--gather',     action='store_true')
    parser.add_argument('--force',      action='store_true')
    args = parser.parse_args()
    main(args)
