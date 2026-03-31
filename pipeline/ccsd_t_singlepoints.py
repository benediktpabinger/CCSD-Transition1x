"""
CCSD(T) single-point energies on converged CCSD NEB images.

Reads the final NEB band from neb.db (last n_images rows = last converged band),
runs CCSD(T) on each image using PySCF, and saves per-image energies +
barrier to ccsdt_singlepoints.json.

Usage:
    python ccsd_t_singlepoints.py --neb-dir ~/ccsd_tz_results/rxn0103 \
        --basis cc-pVTZ --n-threads 8
"""
import argparse
import json
import os
import sys

import numpy as np
import ase.db
from ase import units
from pyscf import gto, scf, cc


def atoms_to_mol(atoms, basis, verbose=2):
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()  # Angstrom
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

    e_ccsd = float(mycc.e_tot)       # Hartree
    et = float(mycc.ccsd_t())        # perturbative triples correction (Ha)
    e_ccsdt = e_ccsd + et

    return {
        'e_hf_Ha':    float(mf.e_tot),
        'e_ccsd_Ha':  e_ccsd,
        'e_t_Ha':     et,
        'e_ccsdt_Ha': e_ccsdt,
    }


def load_final_band(neb_dir, n_images=10):
    """Read the last n_images rows from neb.db — the final converged band."""
    db_path = os.path.join(neb_dir, 'neb.db')
    all_atoms = []
    with ase.db.connect(db_path) as db:
        for row in db.select():
            all_atoms.append(row.toatoms())

    total = len(all_atoms)
    if total < n_images:
        raise ValueError(f"neb.db has only {total} rows, expected >= {n_images}")

    # Detect actual images-per-step from total / n_images divisibility
    # Use last n_images rows as final band
    final_band = all_atoms[-n_images:]
    print(f"neb.db: {total} total rows → using last {n_images} as final band")
    return final_band


def run_single_image(args):
    """Compute CCSD(T) for one image index, save to ccsdt_sp_{i}.json."""
    images = load_final_band(args.neb_dir, args.n_images)
    i = args.image_idx
    if i >= len(images):
        print(f"ERROR: image_idx {i} out of range (band has {len(images)} images)")
        sys.exit(1)

    out_json = os.path.join(args.neb_dir, f'ccsdt_sp_{i}.json')
    if os.path.exists(out_json) and not args.force:
        print(f"Already done: {out_json}. Use --force to recompute.")
        sys.exit(0)

    atoms = images[i]
    print(f"\n=== Image {i}/{len(images)-1} ({len(atoms)} atoms) ===")
    try:
        res = run_ccsdt(atoms, args.basis, args.n_threads)
        res['image_idx'] = i
        res['ccsd_eV'] = float(atoms.get_potential_energy())
        print(f"  E_CCSD(T) = {res['e_ccsdt_Ha']:.8f} Ha")
        print(f"  δ(T)      = {res['e_t_Ha']*1000:.4f} mHa")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        res = {'image_idx': i, 'error': str(exc)}

    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Saved to {out_json}")


def gather_results(args):
    """Combine per-image ccsdt_sp_*.json into ccsdt_singlepoints.json."""
    images = load_final_band(args.neb_dir, args.n_images)
    n = len(images)

    results = []
    e_ccsdt_list = []
    for i in range(n):
        sp_json = os.path.join(args.neb_dir, f'ccsdt_sp_{i}.json')
        if not os.path.exists(sp_json):
            print(f"  WARNING: missing ccsdt_sp_{i}.json")
            results.append({'image_idx': i, 'error': 'missing'})
            e_ccsdt_list.append(None)
            continue
        with open(sp_json) as f:
            res = json.load(f)
        results.append(res)
        e_ccsdt_list.append(res.get('e_ccsdt_Ha'))

    e0 = e_ccsdt_list[0]
    valid = [e for e in e_ccsdt_list if e is not None]
    e_max = max(valid) if valid else None
    ts_idx = e_ccsdt_list.index(e_max) if (e_max is not None and e_max in e_ccsdt_list) else None

    barrier_Ha   = (e_max - e0) if (e_max is not None and e0 is not None) else None
    barrier_eV   = barrier_Ha * units.Hartree if barrier_Ha is not None else None
    barrier_kcal = barrier_Ha * 627.509474 if barrier_Ha is not None else None

    ccsd_energies_eV  = [atoms.get_potential_energy() for atoms in images]
    ccsd_barrier_eV   = max(ccsd_energies_eV) - ccsd_energies_eV[0]
    ccsd_barrier_kcal = ccsd_barrier_eV / units.kcal * units.mol

    summary = {
        'neb_dir':               args.neb_dir,
        'basis':                 args.basis,
        'method':                f'CCSD(T)/{args.basis}//CCSD/{args.basis}',
        'n_images':              n,
        'ts_image_idx':          ts_idx,
        'barrier_Ha':            barrier_Ha,
        'barrier_eV':            barrier_eV,
        'barrier_kcal_mol':      barrier_kcal,
        'ccsd_barrier_eV':       ccsd_barrier_eV,
        'ccsd_barrier_kcal_mol': ccsd_barrier_kcal,
        'delta_T_kcal_mol':      (barrier_kcal - ccsd_barrier_kcal) if barrier_kcal is not None else None,
        'images':                results,
    }

    out_json = os.path.join(args.neb_dir, 'ccsdt_singlepoints.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  CCSD/{args.basis}    barrier: {ccsd_barrier_kcal:.2f} kcal/mol")
    print(f"  CCSD(T)/{args.basis} barrier: {barrier_kcal:.2f} kcal/mol")
    print(f"  δ(T) correction:      {(barrier_kcal - ccsd_barrier_kcal):.2f} kcal/mol")
    print(f"{'='*55}")
    print(f"Results saved to {out_json}")


def main(args):
    if not os.path.isdir(args.neb_dir):
        print(f"ERROR: directory not found: {args.neb_dir}")
        sys.exit(1)

    converged_flag = os.path.join(args.neb_dir, 'converged')
    if not os.path.exists(converged_flag) and not args.force:
        print(f"ERROR: 'converged' flag missing in {args.neb_dir}. Use --force to override.")
        sys.exit(1)

    if args.gather:
        gather_results(args)
    elif args.image_idx is not None:
        run_single_image(args)
    else:
        # Sequential fallback: run all images
        for i in range(args.n_images):
            args.image_idx = i
            run_single_image(args)
        args.image_idx = None
        gather_results(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CCSD(T) single-points on converged CCSD NEB images')
    parser.add_argument('--neb-dir',    required=True, help='Converged NEB output directory')
    parser.add_argument('--basis',      default='cc-pVDZ')
    parser.add_argument('--n-images',   type=int, default=10, help='Images per NEB band')
    parser.add_argument('--n-threads',  type=int, default=8,  help='OMP threads for PySCF')
    parser.add_argument('--image-idx',  type=int, default=None,
                        help='Run only this image index (0-based). Used by SLURM array tasks.')
    parser.add_argument('--gather',     action='store_true',
                        help='Gather per-image ccsdt_sp_*.json into ccsdt_singlepoints.json')
    parser.add_argument('--force',      action='store_true', help='Rerun even if already done')
    args = parser.parse_args()
    main(args)
