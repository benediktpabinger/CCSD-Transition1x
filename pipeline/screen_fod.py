"""
FOD (Fractional Occupation number weighted Density) screening for multireference character.

Runs PySCF PBE/def2-SVP at Tel=5000K on a single TS geometry and computes NFOD:
    NFOD = sum_i |n_i - n0_i|
         = 2 * sum_{virtual} n_i   (by electron conservation)

NFOD > 0.1 flags possible MR character; > 1.0 is strong.

Usage:
    python screen_fod.py <rxn> --neb-dir ~/orca_neb_results --out-dir ~/fod_results
"""
import argparse
import json
import os
import sys

import numpy as np
from pyscf import gto, dft
from pyscf.scf.addons import smearing_
from ase.io import read


K_TO_HA = 3.16681e-6   # Boltzmann constant in Ha/K
T_EL    = 5000.0        # K  (Grimme standard)


def atoms_to_pyscf_atom(atoms):
    lines = []
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f'{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}')
    return '; '.join(lines)


def compute_fod(xyz_path, basis='def2-SVP', xc='PBE'):
    atoms = read(xyz_path)
    atom_str = atoms_to_pyscf_atom(atoms)

    mol = gto.M(
        atom=atom_str,
        basis=basis,
        charge=0,
        spin=0,
        verbose=0,
    )

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.max_cycle = 300
    sigma = T_EL * K_TO_HA
    mf = smearing_(mf, sigma=sigma, method='fermi')

    e = mf.kernel()
    if not mf.converged:
        return None, 'SCF not converged'

    mo_occ = np.array(mf.mo_occ)
    n_elec = mol.nelectron
    n_occ  = n_elec // 2

    # reference occupations at T=0: 2 for lowest n_occ MOs, 0 for rest
    n0 = np.zeros_like(mo_occ)
    n0[:n_occ] = 2.0

    nfod = float(np.sum(np.abs(mo_occ - n0)))
    # equivalently: 2 * sum of virtual occupations
    nfod_check = float(2 * np.sum(mo_occ[n_occ:]))

    return {
        'nfod':       round(nfod, 6),
        'nfod_check': round(nfod_check, 6),
        'n_atoms':    len(atoms),
        'n_elec':     int(n_elec),
        'energy_Ha':  round(e, 8),
    }, None


def main(args):
    rxn     = args.rxn
    xyz     = os.path.join(args.neb_dir, rxn, 'transition_state.xyz')
    out_dir = args.out_dir
    out     = os.path.join(out_dir, f'{rxn}.json')

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(xyz):
        print(f'{rxn}: no TS geometry — skipping', flush=True)
        sys.exit(0)

    print(f'{rxn}: running FOD (Tel={T_EL:.0f}K, {args.basis}/{args.xc})', flush=True)
    result, err = compute_fod(xyz, basis=args.basis, xc=args.xc)

    if err:
        out_data = {'rxn': rxn, 'error': err}
        print(f'{rxn}: FAILED — {err}', flush=True)
    else:
        out_data = {'rxn': rxn, **result}
        print(f'{rxn}: NFOD = {result["nfod"]:.4f}', flush=True)

    with open(out, 'w') as f:
        json.dump(out_data, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--neb-dir', default='/home/energy/s242862/orca_neb_results')
    parser.add_argument('--out-dir', default='/home/energy/s242862/fod_results')
    parser.add_argument('--basis',   default='def2-SVP')
    parser.add_argument('--xc',      default='PBE')
    args = parser.parse_args()
    main(args)
