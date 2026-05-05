"""
RHF -> CCSD -> CCSD(T) single-points using PySCF.
Same geometries and basis as the NEVPT2 pipeline for direct comparison.

Usage:
    python ccsd_pyscf.py --reaction rxn0103 \
        --ts_xyz  ~/orca_neb_results/rxn0103/transition_state.xyz \
        --r_xyz   ~/orca_neb_results/rxn0103/reactant.xyz \
        --p_xyz   ~/orca_neb_results/rxn0103/product.xyz \
        --output  ~/nevpt2_results/rxn0103_ccsd_pyscf \
        --basis   def2-tzvp
"""
import argparse
import json
import os

from pyscf import gto, scf, cc
from ase.io import read


def xyz_to_pyscf_mol(xyz_path, basis='def2-tzvp', charge=0, spin=0):
    atoms = read(xyz_path)
    atom_str = '\n'.join(
        f'{s} {x:.6f} {y:.6f} {z:.6f}'
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    )
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 4
    mol.max_memory = 60000
    mol.build()
    return mol


def run_ccsd(xyz_path, label, output_dir, basis):
    print(f'\n{"="*60}')
    print(f'  {label}: {xyz_path}')
    print(f'{"="*60}')

    mol = xyz_to_pyscf_mol(xyz_path, basis=basis)

    mf = scf.RHF(mol)
    mf.max_cycle = 500
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.kernel()

    e_t = mycc.ccsd_t()

    e_hf_eV    = mf.e_tot    * 27.2114
    e_ccsd_eV  = mycc.e_tot  * 27.2114
    e_ccsdt_eV = (mycc.e_tot + e_t) * 27.2114

    print(f'\nHF      energy: {e_hf_eV:.4f} eV')
    print(f'CCSD    energy: {e_ccsd_eV:.4f} eV')
    print(f'CCSD(T) energy: {e_ccsdt_eV:.4f} eV')

    result = {
        'label':      label,
        'xyz':        xyz_path,
        'basis':      basis,
        'e_hf_eV':    e_hf_eV,
        'e_ccsd_eV':  e_ccsd_eV,
        'e_ccsdt_eV': e_ccsdt_eV,
        'e_t_eV':     e_t * 27.2114,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{label}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main(args):
    os.makedirs(args.output, exist_ok=True)

    results = {}
    for label, xyz in [('ts', args.ts_xyz), ('reactant', args.r_xyz), ('product', args.p_xyz)]:
        results[label] = run_ccsd(xyz, label, args.output, args.basis)

    ts_ccsd   = results['ts']['e_ccsd_eV']
    ts_ccsdt  = results['ts']['e_ccsdt_eV']
    r_ccsd    = results['reactant']['e_ccsd_eV']
    r_ccsdt   = results['reactant']['e_ccsdt_eV']
    p_ccsd    = results['product']['e_ccsd_eV']
    p_ccsdt   = results['product']['e_ccsdt_eV']

    barrier_ccsd_f  = (ts_ccsd  - r_ccsd)  * 1000
    barrier_ccsd_r  = (ts_ccsd  - p_ccsd)  * 1000
    barrier_ccsdt_f = (ts_ccsdt - r_ccsdt) * 1000
    barrier_ccsdt_r = (ts_ccsdt - p_ccsdt) * 1000

    print(f'\n{"="*60}')
    print(f'  CCSD    barrier (forward): {barrier_ccsd_f:.1f} meV')
    print(f'  CCSD    barrier (reverse): {barrier_ccsd_r:.1f} meV')
    print(f'  CCSD(T) barrier (forward): {barrier_ccsdt_f:.1f} meV')
    print(f'  CCSD(T) barrier (reverse): {barrier_ccsdt_r:.1f} meV')
    print(f'{"="*60}')

    summary = {
        'reaction':                   args.reaction,
        'basis':                      args.basis,
        'barrier_ccsd_forward_meV':   barrier_ccsd_f,
        'barrier_ccsd_reverse_meV':   barrier_ccsd_r,
        'barrier_ccsdt_forward_meV':  barrier_ccsdt_f,
        'barrier_ccsdt_reverse_meV':  barrier_ccsdt_r,
        'ts':       results['ts'],
        'reactant': results['reactant'],
        'product':  results['product'],
    }
    out_json = os.path.join(args.output, 'ccsd_results.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reaction', required=True)
    parser.add_argument('--ts_xyz',   required=True)
    parser.add_argument('--r_xyz',    required=True)
    parser.add_argument('--p_xyz',    required=True)
    parser.add_argument('--output',   required=True)
    parser.add_argument('--basis',    default='def2-tzvp')
    args = parser.parse_args()
    main(args)
