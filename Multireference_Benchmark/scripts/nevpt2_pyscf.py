"""
AVAS -> CASSCF -> NEVPT2 pipeline using PySCF.

AVAS automatically selects the active space based on AO projections onto
specified target orbitals (e.g. pi/pi* of C=O, C=C, N lone pairs).

Usage:
    python nevpt2_pyscf.py --reaction rxn0103 \
        --ts_xyz   ~/orca_neb_results/rxn0103/transition_state.xyz \
        --r_xyz    ~/orca_neb_results/rxn0103/reactant.xyz \
        --p_xyz    ~/orca_neb_results/rxn0103/product.xyz \
        --output   ~/nevpt2_results/rxn0103_pyscf \
        --basis    def2-tzvp \
        --avas_ao  'C 2pz' 'O 2pz' 'N 2p'
"""
import argparse
import json
import os
import numpy as np

from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.mrpt import nevpt2
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
    mol.max_memory = 32000
    mol.build()
    return mol


def run_nevpt2(xyz_path, label, output_dir, basis, avas_ao, ncas_fix=None, nelecas_fix=None, nroots=1):
    print(f'\n{"="*60}')
    print(f'  {label}: {xyz_path}')
    print(f'{"="*60}')

    mol = xyz_to_pyscf_mol(xyz_path, basis=basis)

    # RHF reference
    mf = scf.RHF(mol)
    mf.max_cycle = 500
    mf.kernel()

    # AVAS active space selection (MO rotation; size optionally overridden)
    print(f'\nAVAS target AOs: {avas_ao}')
    ncas_avas, nelecas_avas, mo = avas.avas(mf, avas_ao, threshold=0.2, canonicalize=True)
    print(f'AVAS selected: ({nelecas_avas}e, {ncas_avas}o)')

    if ncas_fix is not None and nelecas_fix is not None:
        ncas, nelecas = ncas_fix, nelecas_fix
        print(f'Override: fixing active space to ({nelecas}e, {ncas}o)')
    else:
        ncas, nelecas = ncas_avas, nelecas_avas

    # CASSCF
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = 500
    mc.level_shift = 0.1
    mc.kernel(mo)

    # NEVPT2
    e_nevpt2 = nevpt2.sc_nevpt(mc)
    e_total = mc.e_tot + e_nevpt2
    e_total_eV = e_total * 27.2114

    # Natural orbital occupations
    occ = mc.mo_occ[mc.ncore:mc.ncore + ncas]
    dm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
    occ_nat = np.linalg.eigvalsh(dm1)[::-1]

    print(f'\nNatural orbital occupations: {occ_nat}')
    print(f'NEVPT2 total energy: {e_total:.8f} Ha = {e_total_eV:.4f} eV')

    result = {
        'label':       label,
        'xyz':         xyz_path,
        'ncas':        ncas,
        'nelecas':     int(nelecas) if hasattr(nelecas, '__int__') else nelecas,
        'e_hf_eV':     mf.e_tot * 27.2114,
        'e_casscf_eV': mc.e_tot * 27.2114,
        'e_nevpt2_correction_eV': e_nevpt2 * 27.2114,
        'e_total_eV':  e_total_eV,
        'nat_occ':     occ_nat.tolist(),
        'avas_ao':     avas_ao,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{label}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main(args):
    os.makedirs(args.output, exist_ok=True)
    avas_ao = args.avas_ao

    results = {}
    for label, xyz in [('ts', args.ts_xyz), ('reactant', args.r_xyz), ('product', args.p_xyz)]:
        results[label] = run_nevpt2(xyz, label, args.output, args.basis, avas_ao,
                                    ncas_fix=args.ncas, nelecas_fix=args.nelecas)

    ts_eV = results['ts']['e_total_eV']
    r_eV  = results['reactant']['e_total_eV']
    p_eV  = results['product']['e_total_eV']

    barrier_f = (ts_eV - r_eV) * 1000
    barrier_r = (ts_eV - p_eV) * 1000

    print(f'\n{"="*60}')
    print(f'  NEVPT2 barrier (forward): {barrier_f:.1f} meV')
    print(f'  NEVPT2 barrier (reverse): {barrier_r:.1f} meV')
    print(f'  TS active space:       ({results["ts"]["nelecas"]}e, {results["ts"]["ncas"]}o)')
    print(f'  Reactant active space: ({results["reactant"]["nelecas"]}e, {results["reactant"]["ncas"]}o)')
    print(f'  Product active space:  ({results["product"]["nelecas"]}e, {results["product"]["ncas"]}o)')
    print(f'{"="*60}')

    summary = {
        'reaction':           args.reaction,
        'basis':              args.basis,
        'avas_ao':            avas_ao,
        'barrier_forward_meV': barrier_f,
        'barrier_reverse_meV': barrier_r,
        'ts':       results['ts'],
        'reactant': results['reactant'],
        'product':  results['product'],
    }
    out_json = os.path.join(args.output, 'nevpt2_results.json')
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
    parser.add_argument('--avas_ao',  nargs='+', default=['C 2pz', 'O 2pz', 'N 2p'],
                        help='Target AOs for AVAS selection')
    parser.add_argument('--ncas',    type=int, default=None,
                        help='Fix active space size (orbitals); overrides AVAS selection')
    parser.add_argument('--nelecas', type=int, default=None,
                        help='Fix active space size (electrons); overrides AVAS selection')
    args = parser.parse_args()
    main(args)
