"""
AVAS -> CASSCF -> NEVPT2 pipeline using PySCF.

AVAS is run once at the TS to select the active space. The converged TS MO
coefficients are projected onto the reactant and product geometries, so all
three structures share the same physical orbital space and the barrier is valid.

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


def run_casscf_avas(xyz_path, label, output_dir, basis, avas_ao):
    """Run RHF + AVAS + CASSCF at the TS. Returns mc and the active space size."""
    print(f'\n{"="*60}')
    print(f'  {label} (AVAS selection): {xyz_path}')
    print(f'{"="*60}')

    mol = xyz_to_pyscf_mol(xyz_path, basis=basis)
    mf = scf.RHF(mol)
    mf.max_cycle = 500
    mf.kernel()

    print(f'\nAVAS target AOs: {avas_ao}')
    ncas, nelecas, mo = avas.avas(mf, avas_ao, threshold=0.2, canonicalize=True)
    print(f'AVAS selected: ({nelecas}e, {ncas}o)')

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = 500
    mc.internal_rotation = True
    mc.kernel(mo)

    return mf, mc, ncas, nelecas


def run_nevpt2_with_mo(xyz_path, label, output_dir, basis, ncas, nelecas, mo_coeff_ref, mol_ref):
    """Run RHF + CASSCF (initialized from projected TS MOs) + NEVPT2."""
    print(f'\n{"="*60}')
    print(f'  {label} (projected MOs, ({nelecas}e, {ncas}o)): {xyz_path}')
    print(f'{"="*60}')

    mol = xyz_to_pyscf_mol(xyz_path, basis=basis)
    mf = scf.RHF(mol)
    mf.max_cycle = 500
    mf.kernel()

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = 500
    mc.internal_rotation = True

    # Project TS MOs onto this geometry's basis — tracks same physical orbitals
    mo_init = mcscf.project_init_guess(mc, mo_coeff_ref, prev_mol=mol_ref)
    mc.kernel(mo_init)

    return _compute_nevpt2_result(mf, mc, mol, xyz_path, label, output_dir, ncas, nelecas)


def _compute_nevpt2_result(mf, mc, mol, xyz_path, label, output_dir, ncas, nelecas):
    e_nevpt2 = nevpt2.sc_nevpt(mc)
    e_total = mc.e_tot + e_nevpt2
    e_total_eV = e_total * 27.2114

    dm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
    occ_nat = np.linalg.eigvalsh(dm1)[::-1]

    print(f'\nNatural orbital occupations: {occ_nat}')
    print(f'NEVPT2 total energy: {e_total:.8f} Ha = {e_total_eV:.4f} eV')

    result = {
        'label':                  label,
        'xyz':                    xyz_path,
        'ncas':                   ncas,
        'nelecas':                int(nelecas) if hasattr(nelecas, '__int__') else nelecas,
        'e_hf_eV':                mf.e_tot * 27.2114,
        'e_casscf_eV':            mc.e_tot * 27.2114,
        'e_nevpt2_correction_eV': e_nevpt2 * 27.2114,
        'e_total_eV':             e_total_eV,
        'nat_occ':                occ_nat.tolist(),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{label}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main(args):
    os.makedirs(args.output, exist_ok=True)

    # Step 1: AVAS at TS — defines the active space and orbital reference
    mf_ts, mc_ts, ncas, nelecas = run_casscf_avas(
        args.ts_xyz, 'ts', args.output, args.basis, args.avas_ao
    )
    ts_result = _compute_nevpt2_result(
        mf_ts, mc_ts, mf_ts.mol, args.ts_xyz, 'ts', args.output, ncas, nelecas
    )

    # Step 2: Reactant and product — project TS MOs as starting guess
    results = {'ts': ts_result}
    for label, xyz in [('reactant', args.r_xyz), ('product', args.p_xyz)]:
        results[label] = run_nevpt2_with_mo(
            xyz, label, args.output, args.basis,
            ncas, nelecas, mc_ts.mo_coeff, mf_ts.mol
        )

    ts_eV = results['ts']['e_total_eV']
    r_eV  = results['reactant']['e_total_eV']
    p_eV  = results['product']['e_total_eV']

    barrier_f = (ts_eV - r_eV) * 1000
    barrier_r = (ts_eV - p_eV) * 1000

    print(f'\n{"="*60}')
    print(f'  NEVPT2 barrier (forward): {barrier_f:.1f} meV')
    print(f'  NEVPT2 barrier (reverse): {barrier_r:.1f} meV')
    print(f'  Active space: ({nelecas}e, {ncas}o) — fixed from TS AVAS')
    print(f'{"="*60}')

    summary = {
        'reaction':            args.reaction,
        'basis':               args.basis,
        'avas_ao':             args.avas_ao,
        'ncas':                ncas,
        'nelecas':             int(nelecas) if hasattr(nelecas, '__int__') else nelecas,
        'barrier_forward_meV': barrier_f,
        'barrier_reverse_meV': barrier_r,
        'ts':                  results['ts'],
        'reactant':            results['reactant'],
        'product':             results['product'],
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
    args = parser.parse_args()
    main(args)
