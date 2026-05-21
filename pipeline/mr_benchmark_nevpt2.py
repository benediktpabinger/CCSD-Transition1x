"""
MR benchmark Step 4: NEVPT2/AVAS single points on R, TS, P.
AVAS is run once at the TS to define the active space; the converged TS
MO coefficients are projected onto R and P so all three share the same
physical orbital space and barriers are meaningful.

Geometries: ORCA NEB wB97M-V/def2-TZVP optimised stationary points.
Output:     ~/nevpt2_results/{rxn}_pyscf_avas/nevpt2_results.json

Usage:
    python mr_benchmark_nevpt2.py <rxn> [--basis def2-tzvp] [--n-threads 24]
        [--avas-ao 'C 2p' 'N 2p' 'O 2p' 'F 2p']
"""
import argparse
import json
import os

import numpy as np
from ase.io import read
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.mrpt import nevpt2

HOME    = '/home/energy/s242862'
NEB_DIR = f'{HOME}/orca_neb_results'
OUT_BASE = f'{HOME}/nevpt2_results'

HA_TO_EV = 27.2114


def xyz_to_mol(xyz_path, basis, max_memory=100000):
    atoms = read(xyz_path)
    atom_str = '\n'.join(
        f'{s} {x:.6f} {y:.6f} {z:.6f}'
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    )
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 4
    mol.max_memory = max_memory
    mol.build()
    return mol


def run_rhf(mol):
    mf = scf.RHF(mol)
    mf.max_cycle = 500
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF not converged')
    return mf


def run_casscf_avas(xyz_path, basis, avas_ao, max_memory):
    """RHF + AVAS at TS. Returns mf, mc, ncas, nelecas."""
    mol = xyz_to_mol(xyz_path, basis, max_memory)
    mf = run_rhf(mol)

    print(f'\nAVAS target AOs: {avas_ao}', flush=True)
    ncas, nelecas, mo = avas.avas(mf, avas_ao, threshold=0.2, canonicalize=True)
    print(f'AVAS selected: ({nelecas}e, {ncas}o)', flush=True)

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = args.max_cycle
    mc.max_stepsize = 0.05
    mc.internal_rotation = True
    mc.kernel(mo)
    if not mc.converged:
        raise RuntimeError(f'CASSCF not converged at TS (ncas={ncas}, nelecas={nelecas})')

    return mf, mc, ncas, nelecas


def run_nevpt2_projected(xyz_path, label, basis, ncas, nelecas, mo_coeff_ref, mol_ref, max_memory):
    """RHF + CASSCF (projected MOs) + NEVPT2 for R or P."""
    mol = xyz_to_mol(xyz_path, basis, max_memory)
    mf = run_rhf(mol)

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = args.max_cycle
    mc.max_stepsize = 0.05
    mc.internal_rotation = True
    mo_init = mcscf.project_init_guess(mc, mo_coeff_ref, prev_mol=mol_ref)
    mc.kernel(mo_init)
    if not mc.converged:
        raise RuntimeError(f'CASSCF not converged at {label}')

    return _nevpt2_result(mf, mc, xyz_path, label, ncas, nelecas)


def _nevpt2_result(mf, mc, xyz_path, label, ncas, nelecas):
    e_corr = nevpt2.sc_nevpt(mc)
    e_tot  = mc.e_tot + e_corr

    dm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
    nat_occ = np.linalg.eigvalsh(dm1)[::-1].tolist()

    print(f'{label}: NEVPT2 total = {e_tot:.8f} Ha = {e_tot * HA_TO_EV:.4f} eV', flush=True)

    return {
        'label':                  label,
        'xyz':                    xyz_path,
        'ncas':                   int(ncas),
        'nelecas':                int(nelecas),
        'e_hf_eV':                round(float(mf.e_tot) * HA_TO_EV, 6),
        'e_casscf_eV':            round(float(mc.e_tot) * HA_TO_EV, 6),
        'e_nevpt2_correction_eV': round(float(e_corr) * HA_TO_EV, 6),
        'e_total_eV':             round(float(e_tot) * HA_TO_EV, 6),
        'nat_occ':                nat_occ,
    }


def main(args):
    rxn     = args.rxn
    neb_rxn = f'{NEB_DIR}/{rxn}'
    out_dir = f'{OUT_BASE}/{rxn}_pyscf_avas'
    os.makedirs(out_dir, exist_ok=True)

    ts_xyz = f'{neb_rxn}/transition_state.xyz'
    r_xyz  = f'{neb_rxn}/reactant.xyz'
    p_xyz  = f'{neb_rxn}/product.xyz'

    os.environ['OMP_NUM_THREADS'] = str(args.n_threads)
    max_memory = int(os.environ.get('PYSCF_MAX_MEMORY', 100000))

    print(f'\n{rxn}: AVAS at TS ...', flush=True)
    mf_ts, mc_ts, ncas, nelecas = run_casscf_avas(
        ts_xyz, args.basis, args.avas_ao, max_memory
    )
    ts_result = _nevpt2_result(mf_ts, mc_ts, ts_xyz, 'ts', ncas, nelecas)

    results = {'ts': ts_result}
    for label, xyz in [('reactant', r_xyz), ('product', p_xyz)]:
        print(f'\n{rxn}: projected CASSCF+NEVPT2 at {label} ...', flush=True)
        results[label] = run_nevpt2_projected(
            xyz, label, args.basis, ncas, nelecas,
            mc_ts.mo_coeff, mf_ts.mol, max_memory
        )

    ts_eV = results['ts']['e_total_eV']
    r_eV  = results['reactant']['e_total_eV']
    p_eV  = results['product']['e_total_eV']
    fwd   = (ts_eV - r_eV) * 1000
    rev   = (ts_eV - p_eV) * 1000

    summary = {
        'reaction':            rxn,
        'basis':               args.basis,
        'avas_ao':             args.avas_ao,
        'ncas':                int(ncas),
        'nelecas':             int(nelecas),
        'barrier_forward_meV': round(fwd, 3),
        'barrier_reverse_meV': round(rev, 3),
        'ts':                  results['ts'],
        'reactant':            results['reactant'],
        'product':             results['product'],
    }

    out_json = f'{out_dir}/nevpt2_results.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{rxn}: fwd={fwd:.1f} meV  rev={rev:.1f} meV  active=({nelecas}e,{ncas}o)')
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--basis',     default='def2-tzvp')
    parser.add_argument('--n-threads', type=int, default=24)
    parser.add_argument('--max-cycle', type=int, default=1000)
    parser.add_argument('--avas-ao',   nargs='+',
                        default=['C 2pz', 'N 2p', 'O 2pz', 'F 2pz'])
    args = parser.parse_args()
    main(args)
