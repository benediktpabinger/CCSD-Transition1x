"""
CASSCF(AVAS) OptTS + NEVPT2/def2-TZVP for the 10 High MR benchmark reactions.

Workflow:
  1. RHF + AVAS at ORCA NEB TS → auto-selects (nel, norb) active space
  2. CASSCF at TS with AVAS MOs
  3. Geometric eigenvector-following OptTS (requires 'geometric' package)
  4. CASSCF+NEVPT2 at optimised TS (projected MOs from step 2)
  5. Projected CASSCF+NEVPT2 at ORCA R and P (same active space, MO continuity)

All three points share the same active space and orbital frame → barriers are
directly comparable.

Output:
    ~/nevpt2_optts_results/{rxn}_avas/nevpt2_optts_results.json
    ~/nevpt2_optts_results/{rxn}_avas/ts_casscf_opt.xyz

Usage:
    python mr_casscf_optts.py rxn7949 --n-threads 12
"""
import argparse
import json
import os

import numpy as np
from ase.io import read
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf.mrpt import nevpt2
from pyscf.geomopt import geometric_solver

HOME     = '/home/energy/s242862'
NEB_DIR  = f'{HOME}/orca_neb_results'
OUT_BASE = f'{HOME}/nevpt2_optts_results'

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


def mol_to_xyz(mol, path):
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords  = mol.atom_coords(unit='Angstrom')
    with open(path, 'w') as f:
        f.write(f'{mol.natm}\n\n')
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f'{s} {x:.6f} {y:.6f} {z:.6f}\n')


def run_rhf(mol):
    mf = scf.RHF(mol)
    mf.max_cycle = 500
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('RHF not converged')
    return mf


def run_casscf(mf, ncas, nelecas, mo_init, max_cycle, conv_tol=1e-7):
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro   = max_cycle
    mc.max_stepsize      = 0.05
    mc.internal_rotation = True
    mc.conv_tol          = conv_tol
    mc.conv_tol_grad     = conv_tol ** 0.5
    mc.kernel(mo_init)
    mc.solver_used = 'mc1step'
    if not mc.converged:
        print(f'  mc1step not converged (ncas={ncas}, nelecas={nelecas}); retrying with mc2step ...', flush=True)
        mc.mc2step(mo_init)
        mc.solver_used = 'mc2step'
    if not mc.converged:
        raise RuntimeError(f'CASSCF not converged (ncas={ncas}, nelecas={nelecas})')
    return mc


def prune_active_space(mf, ncas, nelecas, mo_init, occ_low=0.02, occ_high=1.98, min_ncas=2):
    """Drop AVAS-selected active orbitals whose natural occupation (from a
    cheap CASCI diagonalization, no orbital optimisation) is near 0 or 2 —
    these are marginal/intruder orbitals that cause CASSCF macro-iteration
    oscillation instead of contributing real multireference character.

    Safety: if pruning would leave fewer than min_ncas orbitals, it is
    skipped entirely and the original (unpruned) AVAS space is returned.
    This guards against reactions where ALL active orbitals sit just
    outside the occupation window (weak but real, distributed MR character)
    rather than a few genuinely degenerate/intruder orbitals — pruning
    those away would leave nothing (or a degenerate space) and break a
    case that previously converged fine unpruned.

    Returns: (ncas, nelecas, mo, info) where info records what happened.
    """
    ncore = (mf.mol.nelectron - nelecas) // 2
    mc0 = mcscf.CASCI(mf, ncas, nelecas)
    mc0.kernel(mo_init)
    info = {'attempted': True, 'pruned': False, 'orig_ncas': int(ncas), 'orig_nelecas': int(nelecas)}
    if not mc0.converged:
        print('  CASCI diagnostic did not converge; skipping active-space pruning', flush=True)
        info['skipped_reason'] = 'CASCI diagnostic not converged'
        return ncas, nelecas, mo_init, info

    dm1 = mc0.fcisolver.make_rdm1(mc0.ci, ncas, nelecas)
    occ, u = np.linalg.eigh(dm1)
    occ = occ[::-1]
    u   = u[:, ::-1]
    mo_active_no = mo_init[:, ncore:ncore + ncas] @ u

    high = [i for i, o in enumerate(occ) if o > occ_high]
    low  = [i for i, o in enumerate(occ) if o < occ_low]
    keep = [i for i, o in enumerate(occ) if occ_low <= o <= occ_high]

    if not high and not low:
        info['skipped_reason'] = 'nothing to prune'
        return ncas, nelecas, mo_init, info

    if len(keep) < min_ncas:
        print(f'  Pruning would leave only {len(keep)} active orbitals '
              f'(occ={[round(o,3) for o in occ]}); skipping pruning, keeping original AVAS space', flush=True)
        info['skipped_reason'] = f'would leave only {len(keep)} orbitals (< min_ncas={min_ncas})'
        info['occ_at_skip'] = [round(float(o), 4) for o in occ]
        return ncas, nelecas, mo_init, info

    print(f'  Pruning active space: dropping {len(high)} near-doubly-occ '
          f'(occ={[round(occ[i],3) for i in high]}) and {len(low)} near-empty '
          f'(occ={[round(occ[i],3) for i in low]}) orbitals', flush=True)

    mo_core      = mo_init[:, :ncore]
    mo_virt      = mo_init[:, ncore + ncas:]
    new_core_ext = mo_active_no[:, high]
    new_active   = mo_active_no[:, keep]
    new_virt_ext = mo_active_no[:, low]

    new_mo      = np.hstack([mo_core, new_core_ext, new_active, new_virt_ext, mo_virt])
    new_ncas    = new_active.shape[1]
    new_nelecas = nelecas - 2 * len(high)
    print(f'  Active space after pruning: ({new_nelecas}e, {new_ncas}o)', flush=True)
    info.update({'pruned': True, 'n_dropped_high': len(high), 'n_dropped_low': len(low),
                 'new_ncas': int(new_ncas), 'new_nelecas': int(new_nelecas)})
    return new_ncas, new_nelecas, new_mo, info


def project_and_run(mol, ncas, nelecas, mo_ref, mol_ref, max_cycle):
    """RHF + CASSCF at mol, initialising from projected MOs of mol_ref."""
    mf = run_rhf(mol)
    mc_tmp = mcscf.CASSCF(mf, ncas, nelecas)
    mo_init = mcscf.project_init_guess(mc_tmp, mo_ref, prev_mol=mol_ref)
    mc = run_casscf(mf, ncas, nelecas, mo_init, max_cycle)
    return mf, mc


def nevpt2_result(mf, mc, label, ncas, nelecas):
    e_corr  = nevpt2.sc_nevpt(mc)
    e_tot   = mc.e_tot + e_corr
    dm1     = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
    nat_occ = np.linalg.eigvalsh(dm1)[::-1].tolist()
    print(f'{label}: NEVPT2 total = {e_tot:.8f} Ha = {e_tot * HA_TO_EV:.4f} eV', flush=True)
    return {
        'label':                  label,
        'ncas':                   int(ncas),
        'nelecas':                int(nelecas),
        'e_hf_eV':                round(float(mf.e_tot)   * HA_TO_EV, 6),
        'e_casscf_eV':            round(float(mc.e_tot)   * HA_TO_EV, 6),
        'e_nevpt2_correction_eV': round(float(e_corr)     * HA_TO_EV, 6),
        'e_total_eV':             round(float(e_tot)      * HA_TO_EV, 6),
        'nat_occ':                nat_occ,
    }


def main(args):
    rxn     = args.rxn
    out_dir = f'{OUT_BASE}/{rxn}_avas'
    os.makedirs(out_dir, exist_ok=True)

    ts_xyz = f'{NEB_DIR}/{rxn}/transition_state.xyz'
    r_xyz  = f'{NEB_DIR}/{rxn}/reactant.xyz'
    p_xyz  = f'{NEB_DIR}/{rxn}/product.xyz'

    os.environ['OMP_NUM_THREADS'] = str(args.n_threads)
    max_memory = int(os.environ.get('PYSCF_MAX_MEMORY', 100000))

    # ── 1. RHF + AVAS + CASSCF at ORCA NEB TS ───────────────────────────────
    print(f'\n{rxn}: RHF + AVAS at TS ...', flush=True)
    mol_ts = xyz_to_mol(ts_xyz, args.basis, max_memory)
    mf_ts  = run_rhf(mol_ts)

    print(f'AVAS target AOs: {args.avas_ao}', flush=True)
    ncas, nelecas, mo_avas = avas.avas(mf_ts, args.avas_ao, threshold=args.avas_threshold, canonicalize=True)
    print(f'AVAS selected: ({nelecas}e, {ncas}o)', flush=True)
    avas_ncas, avas_nelecas = ncas, nelecas

    prune_info = {'attempted': False}
    if not args.no_prune:
        ncas, nelecas, mo_avas, prune_info = prune_active_space(mf_ts, ncas, nelecas, mo_avas)

    mc_ts = run_casscf(mf_ts, ncas, nelecas, mo_avas, args.max_cycle)

    # ── 2. CASSCF OptTS (geometric eigenvector-following) ────────────────────
    print(f'\n{rxn}: CASSCF OptTS ...', flush=True)
    mol_opt = geometric_solver.optimize(mc_ts, transition=True, maxsteps=300)

    ts_opt_xyz = os.path.join(out_dir, 'ts_casscf_opt.xyz')
    mol_to_xyz(mol_opt, ts_opt_xyz)
    print(f'Optimised TS: {ts_opt_xyz}', flush=True)

    # ── 3. CASSCF+NEVPT2 at optimised TS (project MOs from step 1) ──────────
    print(f'\n{rxn}: CASSCF+NEVPT2 at optimised TS ...', flush=True)
    mf_opt, mc_opt = project_and_run(
        mol_opt, ncas, nelecas, mc_ts.mo_coeff, mf_ts.mol, args.max_cycle
    )
    ts_result = nevpt2_result(mf_opt, mc_opt, 'ts', ncas, nelecas)

    # ── 4. Projected CASSCF+NEVPT2 at ORCA R and P ──────────────────────────
    results = {'ts': ts_result}
    for label, xyz_path in [('reactant', r_xyz), ('product', p_xyz)]:
        print(f'\n{rxn}: projected CASSCF+NEVPT2 at {label} ...', flush=True)
        mol_ep = xyz_to_mol(xyz_path, args.basis, max_memory)
        mf_ep, mc_ep = project_and_run(
            mol_ep, ncas, nelecas, mc_opt.mo_coeff, mf_opt.mol, args.max_cycle
        )
        results[label] = nevpt2_result(mf_ep, mc_ep, label, ncas, nelecas)

    ts_eV = results['ts']['e_total_eV']
    r_eV  = results['reactant']['e_total_eV']
    p_eV  = results['product']['e_total_eV']
    fwd   = (ts_eV - r_eV) * 1000
    rev   = (ts_eV - p_eV) * 1000

    summary = {
        'reaction':            rxn,
        'basis':               args.basis,
        'avas_ao':             args.avas_ao,
        'avas_threshold':      args.avas_threshold,
        'avas_ncas':           int(avas_ncas),
        'avas_nelecas':        int(avas_nelecas),
        'prune_enabled':       not args.no_prune,
        'prune_info':          prune_info,
        'ncas':                int(ncas),
        'nelecas':             int(nelecas),
        'solver_used_ts':      getattr(mc_ts, 'solver_used', None),
        'solver_used_ts_opt':  getattr(mc_opt, 'solver_used', None),
        'ts_geometry':         ts_opt_xyz,
        'barrier_forward_meV': round(fwd, 3),
        'barrier_reverse_meV': round(rev, 3),
        'ts':                  results['ts'],
        'reactant':            results['reactant'],
        'product':             results['product'],
    }

    out_json = os.path.join(out_dir, 'nevpt2_optts_results.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n{rxn}: fwd={fwd:.1f} meV  rev={rev:.1f} meV  active=({nelecas}e,{ncas}o)')
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--basis',     default='def2-tzvp')
    parser.add_argument('--n-threads', type=int, default=12)
    parser.add_argument('--max-cycle', type=int, default=1000)
    parser.add_argument('--avas-ao',        nargs='+',
                        default=['C 2pz', 'N 2p', 'O 2pz', 'F 2pz'])
    parser.add_argument('--avas-threshold', type=float, default=0.4)
    parser.add_argument('--no-prune', action='store_true',
                        help='disable post-AVAS active-space occupation pruning')
    args = parser.parse_args()
    main(args)
