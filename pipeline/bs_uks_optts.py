"""
Broken-symmetry UKS (BS-UKS) OptTS for High-MR + next-HIGH reactions.
Uses wB97M-V/def2-TZVP to match the ORCA NEB reference level of theory.

Workflow:
  1. Run triplet UHF at the ORCA NEB TS to get open-shell MOs
  2. Use triplet MOs as initial guess for BS-UKS singlet (breaks spin symmetry)
  3. geomeTRIC eigenvector-following OptTS with UKS gradient
  4. Record final S², geometry, UKS energy

Output:
    ~/bs_uks_optts_results/{rxn}/bs_uks_ts.xyz
    ~/bs_uks_optts_results/{rxn}/bs_uks_results.json

Usage:
    python bs_uks_optts.py rxn7949 --n-threads 8
"""
import argparse, json, os
import numpy as np
from ase.io import read
from pyscf import gto, scf
from pyscf.geomopt import geometric_solver

HOME     = '/home/energy/s242862'
NEB_DIR  = f'{HOME}/orca_neb_results'
OUT_BASE = f'{HOME}/bs_uks_optts_results'
HA_TO_EV = 27.2114


def xyz_to_mol(xyz_path, basis, max_memory=60000):
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


def run_bs_uks(mol, xc, max_cycle=500):
    """Broken-symmetry UKS singlet.

    Strategy: run triplet UHF to get open-shell MOs, then use those as the
    initial density for a singlet UKS calculation. This reliably produces a
    BS solution for diradical-character TSs without manual orbital rotation.
    Falls back to a perturbed atom-guess if the triplet init collapses to RKS.
    """
    # ── Step 1: Triplet UHF for open-shell initial MOs ──────────────────────
    print('  Running triplet UHF for BS initial guess...', flush=True)
    mol_trip = mol.copy()
    mol_trip.spin = 2
    mol_trip.build()
    mf_trip = scf.UHF(mol_trip)
    mf_trip.max_cycle = 300
    mf_trip.kernel()
    print(f'  Triplet UHF converged: {mf_trip.converged}', flush=True)

    # ── Step 2: BS-UKS singlet from triplet MOs ─────────────────────────────
    mf = scf.UKS(mol)
    mf.xc = xc
    mf.max_cycle = max_cycle
    dm0 = mf.make_rdm1(mf_trip.mo_coeff, mf_trip.mo_occ)
    mf.kernel(dm0=dm0)
    ss, _ = mf.spin_square()
    print(f'  UKS (triplet init): converged={mf.converged}, <S²>={ss:.4f}', flush=True)

    # ── Fallback: perturbed atom guess if BS not found ───────────────────────
    if ss < 0.1:
        print('  BS collapsed to RKS — retrying with perturbed atom guess...', flush=True)
        mf2 = scf.UKS(mol)
        mf2.xc = xc
        mf2.max_cycle = max_cycle
        mf2.init_guess = 'atom'
        dm2 = list(mf2.get_init_guess())
        dm2[0] = dm2[0] * 1.10
        dm2[1] = dm2[1] * 0.90
        mf2.kernel(dm0=dm2)
        ss2, _ = mf2.spin_square()
        print(f'  UKS (atom init):   converged={mf2.converged}, <S²>={ss2:.4f}', flush=True)
        if ss2 > ss:
            mf, ss = mf2, ss2

    return mf, float(ss)


def main(args):
    rxn     = args.rxn
    out_dir = f'{OUT_BASE}/{rxn}'
    os.makedirs(out_dir, exist_ok=True)
    os.environ['OMP_NUM_THREADS'] = str(args.n_threads)
    max_memory = int(os.environ.get('PYSCF_MAX_MEMORY', 60000))

    ts_xyz = f'{NEB_DIR}/{rxn}/transition_state.xyz'
    print(f'\n{rxn}: BS-UKS OptTS  xc={args.xc}  basis={args.basis}', flush=True)

    mol_ts = xyz_to_mol(ts_xyz, args.basis, max_memory)
    mf, ss_init = run_bs_uks(mol_ts, args.xc, args.max_cycle)

    if not mf.converged:
        raise RuntimeError(f'{rxn}: UKS SCF not converged')

    bs_found = ss_init > 0.1
    print(f'\n{rxn}: BS solution found: {bs_found} (<S²>={ss_init:.4f})', flush=True)
    print(f'{rxn}: Starting geomeTRIC OptTS...', flush=True)

    mol_opt = geometric_solver.optimize(mf, transition=True, maxsteps=300)

    # Final SCF at optimised geometry to get clean S²
    mf_opt = scf.UKS(mol_opt)
    mf_opt.xc = args.xc
    mf_opt.max_cycle = 500
    mf_opt.kernel(dm0=mf.make_rdm1())
    ss_final, _ = mf_opt.spin_square()

    ts_out   = os.path.join(out_dir, 'bs_uks_ts.xyz')
    out_json = os.path.join(out_dir, 'bs_uks_results.json')
    mol_to_xyz(mol_opt, ts_out)

    result = {
        'reaction':   rxn,
        'xc':         args.xc,
        'basis':      args.basis,
        'bs_found':   bs_found,
        'ss_initial': round(ss_init, 4),
        'ss_final':   round(float(ss_final), 4),
        'e_uks_eV':   round(float(mf_opt.e_tot) * HA_TO_EV, 6),
        'ts_geometry': ts_out,
    }
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n{rxn}: <S²> init={ss_init:.4f} → final={ss_final:.4f}')
    print(f'{rxn}: Saved {out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--xc',        default='wb97m-v',
                        help='DFT functional (default: wb97m-v to match ORCA reference)')
    parser.add_argument('--basis',     default='def2-tzvp')
    parser.add_argument('--n-threads', type=int, default=8)
    parser.add_argument('--max-cycle', type=int, default=500)
    args = parser.parse_args()
    main(args)
