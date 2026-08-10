"""Null measurement for the TS-optimisation route.

Method B starts at the ORCA RKS reference TS and optimises to the nearest saddle
point.  On externally unstable reactions it moves onto the broken-symmetry
surface, and the distance it travels has been read as the error of the RKS
reference.  That reading assumes the method would not move at all if nothing
changed -- which has never been tested, because the batch only ever ran on
externally unstable reactions.

This runs the same machinery on reactions whose RKS solution is externally
STABLE.  There is no broken-symmetry solution to find, so the optimisation is a
plain RKS TS optimisation starting from a geometry that is already a converged
RKS TS -- in ORCA.  Any displacement measured here is method noise: PySCF vs
ORCA, different convergence criteria, a flat surface.

Reports the same quantities as the real batch so the numbers are comparable:
starting gradient, steps taken, RMSD start -> end.

Usage: python tsopt_null.py <rxn>
"""
import json
import os
import sys
import time

import numpy as np
from ase.io import read as ase_read
from pyscf import dft, gto
from pyscf.geomopt import geometric_solver

HOME = '/home/energy/s242862'
OUTDIR = f'{HOME}/tsopt_null'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
BOHR = 0.529177210903
MAXSTEPS = 300
HA_EVANG = 27.211386245988 / 0.529177210903


def kabsch_rmsd(P, Q):
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum((Pc @ R.T - Qc) ** 2, axis=1))))


def main(rxn):
    t0 = time.time()
    mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))
    out = f'{OUTDIR}/{rxn}'
    os.makedirs(out, exist_ok=True)
    res = {'rxn': rxn, 'method': f'{XC}/{BASIS} RKS TS opt from the ORCA '
                                 f'RKS reference TS (null measurement)'}

    ts_xyz = f'{HOME}/orca_neb_results/{rxn}/transition_state.xyz'
    atoms = ase_read(ts_xyz)
    start = atoms.get_positions().copy()

    mol = gto.Mole()
    mol.atom = '\n'.join(
        f'{s} {x:.8f} {y:.8f} {z:.8f}'
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), start))
    mol.basis = BASIS
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 4
    mol.max_memory = mem
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = 3
    mf.max_cycle = 300
    mf.conv_tol = 1e-10
    mf.max_memory = mem
    mf.kernel()
    if not mf.converged:
        res['status'] = 'RKS_NOT_CONVERGED'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        return 1
    res['e_rks_start'] = round(float(mf.e_tot), 10)

    # confirm the premise: no broken-symmetry solution here
    _, _, int_st, ext_st = mf.stability(internal=True, external=True,
                                        return_status=True)
    res['int_stable'] = bool(int_st)
    res['ext_stable'] = bool(ext_st)
    print(f'  int_stable={int_st}  ext_stable={ext_st}', flush=True)
    if not ext_st:
        res['status'] = 'NOT_A_NULL_CASE'
        res['note'] = 'externally unstable -- not usable as a null measurement'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        return 2

    g0 = mf.nuc_grad_method().kernel()
    res['grad_start'] = {
        'max_evang': round(float(np.abs(g0).max()) * HA_EVANG, 6),
        'rms_evang': round(float(np.sqrt((g0 ** 2).mean())) * HA_EVANG, 6)}
    print(f'  Startgradient max={res["grad_start"]["max_evang"]:.4f} eV/A',
          flush=True)

    print(f'  TS-Optimierung (transition=True, maxsteps={MAXSTEPS})', flush=True)
    conv, ts_mol = geometric_solver.kernel(mf, transition=True,
                                           maxsteps=MAXSTEPS)
    end = ts_mol.atom_coords() * BOHR

    res['opt_converged'] = bool(conv)
    res['rmsd_start_end'] = round(kabsch_rmsd(start, end), 6)
    res['e_rks_end'] = round(float(mf.e_tot), 10)
    res['de_meV'] = round((float(mf.e_tot) - res['e_rks_start']) * 27211.386, 3)
    res['elapsed_s'] = round(time.time() - t0, 1)

    with open(f'{out}/ts_null.xyz', 'w') as fh:
        fh.write(f'{ts_mol.natm}\n')
        fh.write(f'RKS TS opt from ORCA reference, RMSD={res["rmsd_start_end"]}\n')
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), end):
            fh.write(f'{s} {x:.8f} {y:.8f} {z:.8f}\n')

    json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
