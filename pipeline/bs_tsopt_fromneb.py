"""BS-UKS TS optimisation started from the NEB transition state.

Three reactions were abandoned by the earlier batches because the broken-symmetry
solution at the *RKS reference* geometry was too weak to pass the <S^2> > 0.3
gate: rxn4113 (0.14), rxn6196 (0.22), rxn5690 (0.07).

The ORCA BS-NEB then found strongly broken paths for them -- rxn4113 runs at
<S^2> ~ 1.0 across six consecutive images -- which suggests the RKS reference
sits at the edge of the broken-symmetry region while the real saddle lies
further along, where the solution is fully developed.

This starts the same optimisation from the NEB transition state instead. If it
converges to the NEB structure, the two independent routes agree and the case is
closed; if it walks somewhere else, that localises the disagreement.

The <S^2> gate is lowered to 0.05 here. The 0.3 threshold was shown to be wrong
by the frequency job: rxn3107 (0.18) and rxn8885 (0.15) were flagged as failures
and both turned out to be genuine transition states with exactly one imaginary
frequency. The correct criterion is the sign of lambda_min_ext, which is checked
explicitly below.

Usage: python bs_tsopt_fromneb.py <rxn>
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
OUTDIR = f'{HOME}/bs_tsopt_fromneb'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
BOHR = 0.529177210903
S2_MIN = 0.05           # not 0.3 -- see module docstring
MAXSTEPS = 300
HA_MEV = 27211.386

BS = {'dm': None, 'in_newton': False, 'step': 0, 'log': [],
      'lost_at': None, 'on_step': None, 'last_good_dm': None}


class BSUKS(dft.uks.UKS):
    """Class-level kernel: as_scanner() copies the SCF object, so an
    instance attribute would write to the wrong one."""

    def kernel(self, dm0=None, **kwargs):
        if BS['in_newton']:
            return super().kernel(dm0=dm0, **kwargs)
        BS['in_newton'] = True
        try:
            n = self.newton()
            n.max_cycle = 200
            n.conv_tol = 1e-10
            n.kernel(dm0=BS['dm'])      # density matrix, never stale mo_coeff
            self.e_tot, self.mo_coeff = n.e_tot, n.mo_coeff
            self.mo_occ, self.mo_energy = n.mo_occ, n.mo_energy
            self.converged = n.converged
            dm = n.make_rdm1()
            BS['dm'] = dm
            BS['step'] += 1
            s2 = float(n.spin_square()[0])
            if s2 >= S2_MIN:
                BS['last_good_dm'] = dm
            elif BS['lost_at'] is None:
                BS['lost_at'] = BS['step']
                print(f'  *** S2={s2:.4f} < {S2_MIN} at step {BS["step"]} ***',
                      flush=True)
            BS['log'].append({'step': BS['step'], 'e': float(n.e_tot),
                              's2': round(s2, 6), 'conv': bool(n.converged)})
            print(f'  geom {BS["step"]:3d}: E={n.e_tot:.10f}  S2={s2:.4f}  '
                  f'conv={n.converged}', flush=True)
            if BS['on_step']:
                BS['on_step']()
        finally:
            BS['in_newton'] = False
        return self.e_tot


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
    res = {'rxn': rxn,
           'method': f'{XC}/{BASIS} BS-UKS TS opt started from the ORCA BS-NEB TS',
           's2_min_gate': S2_MIN}

    start_xyz = f'{HOME}/bs_uks_neb_results/{rxn}/bs_uks_neb_NEB-TS_converged.xyz'
    if not os.path.exists(start_xyz):
        res['status'] = 'NO_NEB_TS'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        print('kein NEB-TS vorhanden'); return 1

    atoms = ase_read(start_xyz)
    start = atoms.get_positions().copy()
    sym = atoms.get_chemical_symbols()

    mol = gto.Mole()
    mol.atom = '\n'.join(f'{s} {x:.8f} {y:.8f} {z:.8f}'
                         for s, (x, y, z) in zip(sym, start))
    mol.basis = BASIS
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 4
    mol.max_memory = mem
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
    mf.kernel()
    if not mf.converged:
        res['status'] = 'RKS_NOT_CONVERGED'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2); return 1
    e_rks = float(mf.e_tot)
    res['e_rks'] = round(e_rks, 10)

    _, mo_ext, int_st, ext_st = mf.stability(internal=True, external=True,
                                             return_status=True)
    res['int_stable'], res['ext_stable'] = bool(int_st), bool(ext_st)
    print(f'  int={int_st} ext={ext_st}', flush=True)
    if ext_st:
        res['status'] = 'EXT_STABLE_AT_NEB_TS'
        res['note'] = 'no broken-symmetry solution at the NEB TS geometry'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        print('extern stabil -- keine BS-Loesung hier'); return 2

    mf_s = mf.to_uks()                  # to_uks(), not a fresh UKS: mo_occ
    mf_s.xc = XC; mf_s.grids.level = 3
    mf_s.max_cycle = 300; mf_s.conv_tol = 1e-10; mf_s.max_memory = mem
    n0 = mf_s.newton(); n0.max_cycle = 200; n0.conv_tol = 1e-10
    n0.kernel(mf_s.make_rdm1(mo_ext, mf_s.mo_occ))
    s2_0 = float(n0.spin_square()[0])
    res['bs_initial'] = {'e_uks': round(float(n0.e_tot), 10),
                         'de_meV': round((float(n0.e_tot) - e_rks) * HA_MEV, 3),
                         's2': round(s2_0, 6), 'converged': bool(n0.converged)}
    print(f'  BS: dE={res["bs_initial"]["de_meV"]:.1f} meV  S2={s2_0:.4f}',
          flush=True)
    if s2_0 < S2_MIN:
        res['status'] = 'COLLAPSED'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        print('kollabiert'); return 3

    BS['dm'] = n0.make_rdm1()
    mf_u = BSUKS(mol)
    mf_u.xc = XC; mf_u.grids.level = 3
    mf_u.max_cycle = 300; mf_u.conv_tol = 1e-10; mf_u.max_memory = mem
    mf_u.mo_coeff, mf_u.mo_occ = n0.mo_coeff, n0.mo_occ
    mf_u.mo_energy, mf_u.e_tot, mf_u.converged = n0.mo_energy, float(n0.e_tot), True

    def save():
        res['n_geom_steps'] = BS['step']
        res['step_log'] = BS['log']
        res['bs_lost_at_step'] = BS['lost_at']
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
    BS['on_step'] = save

    conv, ts_mol = geometric_solver.kernel(mf_u, transition=True,
                                           maxsteps=MAXSTEPS)
    BS['on_step'] = None
    end = ts_mol.atom_coords() * BOHR

    res['opt_converged'] = bool(conv)
    res['s2_final'] = BS['log'][-1]['s2'] if BS['log'] else None
    res['e_uks_final'] = BS['log'][-1]['e'] if BS['log'] else None
    res['rmsd_vs_neb_ts'] = round(kabsch_rmsd(start, end), 6)
    ref = f'{HOME}/orca_neb_results/{rxn}/transition_state.xyz'
    if os.path.exists(ref):
        res['rmsd_vs_rks_ref'] = round(
            kabsch_rmsd(ase_read(ref).get_positions(), end), 6)
    res['status'] = 'converged' if conv else 'not_converged'
    res['elapsed_s'] = round(time.time() - t0, 1)

    with open(f'{out}/ts_opt.xyz', 'w') as fh:
        fh.write(f'{ts_mol.natm}\n')
        fh.write(f'BS-UKS TS from NEB start  E={res["e_uks_final"]}  '
                 f'S2={res["s2_final"]}  converged={conv}\n')
        for s, (x, y, z) in zip(sym, end):
            fh.write(f'{s} {x:.8f} {y:.8f} {z:.8f}\n')

    save()
    print(json.dumps({k: v for k, v in res.items() if k != 'step_log'},
                     indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
