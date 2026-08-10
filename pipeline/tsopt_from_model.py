"""Is the model geometry a usable starting point for a DFT transition-state
search?

The accuracy question -- how far is the model from the saddle -- is answered.
This asks the practical one: start a transition-state optimisation at the
model's predicted geometry and see whether it reaches the correct saddle, and
how much work that takes. If it does, the expensive path search can be skipped
and the model has paid for itself even where its own geometry is not exact.

Three outcomes matter and are distinguished in the output:
  lands on our saddle      the model geometry is inside the right basin
  lands elsewhere          it is in a different basin -- misleading, not merely
                           imprecise, which is worse for this use than a large
                           but harmless error
  does not converge        no useful answer either way

Which surface is used follows from the stability analysis, not an assumption:
broken symmetry where the RKS solution is externally unstable, plain RKS where
it is stable. For rxn1147 and rxn7060 the model geometries are stable, so those
run as RKS transition-state searches.

Usage: python tsopt_from_model.py <rxn> <model>
"""
import glob
import json
import os
import sys
import time

import numpy as np
from ase.io import read as ase_read
from pyscf import dft, gto
from pyscf.geomopt import geometric_solver

H = '/home/energy/s242862'
OUTDIR = f'{H}/tsopt_from_model'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
BOHR = 0.529177210903
S2_MIN = 0.05
MAXSTEPS = 300
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}

BS = {'dm': None, 'in_newton': False, 'step': 0, 'log': [], 'on_step': None}


class BSUKS(dft.uks.UKS):
    def kernel(self, dm0=None, **kwargs):
        if BS['in_newton']:
            return super().kernel(dm0=dm0, **kwargs)
        BS['in_newton'] = True
        try:
            n = self.newton()
            n.max_cycle = 200
            n.conv_tol = 1e-10
            n.kernel(dm0=BS['dm'])
            self.e_tot, self.mo_coeff = n.e_tot, n.mo_coeff
            self.mo_occ, self.mo_energy = n.mo_occ, n.mo_energy
            self.converged = n.converged
            BS['dm'] = n.make_rdm1()
            BS['step'] += 1
            s2 = float(n.spin_square()[0])
            BS['log'].append({'step': BS['step'], 'e': float(n.e_tot),
                              's2': round(s2, 6)})
            print(f'  geom {BS["step"]:3d}: E={n.e_tot:.10f}  S2={s2:.4f}',
                  flush=True)
            if BS['on_step']:
                BS['on_step']()
        finally:
            BS['in_newton'] = False
        return self.e_tot


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B):
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    V, S, W = np.linalg.svd(Ac.T @ Bc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    return float(np.sqrt((((Ac @ (V @ D @ W)) - Bc) ** 2).sum() / len(A)))


def our_ts(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    return f
    return None


def main(rx, model):
    t0 = time.time()
    mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))
    tag = f'{rx}_{model}'
    out = f'{OUTDIR}/{tag}'
    os.makedirs(out, exist_ok=True)
    res = {'rxn': rx, 'model': model}

    start = f'{H}/{MODELDIR[model]}/{rx}/transition_state.xyz'
    if not os.path.exists(start):
        res['status'] = 'STARTGEOMETRIE_FEHLT'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2); return 1
    atoms = ase_read(start)
    sym = atoms.get_chemical_symbols()
    x0 = atoms.get_positions().copy()
    print(f'{tag}: Start an der {model}-Geometrie', flush=True)

    mol = gto.Mole()
    mol.atom = '\n'.join(f'{s} {x:.8f} {y:.8f} {z:.8f}'
                         for s, (x, y, z) in zip(sym, x0))
    mol.basis = BASIS; mol.charge = 0; mol.spin = 0
    mol.verbose = 4; mol.max_memory = mem
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
    mf.kernel()
    if not mf.converged:
        res['status'] = 'RKS_NICHT_KONVERGIERT'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2); return 1
    _, mo_ext, _, ext_stable = mf.stability(internal=True, external=True,
                                            return_status=True)
    res['ext_stable'] = bool(ext_stable)
    res['surface'] = 'RKS' if ext_stable else 'BS'
    print(f'  extern stabil = {ext_stable} -> {res["surface"]}', flush=True)

    if ext_stable:
        driver = mf                       # RKS is the ground state here
    else:
        mf_s = mf.to_uks()
        mf_s.xc = XC; mf_s.grids.level = 3
        mf_s.max_cycle = 300; mf_s.conv_tol = 1e-10; mf_s.max_memory = mem
        n0 = mf_s.newton(); n0.max_cycle = 200; n0.conv_tol = 1e-10
        n0.kernel(mf_s.make_rdm1(mo_ext, mf_s.mo_occ))
        s2_0 = float(n0.spin_square()[0])
        res['bs_initial'] = {'s2': round(s2_0, 6),
                             'de_meV': round((float(n0.e_tot) - float(mf.e_tot))
                                             * 27211.386, 3)}
        if s2_0 < S2_MIN:
            res['status'] = 'KOLLABIERT'
            json.dump(res, open(f'{out}/result.json', 'w'), indent=2); return 3
        BS['dm'] = n0.make_rdm1()
        driver = BSUKS(mol)
        driver.xc = XC; driver.grids.level = 3
        driver.max_cycle = 300; driver.conv_tol = 1e-10; driver.max_memory = mem
        driver.mo_coeff, driver.mo_occ = n0.mo_coeff, n0.mo_occ
        driver.mo_energy = n0.mo_energy
        driver.e_tot, driver.converged = float(n0.e_tot), True

    def save():
        res['n_geom_steps'] = BS['step']
        res['step_log'] = BS['log']
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
    BS['on_step'] = save

    try:
        conv, ts_mol = geometric_solver.kernel(driver, transition=True,
                                               maxsteps=MAXSTEPS)
    except Exception as exc:
        res['status'] = f'FEHLER: {type(exc).__name__}: {exc}'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        print(res['status']); return 4
    BS['on_step'] = None

    xe = ts_mol.atom_coords() * BOHR
    res['opt_converged'] = bool(conv)
    res['n_geom_steps'] = BS['step']
    res['s2_final'] = BS['log'][-1]['s2'] if BS['log'] else None
    res['e_final'] = BS['log'][-1]['e'] if BS['log'] else float(driver.e_tot)
    res['rmsd_travelled'] = round(kabsch(x0, xe), 6)
    ours = our_ts(rx)
    if ours:
        res['rmsd_vs_our_ts'] = round(kabsch(xe, read_xyz(ours)[1]), 6)
    ref = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
    if os.path.exists(ref):
        res['rmsd_vs_rks_ref'] = round(kabsch(xe, read_xyz(ref)[1]), 6)

    d = res.get('rmsd_vs_our_ts')
    if not conv:
        res['outcome'] = 'nicht konvergiert'
    elif d is not None and d < 0.15:
        res['outcome'] = 'landet auf unserem Sattelpunkt'
    else:
        res['outcome'] = 'landet woanders'
    res['status'] = 'fertig'
    res['elapsed_s'] = round(time.time() - t0, 1)

    with open(f'{out}/ts_opt.xyz', 'w') as fh:
        fh.write(f'{ts_mol.natm}\n')
        fh.write(f'TS from {model} start  E={res["e_final"]}  '
                 f'S2={res["s2_final"]}  converged={conv}\n')
        for s, (a, b, c) in zip(sym, xe):
            fh.write(f'{s} {a:.8f} {b:.8f} {c:.8f}\n')
    save()
    print(json.dumps({k: v for k, v in res.items() if k != 'step_log'},
                     indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
