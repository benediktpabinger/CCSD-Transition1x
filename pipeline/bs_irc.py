"""Does the saddle point connect the right reactant and product?

A frequency calculation is local: it proves a structure is a first-order saddle,
not that the reaction passes through it. The standard answer is an intrinsic
reaction coordinate, which PySCF does not provide. This is the practical
substitute: displace along the imaginary mode in both directions, relax, and see
which minima you arrive at.

That is weaker than a true IRC -- an unconstrained relaxation can slide past a
shallow intermediate that steepest descent in mass-weighted coordinates would
have stopped at -- but it answers the question that matters here: does this
saddle sit between the reactant and product of this reaction, or somewhere else.

Broken symmetry is carried through the relaxation the same way as in the TS
optimisation: the previous step's density matrix seeds the next, never its MO
coefficients. <S^2> is logged, because the relaxation runs out of the broken
region by construction -- the endpoints are closed shell -- and that is correct
rather than a failure.

Usage: python bs_irc.py <rxn>
"""
import glob
import json
import os
import sys
import time

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from pyscf import dft, gto
from pyscf.geomopt import geometric_solver

HOME = '/home/energy/s242862'
OUTDIR = f'{HOME}/bs_irc'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
BOHR = 0.529177210903
STEP = 0.35          # displacement along the normalised mass-weighted mode, A
MAXSTEPS = 200

BS = {'dm': None, 'in_newton': False, 'step': 0, 'log': [], 'on_step': None}


class BSUKS(dft.uks.UKS):
    def kernel(self, dm0=None, **kwargs):
        if BS['in_newton']:
            return super().kernel(dm0=dm0, **kwargs)
        BS['in_newton'] = True
        try:
            n = self.newton()
            n.max_cycle = 200
            n.conv_tol = 1e-9
            n.kernel(dm0=BS['dm'])
            self.e_tot, self.mo_coeff = n.e_tot, n.mo_coeff
            self.mo_occ, self.mo_energy = n.mo_occ, n.mo_energy
            self.converged = n.converged
            BS['dm'] = n.make_rdm1()
            BS['step'] += 1
            s2 = float(n.spin_square()[0])
            BS['log'].append({'step': BS['step'], 'e': float(n.e_tot),
                              's2': round(s2, 6)})
            if BS['step'] % 10 == 1:
                print(f'    Schritt {BS["step"]:3d}  E={n.e_tot:.8f}  '
                      f'S2={s2:.4f}', flush=True)
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


def ts_and_hess(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{HOME}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    ts = f
                    for fd in ('bs_freq', 'bs_freq_v2', 'bs_freq_fromneb'):
                        hp = f'{HOME}/{fd}/{rx}/hessian.npy'
                        if os.path.exists(hp):
                            return ts, hp
                    return ts, None
    return None, None


def build(sym, xyz, mem):
    mol = gto.Mole()
    mol.atom = '\n'.join(f'{s} {x:.8f} {y:.8f} {z:.8f}'
                         for s, (x, y, z) in zip(sym, xyz))
    mol.basis = BASIS; mol.charge = 0; mol.spin = 0
    mol.verbose = 4; mol.max_memory = mem
    mol.build()
    return mol


def seed_bs(mol, mem):
    """RKS -> external stability -> Route 1."""
    mf = dft.RKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
    mf.kernel()
    if not mf.converged:
        return None
    _, mo_ext, _, ext_stable = mf.stability(internal=True, external=True,
                                            return_status=True)
    if ext_stable:
        return ('RKS', mf)          # no broken solution here; RKS is correct
    mf_s = mf.to_uks()
    mf_s.xc = XC; mf_s.grids.level = 3
    mf_s.max_cycle = 300; mf_s.conv_tol = 1e-10; mf_s.max_memory = mem
    n = mf_s.newton(); n.max_cycle = 200; n.conv_tol = 1e-10
    n.kernel(mf_s.make_rdm1(mo_ext, mf_s.mo_occ))
    return ('BS', n) if n.converged else None


def main(rx):
    t0 = time.time()
    mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))
    out = f'{OUTDIR}/{rx}'
    os.makedirs(out, exist_ok=True)
    res = {'rxn': rx, 'step_A': STEP,
           'method': f'{XC}/{BASIS} BS-UKS, Auslenkung entlang der imaginaeren '
                     f'Mode + Relaxation (kein strenger IRC)'}

    ts, hp = ts_and_hess(rx)
    if not (ts and hp):
        res['status'] = 'FEHLENDE_EINGABE'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        print('TS oder Hessian fehlt'); return 1

    sym, x0 = read_xyz(ts)
    hess = np.load(hp)
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    w = np.repeat(1.0 / np.sqrt(m), 3)
    ev, vec = np.linalg.eigh(hess * w[:, None] * w[None, :])
    k = int(np.argmin(ev))
    res['lowest_eigval'] = float(ev[k])
    q = (vec[:, k] * w).reshape(-1, 3)      # back to Cartesian
    q = q / np.linalg.norm(q)
    print(f'{rx}: {len(sym)} Atome, niedrigster Eigenwert {ev[k]:.6f}',
          flush=True)

    ends = {}
    for sign, name in ((+1, 'vorwaerts'), (-1, 'rueckwaerts')):
        print(f'  --- {name} ---', flush=True)
        BS.update({'dm': None, 'in_newton': False, 'step': 0, 'log': []})
        x = x0 + sign * STEP * q
        mol = build(sym, x, mem)
        seed = seed_bs(mol, mem)
        if seed is None:
            ends[name] = {'status': 'SEED_FEHLGESCHLAGEN'}
            continue
        kind, mf0 = seed
        BS['dm'] = mf0.make_rdm1() if kind == 'BS' else None
        if kind == 'RKS':
            mf = dft.RKS(mol)
            mf.xc = XC; mf.grids.level = 3
            mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
        else:
            mf = BSUKS(mol)
            mf.xc = XC; mf.grids.level = 3
            mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
            mf.mo_coeff, mf.mo_occ = mf0.mo_coeff, mf0.mo_occ
            mf.mo_energy, mf.e_tot, mf.converged = mf0.mo_energy, float(mf0.e_tot), True
        try:
            conv, mol_end = geometric_solver.kernel(mf, maxsteps=MAXSTEPS)
        except Exception as exc:
            ends[name] = {'status': f'FEHLER: {type(exc).__name__}: {exc}'}
            continue
        xe = mol_end.atom_coords() * BOHR
        d = {'status': 'converged' if conv else 'nicht konvergiert',
             'start_kind': kind, 'n_steps': BS['step'],
             's2_final': BS['log'][-1]['s2'] if BS['log'] else None,
             'e_final': BS['log'][-1]['e'] if BS['log'] else float(mf.e_tot)}
        for label, path in (('reactant', f'{HOME}/orca_neb_results/{rx}/reactant.xyz'),
                            ('product', f'{HOME}/orca_neb_results/{rx}/product.xyz'),
                            ('ts', ts)):
            if os.path.exists(path):
                d[f'rmsd_{label}'] = round(kabsch(xe, read_xyz(path)[1]), 4)
        ends[name] = d
        with open(f'{out}/end_{name}.xyz', 'w') as fh:
            fh.write(f'{len(sym)}\n{name}, E={d["e_final"]}, S2={d["s2_final"]}\n')
            for s, (a, b, c) in zip(sym, xe):
                fh.write(f'{s} {a:.8f} {b:.8f} {c:.8f}\n')
        print(f'  {name}: {d["status"]}, RMSD zu Edukt '
              f'{d.get("rmsd_reactant")}, zu Produkt {d.get("rmsd_product")}',
              flush=True)

    res['ends'] = ends
    fw, bw = ends.get('vorwaerts', {}), ends.get('rueckwaerts', {})
    if all('rmsd_reactant' in x for x in (fw, bw)):
        a = (fw['rmsd_reactant'], fw['rmsd_product'])
        b = (bw['rmsd_reactant'], bw['rmsd_product'])
        # one side should land on the reactant, the other on the product
        ok = ((a[0] < a[1] and b[1] < b[0]) or (a[1] < a[0] and b[0] < b[1]))
        res['verdict'] = ('verbindet Edukt und Produkt' if ok
                          else 'beide Seiten laufen zum selben Minimum')
    res['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
