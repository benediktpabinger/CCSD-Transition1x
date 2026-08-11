"""Intrinsic reaction coordinate on the broken-symmetry surface.

This replaces `bs_irc.py`, which was withdrawn. That script displaced along the
imaginary mode and then ran a *free* geometry optimisation. A free relaxation
minimises in all 3N directions at once, so within a few steps it has forgotten
which mode was displaced and the outcome depends only on which side of the ridge
the displaced point happened to fall. It produced proven false negatives --
rxn8832, rxn8837 and rxn7949 all reported "both sides reach the same minimum"
although an independent ORCA NEB locates the same saddle between the relaxed
endpoints, so those three demonstrably do connect reactant and product.

A real IRC does not relax. It follows the steepest-descent path in
mass-weighted coordinates with small fixed steps, which is the path the reaction
actually takes, and never leaves it.

  y      = M^(1/2) x                   mass-weighted coordinates
  g_y    = M^(-1/2) dE/dx              mass-weighted gradient
  y_(k+1) = y_k - s * g_y / |g_y|      fixed arc length per step

The first step is not a gradient step -- at a saddle the gradient vanishes -- but
a displacement along the imaginary eigenvector of the mass-weighted Hessian.

What matters here is not which minimum the path ends at, but the *trace*: the
two reactive bond lengths at every step. That answers the question that decided
rxn1147 and rxn7957 by hand -- is a bond that already sits at its normal length
on the descending branch of the path, meaning the structure lies past the
transition state.

Broken symmetry is carried by chaining the density matrix from one step to the
next, never the MO coefficients (those are orthonormal only with respect to the
parent geometry's overlap matrix). <S^2> is logged at every step; it falls
towards zero near the endpoints, and that is correct rather than a failure --
the closed-shell description becomes valid again past the Coulson-Fischer point.

Environment:
  IRC_RXN     reaction id, e.g. rxn1147
  IRC_SRC     'ours' (default) or a model name: UMA-S, UMA-M, eSEN
  IRC_STEP    arc length per step in sqrt(amu)*Bohr, default 0.15
  IRC_MAX     maximum steps per direction, default 80
  IRC_OUT     output directory, default $HOME/bs_irc2
"""
import glob
import json
import os
import sys
import time

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from pyscf import dft, gto

HOME = '/home/energy/s242862'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
BOHR = 0.529177210903

RXN = os.environ.get('IRC_RXN', '')
SRC = os.environ.get('IRC_SRC', 'ours')
STEP = float(os.environ.get('IRC_STEP', 0.15))     # sqrt(amu) * Bohr
MAXSTEPS = int(os.environ.get('IRC_MAX', 80))
OUTDIR = os.environ.get('IRC_OUT', f'{HOME}/bs_irc2')
KICK = 3.0 * STEP        # initial displacement along the imaginary mode
GRAD_CONV = 1.0e-4       # max |dE/dx| in Hartree/Bohr counting as a minimum

MODEL_DIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
             'eSEN': 'esen_neb_results'}


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


def reactive_bonds(rx):
    """The two pairs whose distance changes most between reactant and product."""
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{HOME}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def start_geometry(rx, src):
    """Geometry and Hessian for the point the path is launched from."""
    if src == 'ours':
        for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
            for pat in ('ts', 'final', 'opt'):
                for f in sorted(glob.glob(f'{HOME}/{d}/{rx}/*.xyz')):
                    if pat in os.path.basename(f).lower():
                        for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
                            hp = f'{HOME}/{fd}/{rx}/hessian.npy'
                            if os.path.exists(hp):
                                return f, hp
                        return f, None
        return None, None
    g = f'{HOME}/{MODEL_DIR[src]}/{rx}/transition_state.xyz'
    hp = f'{HOME}/freq_at_model/{rx}_{src}/hessian.npy'
    return (g if os.path.exists(g) else None,
            hp if os.path.exists(hp) else None)


def build(sym, xyz, mem):
    mol = gto.Mole()
    mol.atom = '\n'.join(f'{s} {x:.10f} {y:.10f} {z:.10f}'
                         for s, (x, y, z) in zip(sym, xyz))
    mol.basis = BASIS
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.max_memory = mem
    mol.build()
    return mol


def seed(mol, mem):
    """RKS, external stability, then Route 1 into the broken solution."""
    mf = dft.RKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
    mf.kernel()
    if not mf.converged:
        return None
    _, mo_ext, _, ext_stable = mf.stability(internal=True, external=True,
                                            return_status=True)
    if ext_stable:
        return 'RKS', mf.make_rdm1(), float(mf.e_tot), 0.0
    u = mf.to_uks()
    u.xc = XC; u.grids.level = 3
    u.max_memory = mem
    n = u.newton(); n.max_cycle = 200; n.conv_tol = 1e-10
    n.kernel(u.make_rdm1(mo_ext, u.mo_occ))
    if not n.converged:
        return None
    return 'BS', n.make_rdm1(), float(n.e_tot), float(n.spin_square()[0])


def energy_gradient(sym, xyz, kind, dm0, mem):
    """One SCF plus one analytic gradient, seeded with the previous density."""
    mol = build(sym, xyz, mem)
    if kind == 'RKS':
        mf = dft.RKS(mol)
        mf.xc = XC; mf.grids.level = 3
        mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
        mf.kernel(dm0=dm0)
        if not mf.converged:
            return None
        s2 = 0.0
        dm = mf.make_rdm1()
        g = mf.nuc_grad_method().kernel()
        return float(mf.e_tot), g, dm, s2
    u = dft.UKS(mol)
    u.xc = XC; u.grids.level = 3
    u.max_memory = mem
    n = u.newton(); n.max_cycle = 200; n.conv_tol = 1e-10
    n.kernel(dm0=dm0)
    if not n.converged:
        return None
    u.mo_coeff, u.mo_occ, u.mo_energy = n.mo_coeff, n.mo_occ, n.mo_energy
    u.e_tot, u.converged = n.e_tot, True
    g = u.nuc_grad_method().kernel()
    return float(n.e_tot), g, n.make_rdm1(), float(n.spin_square()[0])


def projector(sym, xyz_bohr, msqrt):
    """Basis of translations and rotations in mass-weighted coordinates.

    Projecting these out keeps the path from drifting or spinning, which would
    otherwise consume arc length without changing the molecule at all.
    """
    nat = len(sym)
    c = xyz_bohr - (xyz_bohr * msqrt[:, None] ** 2).sum(0) / (msqrt ** 2).sum()
    B = []
    for k in range(3):
        v = np.zeros((nat, 3)); v[:, k] = msqrt
        B.append(v.ravel())
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        v = np.cross(np.tile(e, (nat, 1)), c) * msqrt[:, None]
        B.append(v.ravel())
    B = np.array(B)
    # orthonormalise, dropping the null directions of a linear molecule
    U, s, _ = np.linalg.svd(B.T, full_matrices=False)
    return U[:, s > 1e-8]


def bond_lengths(xyz, pairs):
    return [round(float(np.linalg.norm(xyz[a] - xyz[b])), 4)
            for a, b, _ in pairs]


def run_branch(sym, x0_bohr, q_mw, sign, msqrt, pairs, kind, dm_ts, mem,
               refs, name):
    """One half of the path: kick along the mode, then steepest descent."""
    print(f'  --- {name} ---', flush=True)
    y = x0_bohr.ravel() * msqrt.repeat(3) + sign * KICK * q_mw
    dm = dm_ts
    trace = []
    arc = 0.0
    step = STEP
    gprev = None
    status = 'max steps'
    for k in range(MAXSTEPS):
        x = (y / msqrt.repeat(3)).reshape(-1, 3)
        r = energy_gradient(sym, x * BOHR, kind, dm, mem)
        if r is None:
            status = f'SCF failed at step {k}'
            break
        e, g, dm, s2 = r
        P = projector(sym, x, msqrt)
        gy = g.ravel() / msqrt.repeat(3)
        gy -= P @ (P.T @ gy)
        gn = float(np.linalg.norm(gy))
        row = {'k': k, 'arc': round(arc, 3), 'e': e, 's2': round(s2, 4),
               'gmax': round(float(np.abs(g).max()), 6),
               'gnorm_mw': round(gn, 6),
               'd': bond_lengths(x * BOHR, pairs)}
        for lab, xr in refs.items():
            row[f'rmsd_{lab}'] = round(kabsch(x * BOHR, xr), 4)
        trace.append(row)
        if k % 5 == 0 or k < 3:
            print(f'    {k:3d}  arc {arc:6.2f}  E {e:.8f}  S2 {s2:.3f}  '
                  f'|g| {row["gmax"]:.5f}  d {row["d"]}', flush=True)
        if row['gmax'] < GRAD_CONV:
            status = 'reached a minimum'
            break
        if gn < 1e-8:
            status = 'gradient vanished'
            break
        d = -gy / gn
        # a reversal means the step overshot the valley floor; shorten it
        if gprev is not None and float(np.dot(d, gprev)) < 0.0:
            step = max(step * 0.5, 0.02)
        gprev = d
        y = y + step * d
        arc += step
    return {'status': status, 'n_steps': len(trace), 'arc': round(arc, 3),
            'trace': trace}


def main():
    t0 = time.time()
    mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))
    tag = RXN if SRC == 'ours' else f'{RXN}_{SRC}'
    out = f'{OUTDIR}/{tag}'
    os.makedirs(out, exist_ok=True)
    res = {'rxn': RXN, 'source': SRC, 'step_sqrtamu_bohr': STEP,
           'kick': KICK, 'max_steps': MAXSTEPS,
           'method': f'{XC}/{BASIS}, mass-weighted steepest descent IRC, '
                     f'density-matrix chaining, translations and rotations '
                     f'projected out'}

    geom, hp = start_geometry(RXN, SRC)
    pairs = reactive_bonds(RXN)
    if not (geom and hp and pairs):
        res['status'] = 'missing input'
        res['have'] = {'geom': geom, 'hess': hp, 'pairs': len(pairs)}
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        print('missing input:', res['have']); return 1

    sym, x0_ang = read_xyz(geom)
    x0 = x0_ang / BOHR
    hess = np.load(hp)
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    msqrt = np.sqrt(m)
    w = np.repeat(1.0 / msqrt, 3)
    ev, vec = np.linalg.eigh(hess * w[:, None] * w[None, :])
    k = int(np.argmin(ev))
    q_mw = vec[:, k] / np.linalg.norm(vec[:, k])   # already mass-weighted
    res['lowest_eigval'] = float(ev[k])
    res['reactive_bonds'] = [nm for _, _, nm in pairs]
    res['d_start'] = bond_lengths(x0_ang, pairs)
    print(f'{RXN} [{SRC}]  {len(sym)} atoms  lowest eigenvalue {ev[k]:.6f}',
          flush=True)
    print(f'  reactive bonds {res["reactive_bonds"]} at {res["d_start"]} A',
          flush=True)

    refs = {}
    for lab in ('reactant', 'product'):
        p = f'{HOME}/orca_neb_results/{RXN}/{lab}.xyz'
        if os.path.exists(p):
            refs[lab] = read_xyz(p)[1]

    s = seed(build(sym, x0_ang, mem), mem)
    if s is None:
        res['status'] = 'seeding failed'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        print('seeding failed'); return 1
    kind, dm_ts, e_ts, s2_ts = s
    res.update({'surface': kind, 'e_ts': e_ts, 's2_ts': round(s2_ts, 4)})
    print(f'  start point: {kind}, E {e_ts:.8f}, S2 {s2_ts:.4f}', flush=True)

    branches = {}
    for sign, name in ((+1, 'forward'), (-1, 'backward')):
        branches[name] = run_branch(sym, x0, q_mw, sign, msqrt, pairs, kind,
                                    dm_ts, mem, refs, name)
        tr = branches[name]['trace']
        if tr:
            with open(f'{out}/path_{name}.json', 'w') as fh:
                json.dump(tr, fh, indent=1)
            last = tr[-1]
            print(f'  {name}: {branches[name]["status"]} after '
                  f'{len(tr)} steps, bonds {last["d"]}, '
                  f'RMSD reactant {last.get("rmsd_reactant")} '
                  f'product {last.get("rmsd_product")}', flush=True)

    res['branches'] = {k: {kk: vv for kk, vv in v.items() if kk != 'trace'}
                       for k, v in branches.items()}
    fw = branches['forward']['trace']
    bw = branches['backward']['trace']
    if fw and bw and refs:
        a, b = fw[-1], bw[-1]
        ta = 'reactant' if a.get('rmsd_reactant', 9) < a.get('rmsd_product', 9) else 'product'
        tb = 'reactant' if b.get('rmsd_reactant', 9) < b.get('rmsd_product', 9) else 'product'
        res['ends'] = {'forward': ta, 'backward': tb}
        res['connects'] = (ta != tb)
        # the trace is the point of the exercise: how each reactive bond moves
        res['bond_trace'] = {
            'start': res['d_start'],
            'forward_end': a['d'], 'backward_end': b['d'],
            'monotonic': [bool((np.diff([r['d'][i] for r in fw]) >= -1e-3).all()
                               or (np.diff([r['d'][i] for r in fw]) <= 1e-3).all())
                          for i in range(len(pairs))]}
    res['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != 'branches'},
                     indent=1), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
