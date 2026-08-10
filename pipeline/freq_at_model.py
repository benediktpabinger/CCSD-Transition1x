"""Is the model's predicted geometry itself a transition state?

Three reactions are undecided. Their model geometries are nearly stationary --
gradients of 0.05 to 0.14 eV/A, comparable to the reference itself -- yet 0.37
to 0.60 A away from the saddle we found. Both can only be true if another
stationary point sits there. Whether that point is a transition state is what
this settles.

The surface is chosen by the stability analysis rather than assumed: where the
RKS solution is externally unstable the Hessian is built on the broken-symmetry
solution, where it is stable RKS is the ground state and is used directly. For
rxn1147 all three model geometries are externally stable, so those run as RKS.

Reported per structure:
  gradient      how stationary the point actually is
  n_imag        one imaginary frequency means a transition state
  mode test     whether that mode stretches the reactive bonds, since a
                hindered rotation is also a first-order saddle
  energy        against our own saddle. If both are transition states the lower
                one carries the reaction, so this decides which is relevant.

Usage: python freq_at_model.py <rxn> <model>
"""
import glob
import json
import os
import sys
import time

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from pyscf import dft, gto, lib
from pyscf.hessian import thermo

H = '/home/energy/s242862'
OUTDIR = f'{H}/freq_at_model'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
DELTA = 0.01
HA_EVANG = 27.211386245988 / 0.529177210903
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def build(sym, coords_bohr, mem, logf=None):
    mol = gto.Mole()
    mol.atom = [(s, tuple(c)) for s, c in zip(sym, coords_bohr)]
    mol.unit = 'Bohr'; mol.basis = BASIS
    mol.charge = 0; mol.spin = 0; mol.max_memory = mem
    mol.verbose = 4 if logf else 0
    if logf:
        mol.output = logf                 # before build(), or nothing is logged
    mol.build()
    return mol


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def our_ts_energy(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if j.get('e_uks_final') is not None:
                return j['e_uks_final']
    return None


def converge(mol, mem):
    """RKS, then follow the external instability if there is one."""
    mf = dft.RKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
    mf.kernel()
    if not mf.converged:
        return None, None, 'RKS nicht konvergiert'
    _, mo_ext, _, ext_stable = mf.stability(internal=True, external=True,
                                            return_status=True)
    if ext_stable:
        return mf, 'RKS', None
    mf_u = mf.to_uks()
    mf_u.xc = XC; mf_u.grids.level = 3
    mf_u.max_cycle = 300; mf_u.conv_tol = 1e-10; mf_u.max_memory = mem
    n = mf_u.newton(); n.max_cycle = 200; n.conv_tol = 1e-10
    n.kernel(mf_u.make_rdm1(mo_ext, mf_u.mo_occ))
    if not n.converged:
        return None, None, 'BS nicht konvergiert'
    return n, 'BS', None


def main(rx, model):
    t0 = time.time()
    mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))
    tag = f'{rx}_{model.replace("+", "")}'
    out = f'{OUTDIR}/{tag}'
    os.makedirs(out, exist_ok=True)
    res = {'rxn': rx, 'model': model,
           'level': f'{XC}/{BASIS}, numerische Hesse (delta={DELTA} Bohr)'}

    p = f'{H}/{MODELDIR[model]}/{rx}/transition_state.xyz'
    if not os.path.exists(p):
        res['error'] = 'Modellgeometrie fehlt'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=1); return 1
    sym, xyz_ang = read_xyz(p)
    natm = len(sym)
    print(f'{tag}: {natm} Atome', flush=True)

    mol0 = build(sym, xyz_ang / lib.param.BOHR, mem, f'{out}/pyscf_ref.log')
    mf0, kind, err = converge(mol0, mem)
    if err:
        res['error'] = err
        json.dump(res, open(f'{out}/result.json', 'w'), indent=1)
        print('ABBRUCH:', err); return 1
    res['surface'] = kind
    res['e_ref'] = round(float(mf0.e_tot), 10)
    if kind == 'BS':
        res['s2_ref'] = round(float(mf0.spin_square()[0]), 6)
    g0 = mf0.nuc_grad_method().kernel()
    res['grad_max_evang'] = round(float(np.abs(g0).max()) * HA_EVANG, 6)
    dm0 = mf0.make_rdm1()
    print(f'  {kind}: E={mf0.e_tot:.8f}  max|g|={res["grad_max_evang"]:.4f} eV/A',
          flush=True)

    coords0 = mol0.atom_coords()
    nc = 3 * natm
    hess = np.zeros((nc, nc))
    s2_log, bad = [], 0
    for i in range(nc):
        a, c = divmod(i, 3)
        gs = {}
        for sgn in (+1, -1):
            co = coords0.copy()
            co[a, c] += sgn * DELTA
            mol_d = build(sym, co, mem)
            if kind == 'BS':
                mf_d = dft.UKS(mol_d)
            else:
                mf_d = dft.RKS(mol_d)
            mf_d.xc = XC; mf_d.grids.level = 3
            mf_d.max_cycle = 300; mf_d.conv_tol = 1e-10; mf_d.max_memory = mem
            nd = mf_d.newton(); nd.max_cycle = 200; nd.conv_tol = 1e-10
            nd.kernel(dm0)
            if kind == 'BS':
                s2 = float(nd.spin_square()[0])
                s2_log.append(s2)
                if s2 < 0.5 * res.get('s2_ref', 1):
                    bad += 1
            if not nd.converged:
                bad += 1
            gs[sgn] = nd.nuc_grad_method().kernel().reshape(-1)
        hess[i] = (gs[+1] - gs[-1]) / (2 * DELTA)
        if i % 6 == 0:
            print(f'  Koordinate {i+1}/{nc}  {time.time()-t0:.0f}s', flush=True)

    hess = 0.5 * (hess + hess.T)
    h4 = hess.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    an = thermo.harmonic_analysis(mol0, h4)
    fr = np.asarray(an['freq_wavenumber'])
    imag = fr[np.iscomplex(fr)] if np.iscomplexobj(fr) else fr[fr < 0]
    res['n_imag'] = int(len(imag))
    res['imag_freq'] = [round(float(np.imag(x) if np.iscomplex(x) else x), 2)
                        for x in imag]
    res['freq_lowest_6'] = [round(float(x), 2) for x in np.sort(np.real(fr))[:6]]
    if s2_log:
        res['s2_min'], res['s2_max'] = round(min(s2_log), 6), round(max(s2_log), 6)
    res['n_problems'] = bad

    # does the imaginary mode move the reactive bonds?
    pairs = reactive(rx)
    if res['n_imag'] == 1 and pairs:
        m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
        w = np.repeat(1.0 / np.sqrt(m), 3)
        ev, vec = np.linalg.eigh(hess * w[:, None] * w[None, :])
        q = vec[:, int(np.argmin(ev))].reshape(-1, 3)
        q = q / np.linalg.norm(q)
        idx = sorted({i for a, b, _ in pairs for i in (a, b)})
        res['mode_fraction'] = round(float((q[idx] ** 2).sum()), 4)
        res['mode_rates'] = [
            [nm, round(float(np.dot(q[a] - q[b],
                                    (xyz_ang[a] - xyz_ang[b])
                                    / np.linalg.norm(xyz_ang[a] - xyz_ang[b]))), 4)]
            for a, b, nm in pairs]

    ours = our_ts_energy(rx)
    if ours is not None:
        d = (float(mf0.e_tot) - ours) * 27211.386
        res['e_vs_our_ts_meV'] = round(d, 2)
        res['lower'] = 'Modellgeometrie' if d < 0 else 'unser Sattelpunkt'

    res['verdict'] = ('Uebergangszustand' if res['n_imag'] == 1 else
                      'Minimum' if res['n_imag'] == 0 else
                      f'Sattel {res["n_imag"]}. Ordnung')
    res['elapsed_s'] = round(time.time() - t0, 1)
    np.save(f'{out}/hessian.npy', hess)
    json.dump(res, open(f'{out}/result.json', 'w'), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
