"""Numerical BS-UKS frequencies at the optimised broken-symmetry TS geometries.

Question: is the structure a genuine first-order saddle point -- exactly one
imaginary frequency, and does its normal mode move the reactive bonds?

wB97M-V carries VV10 non-local correlation, for which neither PySCF nor
ORCA 5.0.4 provides an analytic Hessian.  The Hessian is therefore built by
central differences of analytic gradients: 6N gradient evaluations, i.e. 66 for
an 11-atom molecule at roughly 90 s each.

The risk in doing it this way is that the SCF at a displaced geometry falls back
onto the restricted solution, which would corrupt the Hessian silently.  Each
displaced point is therefore seeded with the density matrix of the reference BS
solution -- a density matrix, NOT stale MO coefficients, which are orthonormal
only with respect to the overlap at the parent geometry -- and <S^2> is recorded
at every point.  A collapse shows up as S2_min far below the reference value.

Usage: python bs_freq.py <rxn>
"""
import glob
import json
import os
import sys
import time

import numpy as np
from pyscf import dft, gto, lib
from pyscf.hessian import thermo

H = '/home/energy/s242862'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
DELTA = 0.01          # Bohr; central differences
S2_DROP = 0.5         # fraction of the reference <S^2> that counts as collapse


def bs_ts_path(rxn):
    c = glob.glob(f'{H}/bs_tsopt_batch/{rxn}/*.xyz')
    for pat in ('ts', 'final', 'opt'):
        for f in c:
            if pat in os.path.basename(f).lower():
                return f
    return c[0] if c else None


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0])
        xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def build(sym, coords_bohr, mem, logf=None):
    mol = gto.Mole()
    mol.atom = [(s, tuple(c)) for s, c in zip(sym, coords_bohr)]
    mol.unit = 'Bohr'
    mol.basis = BASIS
    mol.charge = 0
    mol.spin = 0
    mol.max_memory = mem
    mol.verbose = 4 if logf else 0
    if logf:
        mol.output = logf          # must be set BEFORE build()
    mol.build()
    return mol


def uks(mol, mem):
    mf = dft.UKS(mol)
    mf.xc = XC
    mf.grids.level = 3
    mf.max_cycle = 300
    mf.conv_tol = 1e-10
    mf.max_memory = mem
    return mf


def reference_bs(sym, xyz_ang, mem, outdir):
    """RKS -> external stability -> Route 1, as in stability_pipeline.py."""
    logf = os.path.join(outdir, 'pyscf_ref.log')
    mol = build(sym, xyz_ang / lib.param.BOHR, mem, logf)
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = 3
    mf.max_cycle = 300
    mf.conv_tol = 1e-10
    mf.max_memory = mem
    mf.kernel()
    if not mf.converged:
        return None, None, 'RKS nicht konvergiert'
    _, mo_ext, _, ext_stable = mf.stability(internal=True, external=True,
                                            return_status=True)
    if ext_stable:
        return None, None, 'RKS extern stabil - keine BS-Loesung an dieser Geometrie'
    mf_u = mf.to_uks()               # to_uks(), not a fresh UKS: mo_occ
    mf_u.xc = XC
    mf_u.grids.level = 3
    mf_u.max_cycle = 300
    mf_u.conv_tol = 1e-10
    mf_u.max_memory = mem
    n = mf_u.newton()
    n.max_cycle = 200
    n.conv_tol = 1e-10
    n.kernel(mf_u.make_rdm1(mo_ext, mf_u.mo_occ))
    if not n.converged:
        return None, None, 'BS nicht konvergiert'
    return n, mol, None


def main(rxn):
    t0 = time.time()
    mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))
    outdir = f'{H}/bs_freq/{rxn}'
    os.makedirs(outdir, exist_ok=True)

    p = bs_ts_path(rxn)
    if not p:
        json.dump({'rxn': rxn, 'error': 'keine BS-TS-Geometrie'},
                  open(f'{outdir}/result.json', 'w'), indent=1)
        print('keine Geometrie'); return
    sym, xyz_ang = read_xyz(p)
    natm = len(sym)
    print(f'{rxn}: {natm} Atome, {p}', flush=True)

    n_ref, mol_ref, err = reference_bs(sym, xyz_ang, mem, outdir)
    if err:
        json.dump({'rxn': rxn, 'error': err},
                  open(f'{outdir}/result.json', 'w'), indent=1)
        print('ABBRUCH:', err); return

    s2_ref = float(n_ref.spin_square()[0])
    e_ref = float(n_ref.e_tot)
    dm0 = n_ref.make_rdm1()
    g_ref = n_ref.nuc_grad_method().kernel()
    print(f'  Referenz: E={e_ref:.8f}  <S^2>={s2_ref:.4f}  '
          f'max|g|={np.abs(g_ref).max():.6f} Ha/Bohr', flush=True)

    coords0 = mol_ref.atom_coords()          # Bohr
    nc = 3 * natm
    hess = np.zeros((nc, nc))
    s2_log, failed = [], []

    for i in range(nc):
        a, c = divmod(i, 3)
        gs = {}
        for sgn in (+1, -1):
            co = coords0.copy()
            co[a, c] += sgn * DELTA
            mol_d = build(sym, co, mem)
            mf_d = uks(mol_d, mem)
            nd = mf_d.newton()
            nd.max_cycle = 200
            nd.conv_tol = 1e-10
            nd.kernel(dm0)                    # density matrix, not MO coeffs
            s2 = float(nd.spin_square()[0])
            s2_log.append(s2)
            if not nd.converged:
                failed.append((i, sgn, 'nicht konvergiert'))
            if s2 < S2_DROP * s2_ref:
                failed.append((i, sgn, f'S2={s2:.4f} << {s2_ref:.4f}'))
            gs[sgn] = nd.nuc_grad_method().kernel().reshape(-1)
        hess[i] = (gs[+1] - gs[-1]) / (2 * DELTA)
        if i % 6 == 0:
            print(f'  Koordinate {i+1}/{nc}  <S^2>={s2_log[-1]:.4f}  '
                  f'{time.time()-t0:.0f}s', flush=True)

    hess = 0.5 * (hess + hess.T)             # symmetrise
    h4 = hess.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    res = thermo.harmonic_analysis(mol_ref, h4)
    freq = np.asarray(res['freq_wavenumber'])
    imag = freq[np.iscomplex(freq)] if np.iscomplexobj(freq) else freq[freq < 0]
    n_imag = int(len(imag))
    freq_real = np.real(freq)

    out = {
        'rxn': rxn, 'natm': natm, 'geometry': p,
        'level': f'{XC}/{BASIS} BS-UKS, numerische Hesse (delta={DELTA} Bohr)',
        'e_bs': round(e_ref, 10), 's2_ref': round(s2_ref, 6),
        'max_grad_ha_bohr': round(float(np.abs(g_ref).max()), 8),
        'n_imag': n_imag,
        'freq_lowest_6': [round(float(x), 2) for x in np.sort(freq_real)[:6]],
        'imag_freq': [round(float(np.imag(x) if np.iscomplex(x) else x), 2)
                      for x in imag],
        's2_min': round(float(min(s2_log)), 6),
        's2_max': round(float(max(s2_log)), 6),
        'n_displacement_problems': len(failed),
        'displacement_problems': failed[:20],
        'verdict': ('echter TS' if n_imag == 1 else
                    'MINIMUM (kein TS)' if n_imag == 0 else
                    f'Sattel {n_imag}. Ordnung'),
        'elapsed_s': round(time.time() - t0, 1),
    }
    np.save(f'{outdir}/hessian.npy', hess)
    json.dump(out, open(f'{outdir}/result.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k != 'displacement_problems'}, indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
