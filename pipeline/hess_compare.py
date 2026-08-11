"""Compare the ORCA and PySCF Hessians at the same geometry.

Two reasons this matters more than a cross-check usually would.

The first is the broken symmetry. ORCA writes a reduced log for NumFreq -- the
SCF output of each displaced point goes to numfreq.lastscf and is overwritten,
so the <S**2> trace over all 6N displacements cannot be recovered. That trace
was the planned test of whether ORCA holds the broken solution across the
displacements, which is exactly what `BrokenSym` failed to do along a NEB band.
Agreement between the two Hessians is the stronger substitute: a displacement
that collapsed to the closed-shell solution would corrupt its column of the
Hessian, and no such corruption survives a full spectral comparison.

The second is that the PySCF Hessians have never been checked against anything.
Every stage-2 verdict (is it a saddle) and every stage-3 verdict (does the mode
belong to this reaction) in the working document rests on them.

Both matrices are read in Eh/Bohr^2 with (atom, xyz) ordering and put through
the same mass-weighting, the same projection of translations and rotations, and
the same diagonalisation here, so the comparison is not contaminated by two
codes' differing conventions for reporting frequencies.

Usage: python hess_compare.py <rxn> <ours|UMA-S|UMA-M|eSEN>
"""
import glob
import json
import os
import sys

import numpy as np
from ase.data import atomic_masses, atomic_numbers

H = '/home/energy/s242862'
# Eh/(Bohr^2 amu) -> cm^-1
CM = 5140.4871
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


def read_orca_hess(path):
    """The $hessian block of an ORCA .hess file, as a dense (3N, 3N) array."""
    lines = open(path).read().split('\n')
    i = next(k for k, l in enumerate(lines) if l.strip() == '$hessian')
    n = int(lines[i + 1].split()[0])
    Hm = np.zeros((n, n))
    k = i + 2
    cols = []
    while True:
        t = lines[k].split()
        k += 1
        if not t:
            continue
        if all(x.lstrip('-').isdigit() for x in t) and len(t) <= 8:
            cols = [int(x) for x in t]          # column header
            continue
        r = int(t[0])
        for c, v in zip(cols, t[1:]):
            Hm[r, c] = float(v)
        if r == n - 1 and cols and cols[-1] == n - 1:
            break
    return Hm


def trans_rot(sym, xyz_bohr, msqrt):
    """Mass-weighted translations and rotations, orthonormalised.

    Without projecting these out the six near-zero modes come back as numerical
    noise scattered between roughly -30 and +30 cm^-1, and the negative ones
    then read as imaginary frequencies. ORCA projects them and prints six exact
    zeros; doing the same here is what makes the two spectra comparable.
    """
    nat = len(sym)
    w2 = msqrt ** 2
    c = xyz_bohr - (xyz_bohr * w2[:, None]).sum(0) / w2.sum()
    B = []
    for k in range(3):
        v = np.zeros((nat, 3)); v[:, k] = msqrt
        B.append(v.ravel())
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        B.append((np.cross(np.tile(e, (nat, 1)), c) * msqrt[:, None]).ravel())
    U, s, _ = np.linalg.svd(np.array(B).T, full_matrices=False)
    return U[:, s > 1e-8]


def analyse(hess, sym, xyz, pairs):
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    msqrt = np.sqrt(m)
    w = np.repeat(1.0 / msqrt, 3)
    Hm = hess * w[:, None] * w[None, :]
    P = trans_rot(sym, xyz / 0.529177210903, msqrt)
    Q = np.eye(len(Hm)) - P @ P.T
    ev, vec = np.linalg.eigh(Q @ Hm @ Q)
    freqs = np.sign(ev) * np.sqrt(np.abs(ev)) * CM
    k = int(np.argmin(ev))
    q = vec[:, k]
    qa = q.reshape(-1, 3)
    qa = qa / np.linalg.norm(qa)
    idx = sorted({i for a, b, _ in pairs for i in (a, b)})
    bonds = []
    for a, b, nm in pairs:
        u = (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b])
        bonds.append((nm, float(np.dot(qa[a] - qa[b], u)),
                      float(np.linalg.norm(xyz[a] - xyz[b]))))
    return {'freqs': freqs, 'vec': q, 'frac': float((qa[idx] ** 2).sum()),
            'bonds': bonds, 'imag': float(freqs[k])}


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def pyscf_inputs(rx, src):
    if src == 'ours':
        for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
            for f in sorted(glob.glob(f'{H}/{d}/{rx}/*.xyz')):
                if any(p in os.path.basename(f).lower()
                       for p in ('ts', 'final', 'opt')):
                    for fd in ('bs_freq_fromneb', 'bs_freq_v2', 'bs_freq'):
                        hp = f'{H}/{fd}/{rx}/hessian.npy'
                        if os.path.exists(hp):
                            return f, hp
        return None, None
    return (f'{H}/{MODEL_DIR[src]}/{rx}/transition_state.xyz',
            f'{H}/freq_at_model/{rx}_{src}/hessian.npy')


def main(rx, src):
    tag = rx if src == 'ours' else f'{rx}_{src}'
    oh = f'{H}/orca_irc/{rx}_{src}/numfreq.hess'
    geom, ph = pyscf_inputs(rx, src)
    if not (os.path.exists(oh) and geom and ph and os.path.exists(ph)):
        print(f'{tag}: missing  orca={os.path.exists(oh)} '
              f'pyscf={ph and os.path.exists(ph)}')
        return 1
    sym, xyz = read_xyz(geom)
    pairs = reactive(rx)
    O = analyse(read_orca_hess(oh), sym, xyz, pairs)
    P = analyse(np.load(ph), sym, xyz, pairs)

    print(f'=== {rx}  [{src}]   {len(sym)} atoms')
    print(f'    reactive bonds: ' + ', '.join(nm for _, _, nm in pairs))
    print()
    print('    mode        ORCA        PySCF        diff')
    nf = len(O['freqs'])
    for i in range(min(nf, 18)):
        a, b = O['freqs'][i], P['freqs'][i]
        flag = ''
        if abs(a) > 1.0 and abs(b) > 1.0:
            flag = f'{a - b:+9.2f}'
        print(f'    {i:4d}  {a:10.2f}   {b:10.2f}   {flag}')
    real = [(a, b) for a, b in zip(O['freqs'], P['freqs'])
            if abs(a) > 20 and abs(b) > 20]
    if real:
        d = np.array([a - b for a, b in real])
        print(f'\n    over {len(real)} modes above 20 cm-1: '
              f'mean {d.mean():+.2f}, max |diff| {np.abs(d).max():.2f} cm-1')
    print(f'\n    imaginary frequency   ORCA {O["imag"]:9.2f}   '
          f'PySCF {P["imag"]:9.2f}   diff {O["imag"] - P["imag"]:+.2f} cm-1')
    ov = abs(float(np.dot(O['vec'], P['vec'])))
    print(f'    overlap of the two imaginary modes:  {ov:.6f}')
    print(f'\n    stage 3           ORCA     PySCF')
    print(f'    mode fraction   {O["frac"]:7.3f}   {P["frac"]:7.3f}')
    for (nm, ro, d0), (_, rp, _) in zip(O['bonds'], P['bonds']):
        print(f'    {nm:<12} {ro:+7.3f}   {rp:+7.3f}    at {d0:.3f} A')
    n_imag_o = int((O['freqs'] < -20).sum())
    n_imag_p = int((P['freqs'] < -20).sum())
    print(f'\n    imaginary modes below -20 cm-1:  ORCA {n_imag_o}   '
          f'PySCF {n_imag_p}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
