"""What are the structures where repeated or continued NEBs ended?

Two checks on the model's own surface, in double precision (decision D21):

  Hessian   central finite differences of the forces at each structure, step
            0.005 A. Translations and rotations are projected out. A true
            transition state has exactly one negative eigenvalue; two or more
            mean a ridge or a higher-order saddle, which would explain a band
            that slowly slides away after meeting the 0.05 eV/A criterion.

  Line scan energy and forces on the straight line between two structures A
            and B, t from -0.2 to 1.2. dE/dt from the energies is compared with
            -F.(B-A) from the forces. On a smooth surface they agree; a large
            gap or a jump means a kink or discontinuity in the model.

Each case names two stored structures of one model and one reaction. The energy
recomputed here is compared with the stored one, because a difference there would
point to the device and not to the surface.

Output  results/ts_hessian.csv, results/ts_linescan.csv, appended per model, so
one job can run per model.

    python pipeline/ts_hessian_scan.py --model esen --device cpu
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from ase import Atoms
from ase.io import read

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omol_neb import make_calc_factory  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
T, P = f'{H}/neb_test', f'{H}/neb_prec'
CASES = [  # model, reaction, label, root of A, root of B
    ('uma-s', 'rxn4113', 'stop_shift', f'{T}/cpu3x8', f'{P}/tight_fromcpu'),
    ('esen', 'rxn4113', 'stop_shift', f'{T}/cpu3x8', f'{P}/tight_fromcpu'),
    ('mace-omol', 'rxn4113', 'stop_shift', f'{T}/cpu3x8', f'{P}/tight_fromcpu'),
    ('mace-omol', 'rxn4518', 'stop_shift', f'{T}/cpu3x8', f'{P}/tight_fromcpu'),
    ('uma-s', 'rxn0346', 'stop_shift', f'{T}/cpu3x8', f'{P}/tight_fromcpu'),
    ('esen', 'rxn0101', 'stop_shift_closed', f'{T}/cpu3x8', f'{P}/tight_fromcpu'),
    ('uma-m', 'rxn0346', 'repeat_gpu', f'{T}/gpu', f'{P}/rep_gpu'),
    ('esen', 'rxn0346', 'repeat_gpu', f'{T}/gpu', f'{P}/rep_gpu'),
    ('uma-s', 'rxn0000', 'dbl_cpu_gpu', f'{P}/dbl_cpu', f'{P}/dbl_gpu'),
]
STEP = 0.005
NEG_TOL = -0.05     # eV/A^2; softer negative values are treated as zero modes


def align(a, b):
    """Rotate and shift a onto b, proper rotation only."""
    ca, cb = a.mean(0), b.mean(0)
    U, S, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(U @ Vt))
    return (a - ca) @ (U @ np.diag([1, 1, d]) @ Vt) + cb


def project_rt(x):
    """Basis of the 3N-6 internal displacements at positions x."""
    c = x - x.mean(0)
    n = len(x)
    B = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = 1
        B.append(v.ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1
        B.append(np.cross(e, c).ravel())
    Q, _ = np.linalg.qr(np.array(B).T)
    return np.eye(3 * n) - Q @ Q.T


def forces(calc, Z, x):
    a = Atoms(numbers=Z, positions=x, info={'charge': 0, 'spin': 1})
    a.calc = calc
    return a.get_potential_energy(), a.get_forces().ravel().astype(float)


def hessian(calc, Z, x):
    n = x.size
    H_ = np.zeros((n, n))
    for i in range(n):
        d = np.zeros(n)
        d[i] = STEP
        H_[i] = -(forces(calc, Z, x + d.reshape(-1, 3))[1]
                  - forces(calc, Z, x - d.reshape(-1, 3))[1]) / (2 * STEP)
    H_ = 0.5 * (H_ + H_.T)
    Pr = project_rt(x)
    w, v = np.linalg.eigh(Pr @ H_ @ Pr)
    keep = np.argsort(np.abs(w))[6:]    # drop the six projected zero modes
    keep = keep[np.argsort(w[keep])]
    return w[keep], v[:, keep]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    calc = make_calc_factory(a.model, a.device, 'float64')()
    hrows, lrows = [], []
    for model, rx, label, ra, rb in CASES:
        if model != a.model:
            continue
        sa, sb = read(f'{ra}/{model}/{rx}/transition_state.xyz'), read(f'{rb}/{model}/{rx}/transition_state.xyz')
        ea = json.load(open(f'{ra}/{model}/{rx}/result.json'))['e_ts_ev']
        eb = json.load(open(f'{rb}/{model}/{rx}/result.json'))['e_ts_ev']
        Z = sa.numbers
        xa, xb = align(sa.positions, sb.positions), sb.positions
        wa, va = hessian(calc, Z, xa)
        wb, vb = hessian(calc, Z, xb)
        dx = (xb - xa).ravel()
        unit = dx / np.linalg.norm(dx)
        for tag, w, v, stored, x in (('A', wa, va, ea, xa), ('B', wb, vb, eb, xb)):
            e = forces(calc, Z, x)[0]
            hrows.append(dict(model=model, rxn=rx, case=label, structure=tag,
                              n_negative=int((w < NEG_TOL).sum()), w1=w[0], w2=w[1], w3=w[2],
                              overlap_softest=abs(float(unit @ v[:, np.argmin(np.abs(w))])),
                              overlap_lowest=abs(float(unit @ v[:, 0])),
                              e_stored_ev=stored, e_recomputed_ev=e, de_stored_mev=(e - stored) * 1e3))
            print('%-9s %s %-17s %s  negative modes %d  lowest %.3f %.3f %.3f eV/A2  E recomputed - stored %+.2f meV'
                  % (model, rx, label, tag, hrows[-1]['n_negative'], w[0], w[1], w[2], hrows[-1]['de_stored_mev']),
                  flush=True)
        ts = np.round(np.arange(-0.2, 1.2001, 0.05), 3)
        E, Fp = [], []
        for t in ts:
            e, f = forces(calc, Z, xa + t * (xb - xa))
            E.append(e)
            Fp.append(-float(f @ dx))           # dE/dt from the forces
        E, Fp = np.array(E), np.array(Fp)
        de_num = np.diff(E) / np.diff(ts)
        de_ana = 0.5 * (Fp[1:] + Fp[:-1])
        for t, e, g in zip(ts, E, Fp):
            lrows.append(dict(model=model, rxn=rx, case=label, t=float(t), e_rel_mev=(e - E.min()) * 1e3, dedt_force=g))
        print('   line: E range %.1f meV, max |dE/dt numeric - from forces| %.1f meV per unit t, length %.3f A'
              % ((E.max() - E.min()) * 1e3, np.abs(de_num - de_ana).max() * 1e3, np.linalg.norm(dx)), flush=True)

    for name, rows in (('ts_hessian.csv', hrows), ('ts_linescan.csv', lrows)):
        p = os.path.join(REPO, 'results', name)
        new = not os.path.exists(p)
        with open(p, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
    print('END', a.model)
