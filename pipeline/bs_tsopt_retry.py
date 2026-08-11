"""Second attempt at the transition states that came out wrong.

Two separate problems, two remedies.

(a) Wrong saddle. rxn1320, rxn4518 and rxn5691 converged to first-order saddles
    whose imaginary mode does not move the reactive bonds -- 89 and 102 cm-1 for
    two of them, so almost certainly a soft torsion. geomeTRIC follows the mode
    with the lowest Hessian eigenvalue, and where a torsion sits below the
    reaction mode it will follow that instead and walk out of the basin. The
    remedy is a tight trust radius: small steps keep the optimiser near the
    starting point long enough for the reaction mode to dominate. rxn1320 shows
    the failure clearly -- its breaking C-H bond ran from 1.98 to 3.36 A, i.e.
    the hydrogen came off entirely.

(b) Wrong basin. rxn8885 optimised away from the broken-symmetry region (S^2
    0.507 -> 0.153) while a model geometry of the same reaction is 71x more
    strongly broken; rxn1283 never converged and has a 46x factor. For these the
    remedy is a different starting point, not a different step size: begin at
    the model geometry where the breaking is strongest, as was done for rxn4113.

The `tight` remedy failed: all three reconverged to the same structure they came
from, rxn1320 to within 0.0009 A. The wrong saddle is where the optimisation
robustly goes from the reference, not an artefact of step size. Only `frommodel`
has ever produced a different answer -- three times out of three tried.

Usage: python bs_tsopt_retry.py <rxn> <mode>     mode = tight | frommodel
Environment: TSOPT_OUT overrides the output directory.
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

HOME = '/home/energy/s242862'
# configurable so a second sweep does not overwrite the first: the tight runs
# for rxn1320, rxn4518 and rxn5691 already occupy bs_tsopt_retry/
OUTDIR = os.environ.get('TSOPT_OUT', f'{HOME}/bs_tsopt_retry')
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
BOHR = 0.529177210903
S2_MIN = 0.05
MAXSTEPS = 400
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


def strongest_model(rx):
    """Model geometry whose broken-symmetry solution is deepest."""
    p = f'{HOME}/stab_pipeline/{rx}/result.json'
    best, name = 0.0, None
    for g in json.load(open(p))['geometries']:
        if g['source'] == 'RKS-ref':
            continue
        de = abs((g.get('bs') or {}).get('de_meV') or 0)
        if de > best:
            best, name = de, g['source']
    return name, best


def main(rx, mode):
    t0 = time.time()
    mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))
    out = f'{OUTDIR}/{rx}'
    os.makedirs(out, exist_ok=True)
    res = {'rxn': rx, 'mode': mode}

    if mode == 'tight':
        start = f'{HOME}/orca_neb_results/{rx}/transition_state.xyz'
        res['start'] = 'RKS-Referenz, enger Vertrauensradius'
        trust, tmax = 0.005, 0.02
    else:
        name, de = strongest_model(rx)
        if not name:
            res['status'] = 'KEINE_MODELLGEOMETRIE'
            json.dump(res, open(f'{out}/result.json', 'w'), indent=2); return 1
        start = f'{HOME}/{MODELDIR[name]}/{rx}/transition_state.xyz'
        res['start'] = f'{name}-Geometrie (dE_BS {de:.0f} meV)'
        res['start_model'] = name
        trust, tmax = None, None

    if not os.path.exists(start):
        res['status'] = 'STARTGEOMETRIE_FEHLT'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2); return 1
    print(f'{rx}: Start = {res["start"]}\n  {start}', flush=True)

    atoms = ase_read(start)
    sym = atoms.get_chemical_symbols()
    x0 = atoms.get_positions().copy()

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
    if ext_stable:
        res['status'] = 'EXTERN_STABIL_AM_START'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
        print('extern stabil -- keine BS-Loesung hier'); return 2

    mf_s = mf.to_uks()
    mf_s.xc = XC; mf_s.grids.level = 3
    mf_s.max_cycle = 300; mf_s.conv_tol = 1e-10; mf_s.max_memory = mem
    n0 = mf_s.newton(); n0.max_cycle = 200; n0.conv_tol = 1e-10
    n0.kernel(mf_s.make_rdm1(mo_ext, mf_s.mo_occ))
    s2_0 = float(n0.spin_square()[0])
    res['bs_initial'] = {'e_uks': round(float(n0.e_tot), 10),
                         'de_meV': round((float(n0.e_tot) - float(mf.e_tot)) * 27211.386, 3),
                         's2': round(s2_0, 6)}
    print(f'  BS: dE={res["bs_initial"]["de_meV"]:.1f} meV  S2={s2_0:.4f}',
          flush=True)
    if s2_0 < S2_MIN:
        res['status'] = 'KOLLABIERT'
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2); return 3

    BS['dm'] = n0.make_rdm1()
    mf_u = BSUKS(mol)
    mf_u.xc = XC; mf_u.grids.level = 3
    mf_u.max_cycle = 300; mf_u.conv_tol = 1e-10; mf_u.max_memory = mem
    mf_u.mo_coeff, mf_u.mo_occ = n0.mo_coeff, n0.mo_occ
    mf_u.mo_energy, mf_u.e_tot, mf_u.converged = n0.mo_energy, float(n0.e_tot), True

    def save():
        res['n_geom_steps'] = BS['step']
        res['step_log'] = BS['log']
        json.dump(res, open(f'{out}/result.json', 'w'), indent=2)
    BS['on_step'] = save

    kw = dict(transition=True, maxsteps=MAXSTEPS)
    if trust is not None:
        kw.update(trust=trust, tmax=tmax)
        res['trust'], res['tmax'] = trust, tmax
    try:
        conv, ts_mol = geometric_solver.kernel(mf_u, **kw)
    except TypeError as exc:
        # older geomeTRIC may not accept the trust keywords
        print(f'  trust nicht unterstuetzt ({exc}); ohne', flush=True)
        res['trust_unsupported'] = str(exc)
        conv, ts_mol = geometric_solver.kernel(mf_u, transition=True,
                                               maxsteps=MAXSTEPS)
    BS['on_step'] = None

    xe = ts_mol.atom_coords() * BOHR
    res['opt_converged'] = bool(conv)
    res['s2_final'] = BS['log'][-1]['s2'] if BS['log'] else None
    res['e_uks_final'] = BS['log'][-1]['e'] if BS['log'] else None
    res['rmsd_vs_start'] = round(kabsch(x0, xe), 6)
    for lab, p in (('rks_ref', f'{HOME}/orca_neb_results/{rx}/transition_state.xyz'),
                   ('old_bs_ts', None)):
        if lab == 'old_bs_ts':
            for d in ('bs_tsopt_v2', 'bs_tsopt_batch'):
                q = f'{HOME}/{d}/{rx}/ts_opt.xyz'
                if os.path.exists(q):
                    p = q
                    break
        if p and os.path.exists(p):
            res[f'rmsd_vs_{lab}'] = round(kabsch(xe, read_xyz(p)[1]), 6)
    res['status'] = 'converged' if conv else 'nicht_konvergiert'
    res['elapsed_s'] = round(time.time() - t0, 1)

    with open(f'{out}/ts_opt.xyz', 'w') as fh:
        fh.write(f'{ts_mol.natm}\n')
        fh.write(f'BS-UKS TS retry ({mode})  E={res["e_uks_final"]}  '
                 f'S2={res["s2_final"]}  converged={conv}\n')
        for s, (a, b, c) in zip(sym, xe):
            fh.write(f'{s} {a:.8f} {b:.8f} {c:.8f}\n')
    save()
    print(json.dumps({k: v for k, v in res.items() if k != 'step_log'},
                     indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
