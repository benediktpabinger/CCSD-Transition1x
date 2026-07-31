"""
Complete stability pipeline: RKS -> stability -> broken symmetry -> gradients
-> stability of the BS solution, for one reaction across four geometry sources.

Supersedes grad_at_model_ts.py + uks_stability.py, which did this in two passes
and had to re-converge every BS solution because the first pass kept no orbitals.

Geometry sources:
    RKS-ref  ~/orca_neb_results/{rxn}/transition_state.xyz
    UMA-S    ~/uma_neb_results/{rxn}/transition_state.xyz
    UMA-M    ~/uma_m_neb_results/{rxn}/transition_state.xyz
    eSEN     ~/esen_neb_results/{rxn}/transition_state.xyz

Per geometry, wB97M-V/def2-TZVP (PySCF, grids 3, conv_tol 1e-10):
  1. RKS + nuclear gradient                   -> max/RMS |grad| [eV/A]
  2. mf.stability(internal, external)         -> int/ext stable, lambda_min
  3. if externally unstable: Route 1 into UKS -> E_UKS, dE [meV], <S^2>,
     Mulliken spins, BS gradient
       Route 2 (triplet-seeded beta-HOMO flip) as fallback if Route 1 lands on a
       HIGHER solution (dE > 0) or collapses.
  4. mf_u.stability(internal, external) on the BS solution
       if internally unstable: follow once -> E2, dE2, <S^2>_2
  5. orbitals saved to bs_<tag>.npz / bs2_<tag>.npz

No geometry optimisation.

Implementation notes carried over from earlier runs:
  * mf.stability() does not return eigenvalues; mol.output is set BEFORE
    mol.build() and the log is parsed by byte offset so the RKS and UKS
    analyses do not mix.
  * the UKS seed must be mf_rks.to_uks(); a fresh dft.UKS(mol) has mo_occ=None.
  * plain DIIS collapses the BS solution -> second-order Newton is required.
  * |lambda| > 1 Ha is unphysical for an orbital Hessian and flags a Davidson
    breakdown (seen twice in job 10688500), recorded as such rather than as a
    number.

Usage: python stability_pipeline.py <rxn>
Output: ~/stab_pipeline/{rxn}/result.json
"""
import json
import os
import re
import sys
import time

import numpy as np
from ase.io import read as ase_read
from pyscf import gto, dft

HOME = '/home/energy/s242862'
OUT_ROOT = f'{HOME}/stab_pipeline'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
HA_BOHR_TO_EV_ANG = 51.42207
HA_TO_MEV = 27211.386
S2_MIN = 0.05          # below this the BS solution counts as collapsed
LAMBDA_MAX = 1.0       # |lambda| above this = Davidson breakdown

SOURCES = [
    ('RKS-ref', f'{HOME}/orca_neb_results/{{r}}/transition_state.xyz'),
    ('UMA-S',   f'{HOME}/uma_neb_results/{{r}}/transition_state.xyz'),
    ('UMA-M',   f'{HOME}/uma_m_neb_results/{{r}}/transition_state.xyz'),
    ('eSEN',    f'{HOME}/esen_neb_results/{{r}}/transition_state.xyz'),
]

EIG_RE = re.compile(r'((?:rhf|uhf|rks|uks)_(?:internal|external))[^\[]*\[([^\]]*)\]')


def xyz_to_mol(path, logfile, max_memory, spin=0):
    atoms = ase_read(path)
    s = '\n'.join(f'{e} {x:.8f} {y:.8f} {z:.8f}'
                  for e, (x, y, z) in zip(atoms.get_chemical_symbols(),
                                          atoms.get_positions()))
    mol = gto.Mole()
    mol.atom = s; mol.basis = BASIS; mol.spin = spin; mol.charge = 0
    mol.verbose = 4; mol.max_memory = max_memory
    mol.output = logfile              # BEFORE build, else eigenvalues are lost
    mol.build()
    return mol


def make_rks(mol, mem, level_shift=0.0):
    mf = dft.RKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
    if level_shift:
        mf.level_shift = level_shift
    return mf


def make_uks(mol, mem):
    mf = dft.UKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = mem
    return mf


def grad_stats(g):
    g = np.asarray(g) * HA_BOHR_TO_EV_ANG
    return {'max_evang': round(float(np.max(np.abs(g))), 6),
            'rms_evang': round(float(np.sqrt(np.mean(g ** 2))), 6)}


def logpos(mol):
    mol.stdout.flush()
    return mol.stdout.tell()


def parse_eigs(logfile, pos):
    with open(logfile, errors='replace') as f:
        f.seek(pos); txt = f.read()
    out = {}
    for kind, body in EIG_RE.findall(txt):
        vals = []
        for tok in body.replace(',', ' ').split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if not vals:
            continue
        out.setdefault('int' if 'internal' in kind else 'ext', vals)
    return out


def lam(eigs, key):
    """lowest eigenvalue, with a plausibility guard against Davidson breakdown"""
    v = eigs.get(key)
    if not v:
        return None, None
    m = min(v)
    if abs(m) > LAMBDA_MAX:
        return None, f'davidson_breakdown ({m:.3g})'
    return round(m, 8), None


def mulliken_spin(mol, mf_u, thr=0.05, top=2):
    dm_a, dm_b = mf_u.make_rdm1()
    S = mf_u.get_ovlp()
    spin_ao = np.einsum('ij,ji->i', dm_a - dm_b, S)
    idx = [a[0] for a in mol.ao_labels(fmt=None)]
    sp = np.zeros(mol.natm)
    for i, k in enumerate(idx):
        sp[k] += spin_ao[i]
    order = np.argsort(-np.abs(sp))
    return [{'atom_idx': int(k), 'symbol': mol.atom_pure_symbol(int(k)),
             'spin_pop': round(float(sp[k]), 5)}
            for k in order[:top] if abs(sp[k]) > thr]


def route2_triplet(xyz, mol, mem, e_rks, outdir, tag):
    """Triplet-seeded beta-HOMO flip. Fallback when Route 1 misbehaves."""
    logf = os.path.join(outdir, f'r2_{tag}.log')
    mol_t = xyz_to_mol(xyz, logf, mem, spin=2)
    mf_t = make_uks(mol_t, mem); mf_t.kernel()
    if not mf_t.converged:
        return None
    mo_a, mo_b = mf_t.mo_coeff
    nalpha_t = (mol_t.nelectron + 2) // 2
    nbs = mol.nelectron // 2
    a2, b2 = mo_a.copy(), mo_b.copy()
    b2[:, nbs - 1] = mo_a[:, nalpha_t - 1]
    mf_s = make_uks(mol, mem)
    mf_s.mo_occ = np.array([np.concatenate([np.ones(nbs), np.zeros(a2.shape[1]-nbs)]),
                            np.concatenate([np.ones(nbs), np.zeros(b2.shape[1]-nbs)])])
    dm = mf_s.make_rdm1(np.array([a2, b2]), mf_s.mo_occ)
    n = mf_s.newton(); n.max_cycle = 200; n.conv_tol = 1e-10
    n.kernel(dm)
    return n


def run_geometry(rxn, tag, xyz, outdir, mem):
    rec = {'source': tag, 'xyz': xyz}
    if not os.path.exists(xyz):
        rec['error'] = 'geometry_not_found'; return rec
    t0 = time.time()
    safe = tag.replace('-', '_')
    logf = os.path.join(outdir, f'pyscf_{safe}.log')
    mol = xyz_to_mol(xyz, logf, mem)
    rec['natm'] = mol.natm; rec['nelectron'] = mol.nelectron

    # ---- 1. RKS (+ level-shift retry) --------------------------------------
    mf = make_rks(mol, mem)
    mf.kernel()
    if not mf.converged:
        print(f'  [{tag}] RKS not converged, retry with level_shift=0.2', flush=True)
        mf = make_rks(mol, mem, level_shift=0.2)
        mf.kernel()
        if mf.converged:
            mf.level_shift = 0.0
            mf.kernel(mf.make_rdm1())      # release the shift, re-converge
        rec['rks_level_shift_used'] = True
    rec['rks_converged'] = bool(mf.converged)
    rec['e_rks'] = round(float(mf.e_tot), 10)
    if not mf.converged:
        rec['error'] = 'rks_not_converged'; return rec
    rec['rks_grad'] = grad_stats(mf.nuc_grad_method().kernel())
    print(f"  [{tag}] E_RKS={mf.e_tot:.8f} max|g|="
          f"{rec['rks_grad']['max_evang']:.4f}", flush=True)

    # ---- 2. RKS stability ---------------------------------------------------
    p0 = logpos(mol)
    _, mo_ext, int_st, ext_st = mf.stability(internal=True, external=True,
                                             return_status=True)
    e = parse_eigs(logf, p0)
    rec['int_stable'] = bool(int_st); rec['ext_stable'] = bool(ext_st)
    rec['lmin_int'], f1 = lam(e, 'int')
    rec['lmin_ext'], f2 = lam(e, 'ext')
    if f1 or f2:
        rec['lambda_flag'] = f1 or f2
    print(f"  [{tag}] int={int_st} ext={ext_st} lmin_ext={rec['lmin_ext']}", flush=True)

    if ext_st:
        rec['bs'] = None
        rec['elapsed_s'] = round(time.time() - t0, 1)
        return rec

    # ---- 3. broken symmetry -------------------------------------------------
    mf_u = mf.to_uks()                      # NOT dft.UKS(mol): mo_occ would be None
    mf_u.xc = XC; mf_u.grids.level = 3
    mf_u.max_cycle = 300; mf_u.conv_tol = 1e-10; mf_u.max_memory = mem
    n1 = mf_u.newton(); n1.max_cycle = 200; n1.conv_tol = 1e-10
    n1.kernel(mf_u.make_rdm1(mo_ext, mf_u.mo_occ))

    e_bs = float(n1.e_tot)
    de = (e_bs - float(mf.e_tot)) * HA_TO_MEV
    s2 = float(n1.spin_square()[0])
    route = 1
    bad = (de > 0) or (s2 < S2_MIN)
    if bad:
        print(f'  [{tag}] Route 1 unusable (dE={de:.1f} meV, S2={s2:.4f}); '
              f'trying Route 2', flush=True)
        n2 = route2_triplet(xyz, mol, mem, float(mf.e_tot), outdir, safe)
        if n2 is not None:
            e2 = float(n2.e_tot); d2 = (e2 - float(mf.e_tot)) * HA_TO_MEV
            s22 = float(n2.spin_square()[0])
            if d2 < 0 and s22 > S2_MIN and (de > 0 or d2 < de):
                n1, e_bs, de, s2, route = n2, e2, d2, s22, 2
                bad = False
                print(f'  [{tag}] Route 2 succeeded: dE={de:.1f} S2={s2:.4f}',
                      flush=True)

    bs = {'route': route, 'converged': bool(n1.converged),
          'e_uks': round(e_bs, 10), 'de_meV': round(de, 3), 's2': round(s2, 6)}
    if bad:
        bs['invalid'] = f'dE={de:.1f} meV, S2={s2:.4f} (both routes failed)'
        rec['bs'] = bs
        rec['elapsed_s'] = round(time.time() - t0, 1)
        print(f'  [{tag}] BS INVALID', flush=True)
        return rec

    mf_u.mo_coeff = n1.mo_coeff; mf_u.mo_occ = n1.mo_occ
    mf_u.mo_energy = n1.mo_energy; mf_u.e_tot = e_bs; mf_u.converged = True
    bs['bs_grad'] = grad_stats(mf_u.nuc_grad_method().kernel())
    bs['spin_populations'] = mulliken_spin(mol, mf_u)
    np.savez_compressed(os.path.join(outdir, f'bs_{safe}.npz'),
                        mo_coeff=np.asarray(n1.mo_coeff),
                        mo_occ=np.asarray(n1.mo_occ),
                        mo_energy=np.asarray(n1.mo_energy),
                        e_tot=e_bs, s2=s2)
    print(f"  [{tag}] BS route {route}: dE={de:.1f} S2={s2:.4f} "
          f"max|g|={bs['bs_grad']['max_evang']:.4f}", flush=True)

    # ---- 4. stability OF the BS solution -------------------------------------
    p1 = logpos(mol)
    mo_i, _, u_int, u_ext = mf_u.stability(internal=True, external=True,
                                           return_status=True)
    eu = parse_eigs(logf, p1)
    bs['uks_int_stable'] = bool(u_int); bs['uks_ext_stable'] = bool(u_ext)
    bs['uks_lmin_int'], g1 = lam(eu, 'int')
    bs['uks_lmin_ext'], g2 = lam(eu, 'ext')
    if g1 or g2:
        bs['uks_lambda_flag'] = g1 or g2
    print(f"  [{tag}] UKS stab: int={u_int} ({bs['uks_lmin_int']}) "
          f"ext={u_ext} ({bs['uks_lmin_ext']})", flush=True)

    if not u_int:
        n3 = mf_u.newton(); n3.max_cycle = 200; n3.conv_tol = 1e-10
        n3.kernel(mf_u.make_rdm1(mo_i, mf_u.mo_occ))
        e3 = float(n3.e_tot); d3 = (e3 - e_bs) * HA_TO_MEV
        s23 = float(n3.spin_square()[0])
        bs['follow_int'] = {'e_uks2': round(e3, 10), 'de2_meV': round(d3, 3),
                            's2_2': round(s23, 6),
                            'converged': bool(n3.converged),
                            'collapsed': bool(abs(d3) < 0.1)}
        np.savez_compressed(os.path.join(outdir, f'bs2_{safe}.npz'),
                            mo_coeff=np.asarray(n3.mo_coeff),
                            mo_occ=np.asarray(n3.mo_occ),
                            mo_energy=np.asarray(n3.mo_energy),
                            e_tot=e3, s2=s23)
        print(f'  [{tag}] followed internal: dE2={d3:.2f} S2={s23:.4f}', flush=True)
    else:
        bs['follow_int'] = None

    rec['bs'] = bs
    rec['elapsed_s'] = round(time.time() - t0, 1)
    return rec


def main(rxn):
    outdir = os.path.join(OUT_ROOT, rxn)
    os.makedirs(outdir, exist_ok=True)
    mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))
    t0 = time.time()
    out = {'rxn': rxn, 'level': f'{XC}/{BASIS} (PySCF, grids 3, conv 1e-10)',
           'geometries': []}
    of = os.path.join(outdir, 'result.json')
    for tag, pat in SOURCES:
        print(f'\n=== {rxn} : {tag} ===', flush=True)
        out['geometries'].append(run_geometry(rxn, tag, pat.format(r=rxn),
                                              outdir, mem))
        json.dump(out, open(of, 'w'), indent=2)     # checkpoint per geometry
    out['elapsed_total_s'] = round(time.time() - t0, 1)
    json.dump(out, open(of, 'w'), indent=2)
    print(f'\nDone {rxn}: {of} ({out["elapsed_total_s"]:.0f}s)', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: stability_pipeline.py <rxn>'); sys.exit(1)
    try:
        main(sys.argv[1])
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)
