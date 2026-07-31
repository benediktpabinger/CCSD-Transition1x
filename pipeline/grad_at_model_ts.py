"""
RKS and broken-symmetry UKS gradients at model-predicted TS geometries.

For one reaction, four geometry sources are treated identically:
    RKS-ref  ~/orca_neb_results/{rxn}/transition_state.xyz
    UMA-S    ~/uma_neb_results/{rxn}/transition_state.xyz
    UMA-M    ~/uma_m_neb_results/{rxn}/transition_state.xyz
    eSEN     ~/esen_neb_results/{rxn}/transition_state.xyz

Per geometry, at wB97M-V/def2-TZVP (PySCF, grids level 3, conv_tol 1e-10):
  1. converge RKS, nuclear gradient           -> max|dE|, RMS|dE|  [eV/A]
  2. mf.stability(internal=True, external=True, return_status=True)
       -> int_stable, lambda_min_int, ext_stable, lambda_min_ext
  3. if externally unstable: Route 1 into UKS
       mf_rks.to_uks(); dm0 = make_rdm1(mo_ext, mo_occ); mf_u.newton().kernel(dm0)
       -> E_UKS, dE [meV], <S^2>, BS gradient max/RMS [eV/A]
       COLLAPSED if <S^2> < 0.3 and |dE| < 0.1 meV

No geometry optimisation.

Implementation notes carried over from earlier runs in this project:
  * mf.stability() does NOT return the eigenvalues; they go to the PySCF logger.
    mol.output is therefore set BEFORE mol.build() and the log is parsed.
  * the UKS seed must come from mf_rks.to_uks(), not a fresh dft.UKS(mol):
    the latter has mo_occ = None and make_rdm1 then raises.
  * plain DIIS collapses the BS solution; second-order Newton is required.

Usage: python grad_at_model_ts.py <rxn>
Output: ~/grad_model_ts/{rxn}/result.json
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
OUT_ROOT = f'{HOME}/grad_model_ts'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
HA_BOHR_TO_EV_ANG = 51.42207
HA_TO_MEV = 27211.386
S2_MIN = 0.3

SOURCES = [
    ('RKS-ref', f'{HOME}/orca_neb_results/{{r}}/transition_state.xyz'),
    ('UMA-S',   f'{HOME}/uma_neb_results/{{r}}/transition_state.xyz'),
    ('UMA-M',   f'{HOME}/uma_m_neb_results/{{r}}/transition_state.xyz'),
    ('eSEN',    f'{HOME}/esen_neb_results/{{r}}/transition_state.xyz'),
]

EIG_RE = re.compile(r'(rhf_internal|rhf_external)[^\[]*\[([^\]]*)\]')


def xyz_to_mol(path, logfile, max_memory, spin=0):
    atoms = ase_read(path)
    atom_str = '\n'.join(
        f'{s} {x:.8f} {y:.8f} {z:.8f}'
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(),
                                atoms.get_positions()))
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = BASIS
    mol.spin = spin
    mol.charge = 0
    mol.verbose = 4
    mol.max_memory = max_memory
    mol.output = logfile          # must be set BEFORE build to capture eigenvalues
    mol.build()
    return mol


def make_rks(mol, max_mem):
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = 3
    mf.max_cycle = 300
    mf.conv_tol = 1e-10
    mf.max_memory = max_mem
    return mf


def make_uks(mol, max_mem):
    mf = dft.UKS(mol)
    mf.xc = XC
    mf.grids.level = 3
    mf.max_cycle = 300
    mf.conv_tol = 1e-10
    mf.max_memory = max_mem
    return mf


def grad_stats(g_ha_bohr):
    g = np.asarray(g_ha_bohr) * HA_BOHR_TO_EV_ANG
    return {'max_evang': round(float(np.max(np.abs(g))), 6),
            'rms_evang': round(float(np.sqrt(np.mean(g ** 2))), 6)}


def parse_eigs(logfile):
    """lowest internal / external stability eigenvalues from the PySCF log."""
    out = {}
    try:
        txt = open(logfile, errors='replace').read()
    except OSError:
        return out
    for kind, body in EIG_RE.findall(txt):
        vals = []
        for tok in body.replace(',', ' ').split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if not vals:
            continue
        key = 'int' if 'internal' in kind else 'ext'
        out.setdefault(key, vals)          # first occurrence = the RKS check
    return out


def mulliken_spin(mol, mf_u, thr=0.05, top=2):
    dm_a, dm_b = mf_u.make_rdm1()
    S = mf_u.get_ovlp()
    spin_ao = np.einsum('ij,ji->i', dm_a - dm_b, S)
    aidx = [a[0] for a in mol.ao_labels(fmt=None)]
    sp = np.zeros(mol.natm)
    for i, k in enumerate(aidx):
        sp[k] += spin_ao[i]
    order = np.argsort(-np.abs(sp))
    return [{'atom_idx': int(k), 'symbol': mol.atom_pure_symbol(int(k)),
             'spin_pop': round(float(sp[k]), 5)}
            for k in order[:top] if abs(sp[k]) > thr]


def run_geometry(rxn, tag, xyz, outdir, max_mem):
    rec = {'source': tag, 'xyz': xyz}
    if not os.path.exists(xyz):
        rec['error'] = 'geometry_not_found'
        print(f'  [{tag}] MISSING {xyz}', flush=True)
        return rec

    t0 = time.time()
    logf = os.path.join(outdir, f'pyscf_{tag.replace("-", "_")}.log')
    mol = xyz_to_mol(xyz, logf, max_mem)
    rec['natm'] = mol.natm
    rec['nelectron'] = mol.nelectron

    # ---- 1. RKS + gradient -------------------------------------------------
    mf = make_rks(mol, max_mem)
    mf.kernel()
    rec['rks_converged'] = bool(mf.converged)
    rec['e_rks'] = round(float(mf.e_tot), 10)
    if not mf.converged:
        rec['error'] = 'rks_not_converged'
        print(f'  [{tag}] RKS NOT converged', flush=True)
        return rec
    rec['rks_grad'] = grad_stats(mf.nuc_grad_method().kernel())
    print(f'  [{tag}] E_RKS={mf.e_tot:.8f}  '
          f"max|g|={rec['rks_grad']['max_evang']:.4f} eV/A", flush=True)

    # ---- 2. stability ------------------------------------------------------
    mo_int, mo_ext, int_stable, ext_stable = mf.stability(
        internal=True, external=True, return_status=True)
    mol.stdout.flush()
    eigs = parse_eigs(logf)
    rec['int_stable'] = bool(int_stable)
    rec['ext_stable'] = bool(ext_stable)
    rec['lmin_int'] = round(min(eigs['int']), 8) if eigs.get('int') else None
    rec['lmin_ext'] = round(min(eigs['ext']), 8) if eigs.get('ext') else None
    print(f'  [{tag}] int={int_stable} ext={ext_stable} '
          f"lmin_ext={rec['lmin_ext']}", flush=True)

    # ---- 3. broken symmetry, Route 1 --------------------------------------
    if ext_stable:
        rec['bs'] = None
        rec['elapsed_s'] = round(time.time() - t0, 1)
        return rec

    mf_u = mf.to_uks()                    # to_uks(), NOT a fresh dft.UKS(mol)
    mf_u.xc = XC
    mf_u.grids.level = 3
    mf_u.max_cycle = 300
    mf_u.conv_tol = 1e-10
    mf_u.max_memory = max_mem
    dm0 = mf_u.make_rdm1(mo_ext, mf_u.mo_occ)
    n = mf_u.newton()                     # second-order; DIIS collapses BS
    n.max_cycle = 200
    n.conv_tol = 1e-10
    n.kernel(dm0)

    e_uks = float(n.e_tot)
    de = (e_uks - float(mf.e_tot)) * HA_TO_MEV
    s2 = float(n.spin_square()[0])
    collapsed = (abs(de) < 0.1 and s2 < 0.05)
    bs = {'route': 1, 'converged': bool(n.converged),
          'collapsed': bool(collapsed),
          'e_uks': round(e_uks, 10), 'de_meV': round(de, 3),
          's2': round(s2, 6)}

    if collapsed or s2 < S2_MIN:
        bs['note'] = f'S2={s2:.4f} below {S2_MIN}' if not collapsed else 'COLLAPSED'

    if not collapsed:
        mfg = make_uks(mol, max_mem)
        mfg.mo_coeff = n.mo_coeff
        mfg.mo_occ = n.mo_occ
        mfg.mo_energy = n.mo_energy
        mfg.e_tot = e_uks
        mfg.converged = True
        bs['bs_grad'] = grad_stats(mfg.nuc_grad_method().kernel())
        bs['spin_populations'] = mulliken_spin(mol, mfg)
        print(f'  [{tag}] BS dE={de:.1f} meV  S2={s2:.4f}  '
              f"max|g|={bs['bs_grad']['max_evang']:.4f} eV/A", flush=True)
    else:
        print(f'  [{tag}] BS COLLAPSED (dE={de:.3f} meV, S2={s2:.4f})', flush=True)

    rec['bs'] = bs
    rec['elapsed_s'] = round(time.time() - t0, 1)
    return rec


def main(rxn):
    outdir = os.path.join(OUT_ROOT, rxn)
    os.makedirs(outdir, exist_ok=True)
    max_mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))
    t0 = time.time()

    result = {'rxn': rxn, 'level': f'{XC}/{BASIS} (PySCF, grids 3, conv 1e-10)',
              'geometries': []}
    outfile = os.path.join(outdir, 'result.json')

    for tag, pat in SOURCES:
        print(f'\n=== {rxn} : {tag} ===', flush=True)
        rec = run_geometry(rxn, tag, pat.format(r=rxn), outdir, max_mem)
        result['geometries'].append(rec)
        json.dump(result, open(outfile, 'w'), indent=2)   # checkpoint each source

    result['elapsed_total_s'] = round(time.time() - t0, 1)
    json.dump(result, open(outfile, 'w'), indent=2)
    print(f'\nDone {rxn}: {outfile}  ({result["elapsed_total_s"]:.0f}s)', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: grad_at_model_ts.py <rxn>')
        sys.exit(1)
    try:
        main(sys.argv[1])
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
