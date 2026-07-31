"""
Stability analysis of the CONVERGED BROKEN-SYMMETRY UKS solutions.

The table from job 10687985 reports mf.stability() on the RKS solution, which
answers "does a BS solution lie below RKS". It does not say whether the BS
solution is itself the lowest: an internally unstable UKS solution means a
still-lower UKS solution exists.

Scope: rows with a successfully converged BS solution (dE_BS < 0 and
<S^2> > 0.05). Skipped: 'stable, no BS solution', 'rks_not_converged', and the
invalid rxn8837/UMA-S row (dE > 0 at S^2 = 0).

The previous job did not retain orbitals (no chkfile, no mo_coeff dump), so the
BS solution is re-converged here by the identical Route-1 path. The re-converged
dE_BS / <S^2> are compared against the stored values as a reproducibility check.
Orbitals ARE saved this time (bs_<tag>.npz).

Per row:
    mf_u.stability(internal=True, external=True, return_status=True)
      -> uks_int_stable, uks_lmin_int, uks_ext_stable, uks_lmin_ext
    if internally unstable: follow the internal instability ONCE
      -> E_UKS2, dE2 [meV] relative to the first BS solution, <S^2>_2
         COLLAPSED if |dE2| < 0.1 meV

Usage: python uks_stability.py <rxn>
Output: ~/uks_stab/{rxn}/result.json
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
PREV = f'{HOME}/grad_model_ts'
OUT_ROOT = f'{HOME}/uks_stab'
BASIS = 'def2-tzvp'
XC = 'wb97m_v'
HA_TO_MEV = 27211.386

SOURCES = [
    ('RKS-ref', f'{HOME}/orca_neb_results/{{r}}/transition_state.xyz'),
    ('UMA-S',   f'{HOME}/uma_neb_results/{{r}}/transition_state.xyz'),
    ('UMA-M',   f'{HOME}/uma_m_neb_results/{{r}}/transition_state.xyz'),
    ('eSEN',    f'{HOME}/esen_neb_results/{{r}}/transition_state.xyz'),
]

# rhf_* for the RKS check, uhf_* for the UKS check
EIG_RE = re.compile(r'((?:rhf|uhf|rks|uks)_(?:internal|external))[^\[]*\[([^\]]*)\]')


def xyz_to_mol(path, logfile, max_memory):
    atoms = ase_read(path)
    s = '\n'.join(f'{e} {x:.8f} {y:.8f} {z:.8f}'
                  for e, (x, y, z) in zip(atoms.get_chemical_symbols(),
                                          atoms.get_positions()))
    mol = gto.Mole()
    mol.atom = s; mol.basis = BASIS; mol.spin = 0; mol.charge = 0
    mol.verbose = 4; mol.max_memory = max_memory
    mol.output = logfile
    mol.build()
    return mol


def parse_new_eigs(logfile, pos):
    """eigenvalues written to the log AFTER byte offset `pos`."""
    with open(logfile, errors='replace') as f:
        f.seek(pos)
        txt = f.read()
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
        key = 'int' if 'internal' in kind else 'ext'
        out.setdefault(key, vals)
    return out


def logpos(mol):
    mol.stdout.flush()
    return mol.stdout.tell()


def run_row(rxn, tag, xyz, prev_bs, outdir, max_mem):
    rec = {'source': tag}
    t0 = time.time()
    logf = os.path.join(outdir, f'uksstab_{tag.replace("-", "_")}.log')
    mol = xyz_to_mol(xyz, logf, max_mem)

    mf = dft.RKS(mol)
    mf.xc = XC; mf.grids.level = 3; mf.max_cycle = 300
    mf.conv_tol = 1e-10; mf.max_memory = max_mem
    mf.kernel()
    if not mf.converged:
        rec['error'] = 'rks_not_converged'
        return rec
    e_rks = float(mf.e_tot)

    _, mo_ext, _, ext_stable = mf.stability(internal=True, external=True,
                                            return_status=True)
    if ext_stable:
        rec['error'] = 'rks_externally_stable_on_recompute'
        return rec

    # ---- re-converge the BS solution, Route 1 (identical to job 10687985) ---
    mf_u = mf.to_uks()
    mf_u.xc = XC; mf_u.grids.level = 3; mf_u.max_cycle = 300
    mf_u.conv_tol = 1e-10; mf_u.max_memory = max_mem
    n1 = mf_u.newton(); n1.max_cycle = 200; n1.conv_tol = 1e-10
    n1.kernel(mf_u.make_rdm1(mo_ext, mf_u.mo_occ))

    e_bs = float(n1.e_tot)
    de_bs = (e_bs - e_rks) * HA_TO_MEV
    s2_bs = float(n1.spin_square()[0])
    rec['bs_reconverged'] = {'e_uks': round(e_bs, 10),
                             'de_meV': round(de_bs, 3),
                             's2': round(s2_bs, 6),
                             'converged': bool(n1.converged)}
    # reproducibility against the stored values
    if prev_bs:
        rec['repro'] = {
            'de_meV_prev': prev_bs.get('de_meV'),
            'de_meV_diff': round(de_bs - prev_bs.get('de_meV', 0), 3),
            's2_prev': prev_bs.get('s2'),
            's2_diff': round(s2_bs - prev_bs.get('s2', 0), 6)}
    print(f'  [{tag}] BS re-conv: dE={de_bs:.1f} meV (stored '
          f'{prev_bs.get("de_meV") if prev_bs else "?"})  S2={s2_bs:.4f}', flush=True)

    # ---- stability OF THE BS SOLUTION --------------------------------------
    mf_u.mo_coeff = n1.mo_coeff
    mf_u.mo_occ = n1.mo_occ
    mf_u.mo_energy = n1.mo_energy
    mf_u.e_tot = e_bs
    mf_u.converged = True

    np.savez_compressed(os.path.join(outdir, f'bs_{tag.replace("-", "_")}.npz'),
                        mo_coeff=np.asarray(n1.mo_coeff),
                        mo_occ=np.asarray(n1.mo_occ),
                        mo_energy=np.asarray(n1.mo_energy),
                        e_tot=e_bs, s2=s2_bs)

    p0 = logpos(mol)
    mo_i, mo_e, u_int, u_ext = mf_u.stability(internal=True, external=True,
                                              return_status=True)
    eigs = parse_new_eigs(logf, p0)
    rec['uks_int_stable'] = bool(u_int)
    rec['uks_ext_stable'] = bool(u_ext)
    rec['uks_lmin_int'] = round(min(eigs['int']), 8) if eigs.get('int') else None
    rec['uks_lmin_ext'] = round(min(eigs['ext']), 8) if eigs.get('ext') else None
    print(f'  [{tag}] UKS stab: int={u_int} ({rec["uks_lmin_int"]})  '
          f'ext={u_ext} ({rec["uks_lmin_ext"]})', flush=True)

    # ---- follow the internal instability once ------------------------------
    if not u_int:
        n2 = mf_u.newton(); n2.max_cycle = 200; n2.conv_tol = 1e-10
        n2.kernel(mf_u.make_rdm1(mo_i, mf_u.mo_occ))
        e2 = float(n2.e_tot)
        de2 = (e2 - e_bs) * HA_TO_MEV
        s22 = float(n2.spin_square()[0])
        rec['follow_int'] = {'e_uks2': round(e2, 10),
                             'de2_meV': round(de2, 3),
                             's2_2': round(s22, 6),
                             'converged': bool(n2.converged),
                             'collapsed': bool(abs(de2) < 0.1)}
        print(f'  [{tag}] followed internal: dE2={de2:.3f} meV  S2={s22:.4f}'
              + ('  COLLAPSED' if abs(de2) < 0.1 else ''), flush=True)
        np.savez_compressed(
            os.path.join(outdir, f'bs2_{tag.replace("-", "_")}.npz'),
            mo_coeff=np.asarray(n2.mo_coeff), mo_occ=np.asarray(n2.mo_occ),
            mo_energy=np.asarray(n2.mo_energy), e_tot=e2, s2=s22)
    else:
        rec['follow_int'] = None

    rec['elapsed_s'] = round(time.time() - t0, 1)
    return rec


def main(rxn):
    outdir = os.path.join(OUT_ROOT, rxn)
    os.makedirs(outdir, exist_ok=True)
    max_mem = int(os.environ.get('PYSCF_MAX_MEMORY', 50000))

    prev = json.load(open(f'{PREV}/{rxn}/result.json'))
    prevmap = {g['source']: g for g in prev.get('geometries', [])}

    out = {'rxn': rxn, 'level': f'{XC}/{BASIS} (PySCF, grids 3, conv 1e-10)',
           'rows': []}
    outfile = os.path.join(outdir, 'result.json')

    for tag, pat in SOURCES:
        g = prevmap.get(tag, {})
        bs = g.get('bs')
        if 'error' in g:
            out['rows'].append({'source': tag, 'skipped': g['error']})
            print(f'\n=== {rxn} {tag}: SKIP ({g["error"]}) ===', flush=True)
            continue
        if not bs:
            out['rows'].append({'source': tag, 'skipped': 'no_bs_solution'})
            print(f'\n=== {rxn} {tag}: SKIP (stable, no BS) ===', flush=True)
            continue
        de, s2 = bs.get('de_meV'), bs.get('s2')
        if de is None or s2 is None or de >= 0 or s2 <= 0.05:
            out['rows'].append({'source': tag,
                                'skipped': f'invalid_bs (dE={de}, S2={s2})'})
            print(f'\n=== {rxn} {tag}: SKIP (invalid dE={de} S2={s2}) ===',
                  flush=True)
            continue

        print(f'\n=== {rxn} {tag} ===', flush=True)
        rec = run_row(rxn, tag, pat.format(r=rxn), bs, outdir, max_mem)
        out['rows'].append(rec)
        json.dump(out, open(outfile, 'w'), indent=2)

    json.dump(out, open(outfile, 'w'), indent=2)
    print(f'\nDone {rxn}: {outfile}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: uks_stability.py <rxn>'); sys.exit(1)
    try:
        main(sys.argv[1])
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)
