#!/usr/bin/env python3
"""
Broken-symmetry UKS transition-state optimization -- v2 (branch-jump recovery), one reaction
per SLURM array task.

wB97M-V / def2-TZVP, PySCF.  Start geometry: ORCA RKS NEB TS.

  Step 0  RKS -> stability(internal, external) -> follow external instability
          eigenvector into UKS -> newton().  Require <S^2> > 0.3.
          If collapsed, try triplet-seeded beta-HOMO-flip guess ONCE.
          If still collapsed -> status COLLAPSED, stop.
  Step 1  geomeTRIC TS optimization, transition=True, maxsteps=300.
          hessian='first' is deliberately NOT passed (wB97M-V uses VV10 NLC and
          has no analytic Hessian in PySCF; passing it crashed job 10679114).
          BS maintained at every geometry step by reusing the previous step's
          converged BS orbitals as the newton guess.
          <S^2> logged every step; < 0.3 at any step -> BS_LOST flag.
  Step 2  SKIPPED -- no frequency/Hessian calculation in this batch.
  Step 3  Kabsch RMSD vs 7 references + the two reactive bond lengths.

Reactive bonds are derived exactly as in ~/_rxn_coord_full.py: the top-2 atom
pairs by |d_product - d_reactant| over the union of reactant/product bonds.

Usage : python bs_tsopt_batch.py <rxn>
Output: ~/bs_tsopt_batch/<rxn>/result.json   (rewritten after every geom step)
        ~/bs_tsopt_batch/<rxn>/ts_opt.xyz
"""

import os, sys, json, time
import numpy as np
import h5py
from ase.io import read as ase_read
from ase.data import covalent_radii
from pyscf import gto, dft
from pyscf.geomopt import geometric_solver

HOME   = '/home/energy/s242862'
OUTDIR = f'{HOME}/bs_tsopt_v2'
BASIS  = 'def2-tzvp'
XC     = 'wb97m_v'
BOHR   = 0.529177210903
S2_MIN = 0.3
MAXSTEPS = 300

MODELS = [
    ('MACE',  'mace_bare_neb_results'),
    ('Delta', 'mace_delta_neb_results_fw2'),
    ('UMA-S', 'uma_neb_results'),
    ('UMA-M', 'uma_m_neb_results'),
    ('eSEN',  'esen_neb_results'),
]
T1X_H5 = f'{HOME}/data/Transition1x.h5'


# -- shared BS state (module-level dict: copied by reference into scanner copies)
# 'dm' is the AO-basis density matrix of the previous geometry -- NOT mo_coeff.
BS = {'dm': None, 'in_newton': False, 'step': 0, 'log': [],
      'lost_at': None, 'bad_nelec_at': None, 'on_step': None,
      # v2 recovery state
      'last_good_dm': None, 'recoveries': [], 'n_recover': 0}

MAX_RECOVER = 10          # cap: recovery costs a full stability analysis


def rederive_bs(mol, mem):
    """Re-derive the broken-symmetry solution from scratch at THIS geometry.

    Used when the density chain jumps branches mid-optimisation.  Observed in
    job 10684xxx on five reactions: at essentially unchanged geometry the SCF
    converged onto a higher, less spin-polarised solution -- energy up 20-60
    meV, <S^2> down, conv=True throughout.  Reseeding from the previous good
    density is not always enough, so this goes back to RKS and follows the
    external instability again, which is constructive rather than a guess.

    Returns the converged newton object, or None.
    """
    mf_r = dft.RKS(mol)
    mf_r.xc = XC; mf_r.grids.level = 3
    mf_r.max_cycle = 300; mf_r.conv_tol = 1e-10; mf_r.max_memory = mem
    mf_r.kernel()
    if not mf_r.converged:
        return None
    _, mo_ext, _, ext_stable = mf_r.stability(internal=True, external=True,
                                              return_status=True)
    if ext_stable:
        return None                     # no BS solution exists here any more
    mf_s = mf_r.to_uks()                # to_uks(), not a fresh UKS: mo_occ
    mf_s.xc = XC; mf_s.grids.level = 3
    mf_s.max_cycle = 300; mf_s.conv_tol = 1e-10; mf_s.max_memory = mem
    n = mf_s.newton(); n.max_cycle = 200; n.conv_tol = 1e-10
    n.kernel(mf_s.make_rdm1(mo_ext, mf_s.mo_occ))
    return n if n.converged else None


class BSUKS(dft.uks.UKS):
    """UKS whose kernel() reconverges the broken-symmetry solution with
    second-order Newton, seeded from the previous geometry's BS orbitals.

    Must be a CLASS-level method: nuc_grad_method().as_scanner() builds its
    self.base as a *copy* of the SCF object.  An instance-attribute kernel whose
    closure wrote to the original object left the copy's mo_coeff at None, which
    crashed pyscf/grad/uhf.py:46 on the first geometry step (jobs 10679114 /
    10679457).  As a bound method `self` is whichever copy invoked it.
    """

    def kernel(self, dm0=None, **kwargs):
        if BS['in_newton']:
            return super().kernel(dm0=dm0, **kwargs)
        BS['in_newton'] = True
        try:
            n = self.newton()
            n.max_cycle = 200
            n.conv_tol  = 1e-10
            # Seed with the previous geometry's AO DENSITY MATRIX, never with
            # its mo_coeff.  MOs converged at geometry A are orthonormal w.r.t.
            # S(A); handing them to newton() at geometry B, which assumes they
            # are orthonormal there, corrupts the density.  Measured on H2O for
            # a 0.15 A step: nelec 10.000 -> 10.056, E 654 meV too low, <S^2>
            # negative, max|C^T S(B) C - I| = 0.166.  This invalidated the whole
            # of job 10682479 from the first real optimizer step onwards.
            # A density matrix is basis-consistent: PySCF rebuilds Fock and
            # diagonalises, yielding properly orthonormal MOs while preserving
            # the spin polarisation that defines the BS solution.
            n.kernel(dm0=BS['dm'])

            # ---- v2: recover from a branch jump --------------------------
            s2_try = float(n.spin_square()[0])
            if s2_try < S2_MIN and BS['n_recover'] < MAX_RECOVER:
                e_bad = float(n.e_tot)
                print(f'  !! S2={s2_try:.4f} at step {BS["step"]+1} -- '
                      f'attempting recovery', flush=True)
                fixed, how = None, None
                # (a) cheap: reseed from the last density that was still BS
                if BS['last_good_dm'] is not None:
                    n_a = self.newton(); n_a.max_cycle = 200; n_a.conv_tol = 1e-10
                    n_a.kernel(dm0=BS['last_good_dm'])
                    if n_a.converged and float(n_a.spin_square()[0]) >= S2_MIN:
                        fixed, how = n_a, 'last_good_dm'
                # (b) expensive: back to RKS and follow the instability again
                if fixed is None:
                    n_b = rederive_bs(self.mol, self.max_memory)
                    if n_b is not None and float(n_b.spin_square()[0]) >= S2_MIN:
                        fixed, how = n_b, 'rederive'
                BS['n_recover'] += 1
                if fixed is not None:
                    n = fixed
                    print(f'  ++ recovered via {how}: E={n.e_tot:.10f}  '
                          f'S2={float(n.spin_square()[0]):.4f}  '
                          f'dE={(float(n.e_tot)-e_bad)*27211.386:+.1f} meV',
                          flush=True)
                else:
                    print('  -- recovery failed, continuing on the collapsed '
                          'branch', flush=True)
                BS['recoveries'].append(
                    {'step': BS['step'] + 1, 's2_before': round(s2_try, 6),
                     'method': how, 'ok': fixed is not None,
                     'e_before': round(e_bad, 10),
                     'e_after': round(float(n.e_tot), 10)})

            self.e_tot     = n.e_tot
            self.mo_coeff  = n.mo_coeff
            self.mo_occ    = n.mo_occ
            self.mo_energy = n.mo_energy
            self.converged = n.converged

            dm = n.make_rdm1()
            BS['dm'] = dm
            BS['step'] += 1
            s2 = float(n.spin_square()[0])
            if s2 >= S2_MIN:
                BS['last_good_dm'] = dm

            # guard: a valid density integrates to the exact electron count
            nel = float(np.einsum('sij,ji->', np.asarray(dm), self.get_ovlp()))
            nel_err = abs(nel - self.mol.nelectron)
            if nel_err > 1e-4 and BS['bad_nelec_at'] is None:
                BS['bad_nelec_at'] = BS['step']
                print(f'  *** BAD_DENSITY: nelec={nel:.6f} vs '
                      f'{self.mol.nelectron} at geom step {BS["step"]} ***',
                      flush=True)

            BS['log'].append({'step': BS['step'], 'e': float(n.e_tot),
                              's2': round(s2, 6), 'conv': bool(n.converged),
                              'nelec_err': round(nel_err, 8)})
            if s2 < S2_MIN and BS['lost_at'] is None:
                BS['lost_at'] = BS['step']
                print(f'  *** BS_LOST: S2={s2:.4f} < {S2_MIN} at geom step '
                      f'{BS["step"]} ***', flush=True)
            print(f'  geom {BS["step"]:3d}: E={n.e_tot:.10f}  S2={s2:.4f}'
                  f'  nelec_err={nel_err:.2e}  conv={n.converged}', flush=True)
            if BS['on_step']:
                BS['on_step']()
        finally:
            BS['in_newton'] = False
        return self.e_tot


# -- helpers -----------------------------------------------------------------

def make_uks(mol, max_mem, spin=0):
    mf = dft.UKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = max_mem
    return mf


def make_bsuks(mol, max_mem):
    mf = BSUKS(mol)
    mf.xc = XC; mf.grids.level = 3
    mf.max_cycle = 300; mf.conv_tol = 1e-10; mf.max_memory = max_mem
    return mf


def xyz_to_mol(xyz_path, spin=0, max_memory=160000):
    atoms = ase_read(xyz_path)
    atom_str = '\n'.join(
        f'{s} {x:.6f} {y:.6f} {z:.6f}'
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()))
    mol = gto.Mole()
    mol.atom = atom_str; mol.basis = BASIS; mol.spin = spin
    mol.charge = 0; mol.verbose = 4; mol.max_memory = max_memory
    mol.build()
    return mol


def kabsch_rmsd(P, Q):
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(np.mean(np.sum((Pc @ R.T - Qc) ** 2, axis=1))))


def get_all_bonds(atoms, scale=1.3):
    nums = atoms.get_atomic_numbers(); pos = atoms.get_positions()
    bonds = {}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < scale * (covalent_radii[nums[i]] + covalent_radii[nums[j]]):
                bonds[(i, j)] = float(d)
    return bonds


def reactive_bonds(rxn, syms):
    """Top-2 pairs by |d_P - d_R| -- same rule as ~/_rxn_coord_full.py."""
    r = ase_read(f'{HOME}/orca_neb_results/{rxn}/reactant.xyz')
    p = ase_read(f'{HOME}/orca_neb_results/{rxn}/product.xyz')
    br, bp = get_all_bonds(r), get_all_bonds(p)
    rp, pp = r.get_positions(), p.get_positions()
    deltas = []
    for (i, j) in set(br) | set(bp):
        dr = br.get((i, j), float(np.linalg.norm(rp[i] - rp[j])))
        dp = bp.get((i, j), float(np.linalg.norm(pp[i] - pp[j])))
        deltas.append(((i, j), dr, dp, abs(dp - dr)))
    deltas.sort(key=lambda x: -x[3])
    return [{'pair': [int(i), int(j)], 'sym': f'{syms[i]}{i}-{syms[j]}{j}',
             'd_R': round(dr, 4), 'd_P': round(dp, 4)}
            for (i, j), dr, dp, _ in deltas[:2]]


def get_t1x_ts(rxn_id, zref):
    """Transition1x TS. positions has a leading singleton axis -> [0]."""
    try:
        with h5py.File(T1X_H5, 'r') as f:
            # 'data' is a mirror of the split groups (verified byte-identical);
            # prefer the real split label so provenance is explicit.
            splits = [s for s in f.keys() if s != 'data'] + \
                     [s for s in f.keys() if s == 'data']
            for split in splits:
                g = f[split]
                if not hasattr(g, 'keys'):
                    continue
                for formula in g.keys():
                    if rxn_id in g[formula]:
                        node = g[formula][rxn_id]['transition_state']
                        pos = np.array(node['positions'])[0]
                        z   = np.array(node['atomic_numbers'])
                        if not np.array_equal(z, zref):
                            return None, f'atom_order_mismatch({split})'
                        return pos, split
    except Exception as e:
        return None, f'h5_error:{e}'
    return None, 'not_found'


def mulliken_spin(mol, mf_u, thr=0.05):
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
            for k in order[:2] if abs(sp[k]) > thr]


# -- main --------------------------------------------------------------------

def main(rxn):
    out = f'{OUTDIR}/{rxn}'
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    max_mem = int(os.environ.get('PYSCF_MAX_MEMORY', 160000))
    ts_xyz = f'{HOME}/orca_neb_results/{rxn}/transition_state.xyz'

    print('=' * 70, flush=True)
    print(f'BS-UKS TS opt  {rxn}  {XC}/{BASIS}', flush=True)

    ref_atoms = ase_read(ts_xyz)
    zref  = ref_atoms.get_atomic_numbers()
    syms  = ref_atoms.get_chemical_symbols()
    rksp  = ref_atoms.get_positions()
    rbonds = reactive_bonds(rxn, syms)
    print(f'  reactive bonds: {[b["sym"] for b in rbonds]}', flush=True)

    result = {'rxn': rxn, 'method': f'{XC}/{BASIS} BS-UKS TS opt',
              'status': 'RUNNING', 'reactive_bonds': rbonds}

    def save():
        with open(f'{out}/result.json', 'w') as fh:
            json.dump(result, fh, indent=2)

    save()

    # -- Step 0: RKS + stability + BS ---------------------------------------
    print('\n--- Step 0: RKS + stability + BS ---', flush=True)
    mol = xyz_to_mol(ts_xyz, max_memory=max_mem)

    mf_rks = dft.RKS(mol)
    mf_rks.xc = XC; mf_rks.grids.level = 3
    mf_rks.max_cycle = 300; mf_rks.conv_tol = 1e-10; mf_rks.max_memory = max_mem
    mf_rks.kernel()
    if not mf_rks.converged:
        result['status'] = 'RKS_NOT_CONVERGED'; save(); return
    e_rks = float(mf_rks.e_tot)
    result['e_rks'] = round(e_rks, 10)
    print(f'  RKS E = {e_rks:.10f} Ha', flush=True)

    _, mo_ext, int_stable, ext_stable = mf_rks.stability(
        internal=True, external=True, return_status=True)
    result['int_stable'] = bool(int_stable)
    result['ext_stable'] = bool(ext_stable)
    print(f'  int_stable={int_stable}  ext_stable={ext_stable}', flush=True)

    # Route 1: follow the external instability eigenvector.
    # NB: must be mf_rks.to_uks(), NOT a fresh UKS -- to_uks() converts the
    # *converged* RKS object and carries mo_occ over.  A blank dft.UKS(mol) has
    # mo_occ=None, which makes make_rdm1 raise 'NoneType' is not subscriptable
    # (this killed job 10679833).
    mf_s = mf_rks.to_uks()
    mf_s.xc = XC; mf_s.grids.level = 3
    mf_s.max_cycle = 300; mf_s.conv_tol = 1e-10; mf_s.max_memory = max_mem
    n0 = mf_s.newton(); n0.max_cycle = 200; n0.conv_tol = 1e-10
    n0.kernel(mf_s.make_rdm1(mo_ext, mf_s.mo_occ))
    s2_0 = float(n0.spin_square()[0])
    de_0 = (float(n0.e_tot) - e_rks) * 27211.386
    print(f'  Route1: E={n0.e_tot:.10f}  dE={de_0:.2f} meV  S2={s2_0:.4f}'
          f'  conv={n0.converged}', flush=True)
    route = 1
    best = n0

    # Route 2: triplet-seeded beta-HOMO flip, ONCE, only if route 1 collapsed
    if s2_0 < S2_MIN:
        print(f'  Route1 collapsed (S2={s2_0:.4f}); trying Route2 triplet seed',
              flush=True)
        try:
            mol_t = xyz_to_mol(ts_xyz, spin=2, max_memory=max_mem)
            mf_t = make_uks(mol_t, max_mem); mf_t.kernel()
            mo_a_t, mo_b_t = mf_t.mo_coeff
            nalpha_t = (mol_t.nelectron + 2) // 2
            nbs = mol.nelectron // 2
            mo_bs_a = mo_a_t.copy(); mo_bs_b = mo_b_t.copy()
            mo_bs_b[:, nbs - 1] = mo_a_t[:, nalpha_t - 1]
            mf_s2 = mf_rks.to_uks()          # same reason as Route 1: mo_occ
            mf_s2.xc = XC; mf_s2.grids.level = 3
            mf_s2.max_cycle = 300; mf_s2.conv_tol = 1e-10; mf_s2.max_memory = max_mem
            dm2 = mf_s2.make_rdm1(np.array([mo_bs_a, mo_bs_b]), mf_s2.mo_occ)
            n2 = mf_s2.newton(); n2.max_cycle = 200; n2.conv_tol = 1e-10
            n2.kernel(dm2)
            s2_2 = float(n2.spin_square()[0])
            de_2 = (float(n2.e_tot) - e_rks) * 27211.386
            print(f'  Route2: E={n2.e_tot:.10f}  dE={de_2:.2f} meV  S2={s2_2:.4f}'
                  f'  conv={n2.converged}', flush=True)
            if s2_2 > s2_0:
                best, s2_0, de_0, route = n2, s2_2, de_2, 2
        except Exception as exc:
            print(f'  Route2 FAILED: {exc}', flush=True)

    result['bs_initial'] = {'route': route, 'e_uks': round(float(best.e_tot), 10),
                            'de_meV': round(de_0, 3), 's2': round(s2_0, 6),
                            'converged': bool(best.converged)}
    save()

    if s2_0 < S2_MIN:
        result['status'] = 'COLLAPSED'
        result['note'] = (f'both routes collapsed: best S2={s2_0:.4f} < {S2_MIN}')
        result['elapsed_s'] = round(time.time() - t0, 1)
        save()
        print(f'\nCOLLAPSED -- stopping {rxn}', flush=True)
        return

    BS['dm'] = best.make_rdm1()   # AO-basis density, see BSUKS.kernel

    mf_u = make_bsuks(mol, max_mem)
    mf_u.mo_coeff, mf_u.mo_occ = best.mo_coeff, best.mo_occ
    mf_u.mo_energy = best.mo_energy
    mf_u.e_tot, mf_u.converged = float(best.e_tot), True

    # checkpoint every geometry step
    def on_step():
        result['n_geom_steps'] = BS['step']
        result['step_log'] = BS['log']
        result['bs_lost_at_step'] = BS['lost_at']
        result['recoveries'] = BS['recoveries']
        save()
    BS['on_step'] = on_step

    # -- Step 1: TS optimization --------------------------------------------
    print(f'\n--- Step 1: TS optimization (transition=True, maxsteps={MAXSTEPS}) ---',
          flush=True)
    t_opt = time.time()
    opt_success, opt_error = False, None
    try:
        # kernel() returns (converged, mol); optimize() throws the flag away and
        # returns mol only, so it reports success even when geomeTRIC printed
        # "Geometry optimization failed to converge in N iterations".  That made
        # every task in job 10682479 look converged when none was.
        opt_success, ts_mol = geometric_solver.kernel(
            mf_u, transition=True, maxsteps=MAXSTEPS)
        opt_success = bool(opt_success)
        print(f'  Optimization finished: converged={opt_success}', flush=True)
        if not opt_success:
            opt_error = f'did not converge within {MAXSTEPS} steps'
    except Exception as exc:
        import traceback
        opt_error = f'{type(exc).__name__}: {exc}'
        print(f'  Optimization FAILED: {opt_error}', flush=True)
        traceback.print_exc()
        ts_mol = mf_u.mol
    t_opt_s = round(time.time() - t_opt, 1)
    BS['on_step'] = None

    pos_ang = ts_mol.atom_coords() * BOHR
    s2_fin = BS['log'][-1]['s2'] if BS['log'] else None
    e_fin  = BS['log'][-1]['e']  if BS['log'] else None

    with open(f'{out}/ts_opt.xyz', 'w') as fh:
        fh.write(f'{ts_mol.natm}\n')
        fh.write(f'BS-UKS {XC}/{BASIS} TS opt  E={e_fin}  S2={s2_fin}  '
                 f'converged={opt_success}\n')
        for s, (x, y, z) in zip(syms, pos_ang):
            fh.write(f'{s}  {x:.8f}  {y:.8f}  {z:.8f}\n')

    if BS['bad_nelec_at'] is not None:
        status = 'BAD_DENSITY'
    elif BS['lost_at'] is not None:
        status = 'BS_LOST'
    elif opt_success:
        status = 'converged'
    else:
        status = 'not-converged'

    result.update({
        'status': status,
        'opt_converged': opt_success,
        'opt_error': opt_error,
        'bs_lost_at_step': BS['lost_at'],
        'bad_density_at_step': BS['bad_nelec_at'],
        'n_geom_steps': BS['step'],
        'step_log': BS['log'],
        'e_uks_final': round(e_fin, 10) if e_fin is not None else None,
        's2_final': s2_fin,
        'de_vs_rks_meV': (round((e_fin - e_rks) * 27211.386, 3)
                          if e_fin is not None else None),
        'elapsed_opt_s': t_opt_s,
    })
    save()

    # -- Step 3: RMSD + reactive bonds ---------------------------------------
    print('\n--- Step 3: RMSD vs references ---', flush=True)
    rmsd = {}
    r = kabsch_rmsd(pos_ang, rksp)
    rmsd['RKS_ref'] = round(r, 5)
    print(f'  RKS_ref: {r:.5f} A', flush=True)

    t1x_pos, t1x_info = get_t1x_ts(rxn, zref)
    if t1x_pos is not None:
        r = kabsch_rmsd(pos_ang, t1x_pos)
        rmsd['T1x'] = round(r, 5)
        print(f'  T1x ({t1x_info}): {r:.5f} A', flush=True)
    else:
        rmsd['T1x'] = None
        print(f'  T1x: {t1x_info}', flush=True)

    for name, sub in MODELS:
        p = f'{HOME}/{sub}/{rxn}/transition_state.xyz'
        if not os.path.exists(p):
            rmsd[name] = None
            print(f'  {name}: file not found', flush=True); continue
        a = ase_read(p)
        if not np.array_equal(a.get_atomic_numbers(), zref):
            rmsd[name] = None
            print(f'  {name}: atom order mismatch', flush=True); continue
        r = kabsch_rmsd(pos_ang, a.get_positions())
        rmsd[name] = round(r, 5)
        print(f'  {name}: {r:.5f} A', flush=True)

    bonds = {}
    for b in rbonds:
        i, j = b['pair']
        d_bs  = float(np.linalg.norm(pos_ang[i] - pos_ang[j]))
        d_rks = float(np.linalg.norm(rksp[i] - rksp[j]))
        bonds[b['sym']] = {'BS_UKS': round(d_bs, 4), 'RKS': round(d_rks, 4),
                           'delta': round(d_bs - d_rks, 4)}
        print(f'  {b["sym"]}: BS-UKS={d_bs:.4f}  RKS={d_rks:.4f}  '
              f'delta={d_bs-d_rks:+.4f} A', flush=True)

    sp = mulliken_spin(mol, mf_u)
    result.update({'rmsd': rmsd, 'bond_lengths': bonds,
                   'spin_populations_final': sp,
                   'elapsed_total_s': round(time.time() - t0, 1)})
    save()
    print(f'\n{rxn}: {status}  steps={BS["step"]}  '
          f'({result["elapsed_total_s"]:.0f}s)', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: bs_tsopt_batch.py <rxn>'); sys.exit(1)
    os.environ.setdefault('OMP_NUM_THREADS', '40')
    try:
        main(sys.argv[1])
    except Exception:
        import traceback; traceback.print_exc()
        sys.exit(1)
