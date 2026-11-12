"""D42: how much barrier error does the model's saddle geometry cause? DFT refinement pilot only.

For every pilot run (results/marks_pilot.csv) whose OptTS converged on a consistent solution
with exactly one imaginary mode:
  geometry effect  E_DFT(model saddle) - E_DFT(refined saddle), meV; same reactant, so this is
                   the barrier error caused by the geometry. E_DFT(model saddle): EnGrad on the
                   UKS orbitals at the MACE-OMol freezing-string structure (D37); E_DFT(refined):
                   last OptTS energy. Both gradient-type (SCNL), D36.
  energy error     model barrier - DFT barrier at the model's own saddle and reactant (D40), meV
  displacement     RMSD after Kabsch alignment between the two saddles, and OptTS cycles
Reading fixed in D42: if |geometry effect| < 15 meV in every usable run, it is reported as a
bound in the supporting information (n, MACE-OMol only); otherwise it is dropped.

Output results/geometry_effect.csv

    python pipeline/geometry_effect.py
"""
import csv
import json
import os
import re

import numpy as np
from ase.io import read

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
HA = 27211.386245988      # meV
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*(-?[\d.]+)')


def last(regex, p):
    t = open(p, errors='replace').read() if os.path.exists(p) else ''
    v = regex.findall(t)
    return float(v[-1]) if v else None


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt((((a @ r) - b) ** 2).sum(1).mean()))


rows = []
for r in csv.DictReader(open(f'{RES}/marks_pilot.csv')):
    if r['outcome'] in ('pending', 'technical', 'solution_switch', 'dft_not_converged', 'ends_pending'):
        continue
    if r['n_imag'] != '1':
        continue
    d = f"{H}/marks_pilot/{r['run']}"
    rx = r['rxn']
    ts_dir = f'{H}/orca6_fsm_models/{rx}_mace-omol_ts'
    e_model_ts = last(E_RE, f'{ts_dir}/uks_engrad.out')
    e_ref = last(E_RE, f'{d}/optts.out')
    e_r = last(E_RE, f'{H}/orca6_fsm_models/{rx}_mace-omol_r/uks_engrad.out')
    s2_model, s2_ref = last(S2_RE, f'{ts_dir}/uks_sp.out'), last(S2_RE, f'{d}/optts.out')
    x_model = read(f'{H}/neb_fsm/fsm/mace-omol/{rx}/transition_state.xyz').positions
    x_pilot = read(f'{d}/ts_model.xyz').positions
    assert np.abs(x_model - x_pilot).max() < 1e-4, f'{rx}: pilot start is not the D37 structure'
    x_ref = read(f'{d}/optts.xyz').positions
    res = json.load(open(f'{H}/neb_fsm/fsm/mace-omol/{rx}/result.json'))
    out = dict(run=r['run'], kind=r['kind'], group=r['group'], nfod=float(r['nfod']), outcome=r['outcome'],
               optts_cycles=int(r['optts_cycles']), rmsd_A=kabsch_rmsd(x_model, x_ref),
               s2_model=s2_model, s2_refined=s2_ref,
               geometry_effect_mev=(e_model_ts - e_ref) * HA if e_model_ts and e_ref else np.nan,
               energy_error_mev=(res['barrier_ev'] * 1000 - (e_model_ts - e_r) * HA) if e_model_ts and e_r else np.nan)
    out['same_solution'] = int(s2_model is not None and s2_ref is not None and abs(s2_model - s2_ref) <= 0.1)
    rows.append(out)

with open(f'{RES}/geometry_effect.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
print('%-18s %-8s %-6s %5s %6s %7s %7s %9s %9s' % ('run', 'kind', 'group', 'cyc', 'rmsd', 's2_mod', 's2_ref', 'geom meV', 'energy meV'))
for r in rows:
    print('%-18s %-8s %-6s %5d %6.3f %7.3f %7.3f %9.1f %9.1f' % (r['run'], r['kind'], r['group'], r['optts_cycles'], r['rmsd_A'],
          r['s2_model'] or np.nan, r['s2_refined'] or np.nan, r['geometry_effect_mev'], r['energy_error_mev']))
g = np.array([r['geometry_effect_mev'] for r in rows if r['same_solution']])
e = np.array([r['energy_error_mev'] for r in rows if r['same_solution']])
print(f'\nusable (same solution): {len(g)}; |geometry effect| median {np.nanmedian(np.abs(g)):.1f} meV, max {np.nanmax(np.abs(g)):.1f}; '
      f'|energy error| median {np.nanmedian(np.abs(e)):.1f} meV, max {np.nanmax(np.abs(e)):.1f}')
print('D42 reading:', 'clear-cut, report as a bound' if np.nanmax(np.abs(g)) < 15 else 'not clear-cut, drop or extend the pilot')
