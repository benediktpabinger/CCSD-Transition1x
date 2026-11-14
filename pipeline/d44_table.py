"""D44: barrier error caused by the model's saddle geometry, 30 MACE-OMol saddles per group.

Per selected structure (results/d44_selection.csv; run folder ~/geomfx/<rxn>, or the D32 pilot
folder when reused): geometry effect = E_DFT(model saddle, D37 EnGrad) - E_DFT(DFT-refined saddle,
last OptTS energy), meV, both gradient-type; energy error = model barrier - DFT barrier at the
model's saddle and reactant (results/d40_barriers.csv); RMSD between the saddles after Kabsch
alignment; OptTS cycles; stability rounds. Refined saddles more than 0.1 A away are reported
separately. Reading fixed in D44: the geometry effect is a result if its median absolute value
in the closed-shell high-MR or the broken-symmetry group exceeds the baseline, with the 95 %
bootstrap interval of the difference above 0.

Output results/d44_geometry_effect.csv

    python pipeline/d44_table.py
"""
import csv
import os
import re

import numpy as np
from ase.io import read

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
HA = 27211.386245988
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*(-?[\d.]+)')


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt((((a @ (u @ np.diag([1, 1, d]) @ vt)) - b) ** 2).sum(1).mean()))


def text(p):
    return open(p, errors='replace').read() if os.path.exists(p) else ''


def last(regex, t):
    v = regex.findall(t)
    return float(v[-1]) if v else None


bar = {r['rxn']: float(r['error_mev']) for r in csv.DictReader(open(f'{RES}/d40_barriers.csv')) if r['model'] == 'mace-omol'}
rows = []
for s in csv.DictReader(open(f'{RES}/d44_selection.csv')):
    rx = s['rxn']
    d = f'{H}/marks_pilot/{rx}' if s['source'] == 'pilot' else f'{H}/geomfx/{rx}'
    t_opt, t_stab = text(f'{d}/optts.out'), text(f'{d}/stab.out')
    t_ts = text(f'{H}/orca6_fsm_models/{rx}_mace-omol_ts/uks_engrad.out')
    rounds = text(f'{d}/stab_rounds').strip()
    out = dict(rxn=rx, group=s['group'], nfod=float(s['nfod']), source=s['source'],
               stab_rounds=int(rounds) if rounds else 0, optts_cycles=t_opt.count('GEOMETRY OPTIMIZATION CYCLE'))
    if 'THE OPTIMIZATION HAS CONVERGED' not in t_opt or 'ORCA TERMINATED NORMALLY' not in t_stab:
        out['status'] = 'not converged' if t_opt else 'no OptTS'
        if os.path.exists(f'{d}/optts.out') and 'ORCA TERMINATED NORMALLY' not in t_stab:
            out['status'] = 'stability run failed'
        rows.append(out)
        continue
    s2_ts, s2_ref = last(S2_RE, text(f'{H}/orca6_fsm_models/{rx}_mace-omol_ts/uks_sp.out')), last(S2_RE, t_opt)
    x0 = read(f'{H}/neb_fsm/fsm/mace-omol/{rx}/transition_state.xyz').positions
    x1 = read(f'{d}/optts.xyz').positions
    out.update(status='done', s2_model=s2_ts, s2_refined=s2_ref, rmsd_A=kabsch_rmsd(x0, x1),
               geometry_effect_mev=(last(E_RE, t_ts) - last(E_RE, t_opt)) * HA, energy_error_mev=bar.get(rx, np.nan))
    out['same_solution'] = int(abs(s2_ts - s2_ref) <= 0.1)
    rows.append(out)

keys = list(dict.fromkeys(k for r in rows for k in r))
with open(f'{RES}/d44_geometry_effect.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys, restval='')
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})

rng = np.random.default_rng(0)
print('%-9s %3s %5s %6s %6s %10s %8s %10s %6s' % ('group', 'n', 'used', 'moved', 'other', '|geom|med', '>43meV', '|energy|med', 'switch'))
G = {}
for g in ('baseline', 'cs_mr', 'bs'):
    R = [r for r in rows if r['group'] == g]
    ok = [r for r in R if r['status'] == 'done' and r['same_solution']]
    near = [r for r in ok if r['rmsd_A'] <= 0.1]
    G[g] = np.abs([r['geometry_effect_mev'] for r in near])
    print('%-9s %3d %5d %6d %6d %10.1f %7.0f%% %10.1f %6d' % (
        g, len(R), len(near), len(ok) - len(near), len(R) - len(ok), np.median(G[g]),
        100 * np.mean(np.abs([r['geometry_effect_mev'] for r in ok]) > 43),
        np.nanmedian(np.abs([r['energy_error_mev'] for r in ok])), sum(r['stab_rounds'] > 0 for r in R)))
for g in ('cs_mr', 'bs'):
    diff = [np.median(rng.choice(G[g], len(G[g]))) - np.median(rng.choice(G['baseline'], len(G['baseline']))) for _ in range(2000)]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f'{g} minus baseline, median |geometry effect|: {np.median(G[g]) - np.median(G["baseline"]):.1f} meV [{lo:.1f}, {hi:.1f}]'
          f' -> {"result" if lo > 0 else "no result"}')
print('not usable:', [(r['rxn'], r['group'], r['status']) for r in rows if r['status'] != 'done'])
