"""Are the lower pilot barriers transition states of the reaction? (decision D23)

Takes every pilot run pair where both starts met the criterion and the interpolated
start gave a barrier at least 43 meV lower (results/pilot_runs.csv). Both structures
of each pair are checked on the model's own surface, in double precision:

  Hessian   as in ts_hessian_scan.py. Negative means below -0.05 eV/A^2.
  Downhill  only with exactly one negative mode: displace 0.15 A along it (largest
            atom), both ways, BFGS with steps of at most 0.05 A to 0.02 eV/A.
            Each end is compared with the reactant and product of the run by bond
            graph (bond if d < 1.15 * sum of covalent radii, atom order fixed; H-H
            bond if d < 0.95 A, since 1.15 * 2 * 0.31 A = 0.71 A misses H2 at 0.74 A).
            Both downhill runs must converge (D35). End structures, their energies
            and the bonds by which each differs from reactant and product are stored.

Class of a structure:
  route      one negative mode, one side is the reactant, the other the product
  one_end    one negative mode, only one side is an endpoint (the other is new)
  same_end   one negative mode, both sides are the same endpoint
  elsewhere  one negative mode, neither side is an endpoint
  unconverged  one negative mode, a downhill run hit the step limit (not a pass)
  no_mode    no negative mode
  multi      two or more negative modes

A run pair is a verified lower route if its interpolated structure is class route.
A reaction is verified start-sensitive if at least 2 models have one, the D22 rule
with this extra condition. The D22 count stays the main test and is reported next
to this one. Limits: the downhill check is a relaxation, not an IRC, so it can fall
into a neighbouring basin, and it says nothing about DFT.

Output  ~/neb_pilot/check/<model>/<rxn>_<start>.json, results/pilot_check.csv

    python pipeline/pilot_check.py --list
    python pipeline/pilot_check.py --model esen
    python pipeline/pilot_check.py --table
"""
import argparse
import csv
import json
import os
import sys
from math import comb

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize.bfgs import BFGS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
GAP = 0.043
NEG_TOL = -0.05
HH_CUT = 0.95


def bonds(Z, x):
    a = Atoms(numbers=Z, positions=x)
    nl = NeighborList([1.15 * c for c in natural_cutoffs(a)], self_interaction=False, bothways=False, skin=0)
    nl.update(a)
    b = {(min(i, int(j)), max(i, int(j))) for i in range(len(a)) for j in nl.get_neighbors(i)[0]}
    hh = [i for i, z in enumerate(Z) if z == 1]
    b |= {(i, j) for k, i in enumerate(hh) for j in hh[k + 1:] if np.linalg.norm(x[i] - x[j]) < HH_CUT}
    return b


def classify(n_neg, ends, conv=(1, 1)):
    if n_neg == 0:
        return 'no_mode'
    if n_neg > 1:
        return 'multi'
    if not all(conv):
        return 'unconverged'
    if set(ends) == {'R', 'P'}:
        return 'route'
    if ends[0] == ends[1] and ends[0] in 'RP':
        return 'same_end'
    return 'one_end' if set(ends) & {'R', 'P'} else 'elsewhere'


def check(c, factory, root):
    from ts_hessian_scan import hessian
    d = f"{root}/{c['start']}/{c['model']}/{c['rxn']}"
    ts, R, P = read(f'{d}/transition_state.xyz'), read(f'{d}/reactant.xyz'), read(f'{d}/product.xyz')
    stored = json.load(open(f'{d}/result.json'))
    Z, x = ts.numbers, ts.positions.copy()
    bR, bP = bonds(Z, R.positions), bonds(Z, P.positions)
    w, v = hessian(factory(), Z, x)
    n_neg = int((w < NEG_TOL).sum())
    out = dict(c, n_negative=n_neg, w1=float(w[0]), w2=float(w[1]), f_band=stored['f_band_final'],
               ts_minus_p_ev=stored['e_ts_ev'] - stored['e_p_ev'], barrier_ev=stored['barrier_ev'],
               n_rp_diff=len(bR ^ bP), end_a='', end_b='', steps_a=0, steps_b=0, conv_a=0, conv_b=0)
    ends = ['', '']
    if n_neg == 1:
        mode = v[:, 0].reshape(-1, 3)
        for k, (sign, tag) in enumerate(((1, 'a'), (-1, 'b'))):
            a = Atoms(numbers=Z, positions=x + sign * 0.15 * mode / np.abs(mode).max(), info={'charge': 0, 'spin': 1})
            a.calc = factory()
            opt = BFGS(a, logfile=None, maxstep=0.05)
            out[f'conv_{tag}'] = int(opt.run(fmax=0.02, steps=2000))
            out[f'steps_{tag}'] = opt.get_number_of_steps()
            b = bonds(Z, a.positions)
            ends[k] = 'R' if b == bR else 'P' if b == bP else 'X'
            out[f'end_{tag}'] = ends[k]
            out[f'e_end_{tag}_ev'] = float(a.get_potential_energy())
            out[f'diff_r_{tag}'] = sorted(b ^ bR)     # bonds made or broken against the reactant
            out[f'diff_p_{tag}'] = sorted(b ^ bP)
            out[f'pos_{tag}'] = a.positions.round(5).tolist()
    out['cls'] = classify(n_neg, ends, (out['conv_a'], out['conv_b']) if n_neg == 1 else (1, 1))
    return out


def table(outdir):
    rows = []
    for c in csv.DictReader(open(f'{outdir}/cases.csv')):
        p = f"{outdir}/{c['model']}/{c['rxn']}_{c['start']}.json"
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    n = sum(1 for _ in csv.DictReader(open(f'{outdir}/cases.csv')))
    print('%d of %d structures checked' % (len(rows), n))
    cls = ('route', 'one_end', 'same_end', 'elsewhere', 'unconverged', 'no_mode', 'multi')
    for start in ('idpp', 't1x'):
        for g in ('bs', 'closed'):
            S = [r for r in rows if r['start'] == start and r['group'] == g]
            print('%-5s %-7s %3d ' % (start, g, len(S)), '  '.join('%s %d' % (k, sum(r['cls'] == k for r in S)) for k in cls))
    print('structures with a downhill run that hit the step limit: %d' % sum(
        r['cls'] != 'no_mode' and r['n_negative'] == 1 and not (r['conv_a'] and r['conv_b']) for r in rows))
    print('interpolated structures within 50 meV of the product: %d, of those route %d' % (
        sum(r['start'] == 'idpp' and abs(r['ts_minus_p_ev']) <= 0.05 for r in rows),
        sum(r['start'] == 'idpp' and abs(r['ts_minus_p_ev']) <= 0.05 and r['cls'] == 'route' for r in rows)))
    by = {(r['rxn'], r['model'], r['start']): r for r in rows}
    ok = {k[:2] for k, r in by.items() if k[2] == 'idpp' and r['cls'] == 'route'}
    both = {k for k in ok if by.get((*k, 't1x'), {}).get('cls') == 'route'}
    print('verified lower routes %d of %d run pairs, %d with the Transition1x structure also a route' % (
        len(ok), len({k[:2] for k in by if k[2] == 'idpp'}), len(both)))
    pairs = list(csv.DictReader(open(f'{RES}/pilot_pairs.csv')))
    status = {r['rxn']: r['status'] for r in csv.DictReader(open(f'{RES}/pilot_reactions.csv'))}
    nok = {}
    for rx, m in ok:
        nok[rx] = nok.get(rx, 0) + 1
    done = [p for p in pairs if status[p['rxn_bs']] != 'pending' and status[p['rxn_cs']] != 'pending'
            and status[p['rxn_bs']] != 'unresolved' and status[p['rxn_cs']] != 'unresolved']
    sens = lambda rx: nok.get(rx, 0) >= 2
    b = sum(sens(p['rxn_bs']) and not sens(p['rxn_cs']) for p in done)
    c = sum(sens(p['rxn_cs']) and not sens(p['rxn_bs']) for p in done)
    m = b + c
    pv = min(1.0, 2 * sum(comb(m, k) for k in range(0, min(b, c) + 1)) / 2 ** m) if m else 1.0
    print('complete pairs %d: verified start-sensitive BS %d, closed-shell %d; only BS %d, only closed %d; exact McNemar p = %.4f'
          % (len(done), sum(sens(p['rxn_bs']) for p in done), sum(sens(p['rxn_cs']) for p in done), b, c, pv))
    keys = list(dict.fromkeys(k for r in rows for k in r if not k.startswith('pos_')))
    with open(f'{RES}/pilot_check.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval='', extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('%.5f' % v if isinstance(v, float) else ' '.join('%d-%d' % t for t in v)
                            if isinstance(v, list) else v) for k, v in r.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--table', action='store_true')
    ap.add_argument('--model')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--root', default=f'{H}/neb_pilot')
    ap.add_argument('--outdir', default=f'{H}/neb_pilot/check')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.list:
        L = [r for r in csv.DictReader(open(f'{RES}/pilot_runs.csv')) if r['met_t1x'] == '1' and r['met_idpp'] == '1'
             and float(r['barrier_idpp']) - float(r['barrier_t1x']) <= -GAP]
        with open(f'{a.outdir}/cases.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['rxn', 'group', 'bin', 'model', 'start'])
            for r in L:
                for start in ('idpp', 't1x'):
                    w.writerow([r['rxn'], r['group'], r['bin'], r['model'], start])
        print('%d lower run pairs, %d structures' % (len(L), 2 * len(L)))
        raise SystemExit(0)
    if a.table:
        table(a.outdir)
        raise SystemExit(0)

    from omol_neb import make_calc_factory
    JOB = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    cases = [c for c in csv.DictReader(open(f'{a.outdir}/cases.csv')) if c['model'] == a.model]
    os.makedirs(f'{a.outdir}/{a.model}', exist_ok=True)
    todo = [c for c in cases if not os.path.exists(f"{a.outdir}/{a.model}/{c['rxn']}_{c['start']}.json")]
    print(f'{a.model}: {len(todo)} of {len(cases)} structures to do, OMP {os.environ.get("OMP_NUM_THREADS")}', flush=True)
    if todo:
        factory = make_calc_factory(a.model, a.device, 'float64')
    for c in todo:
        p = f"{a.outdir}/{a.model}/{c['rxn']}_{c['start']}"
        try:                                # workers of a job share the list, as in omol_neb.py
            os.close(os.open(f'{p}.claim.{JOB}', os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        r = check(c, factory, a.root)
        with open(f'{p}.tmp', 'w') as fh:
            json.dump(r, fh, indent=1)
        os.replace(f'{p}.tmp', f'{p}.json')
        print('  %s %-4s %-7s %-9s negative %d  lowest %.3f  ends %s%s  steps %d %d'
              % (c['rxn'], c['start'], c['group'], r['cls'], r['n_negative'], r['w1'], r['end_a'] or '-',
                 r['end_b'] or '-', r['steps_a'], r['steps_b']), flush=True)
    print('END', a.model, flush=True)
