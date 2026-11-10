"""D37: model force error against DFT at the freezing-string structures of all four models.

For every converged freezing-string run of the 510 reactions (D24, D29) and every model:
model forces at the refined structure (saved as model_forces.npy next to the run, as in
D28), DFT forces on the lowest UKS solution (pipeline/job_border_zone.sh, EnGrad on the UKS
orbitals), both in the frame of transition_state.xyz. Force measures as fixed in D34:
f_err_mae (mean absolute component error, the main measure), f_err_max (largest per-atom
error norm), f_dft (largest per-atom DFT force norm, against 0.05 eV/A).

Reading fixed in D37: per model, ratio of median f_err_mae, high-MR group (N_FOD >= 1) and
the closed-shell reactions with N_FOD >= 0.5 against the baseline, 95 % interval from 2000
bootstrap resamples over reactions; the MR effect is general if the interval excludes 1 on
at least three of four models. Broken-symmetry against twin: paired median ratio over pairs
with both structures.

    python pipeline/d37_table.py --model esen      # model forces (SLURM, pipeline/job_d37_forces.sh)
    python pipeline/d37_table.py --table
Output results/d37_forces.csv, results/d37_summary.csv, results/d37_forces_missing.csv
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np
from ase import Atoms
from ase.io import read

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
EVA = 51.42208
M = ('esen', 'uma-s', 'uma-m', 'mace-omol')
S = {r['rxn']: r for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{RES}/{n}.csv'))}


def runs(m):
    for rx, r in S.items():
        d = f'{H}/neb_fsm/fsm/{m}/{rx}'
        if os.path.exists(f'{d}/result.json') and json.load(open(f'{d}/result.json'))['criterion_met']:
            dft = f'{H}/orca6_fsm/{rx}_ts' if m == 'esen' else f'{H}/orca6_fsm_models/{rx}_{m}_ts'
            yield rx, r, d, dft


def dft_forces(p):
    t = open(p, errors='replace').read() if os.path.exists(p) else ''
    if 'CARTESIAN GRADIENT' not in t:
        return None
    G = []
    for line in t[t.rfind('CARTESIAN GRADIENT'):].split('\n')[3:]:
        f = line.split()
        if len(f) < 6:
            break
        try:
            G.append([float(v) for v in f[3:6]])
        except ValueError:
            break
    return -np.array(G) * EVA


def model_forces(m):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from omol_neb import make_calc_factory
    calc = make_calc_factory(m, 'cpu')()
    n = 0
    for rx, r, d, dft in runs(m):
        if os.path.exists(f'{d}/model_forces.npy'):
            continue
        x = read(f'{d}/transition_state.xyz')
        a = Atoms(numbers=x.numbers, positions=x.positions, info={'charge': 0, 'spin': 1})
        a.calc = calc
        np.save(f'{d}/model_forces.npy', a.get_forces())
        n += 1
    print(m, n, 'new model force files', flush=True)


def boot_ratio(a, b, rng, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    r = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b))) for _ in range(n)]
    return np.median(a) / np.median(b), *np.percentile(r, [2.5, 97.5])


def table():
    rows, missing = [], []
    for m in M:
        for rx, r, d, dft in runs(m):
            F = dft_forces(f'{dft}/uks_engrad.out')
            if F is None or not os.path.exists(f'{d}/model_forces.npy'):
                missing.append(dict(model=m, rxn=rx, group=r['group'],
                                    reason='no DFT gradient' if F is None else 'no model forces'))
                continue
            Fm = np.load(f'{d}/model_forces.npy')
            assert F.shape == Fm.shape, (m, rx)
            s2 = re.findall(r'Expectation value of <S\*\*2>\s*:\s*(-?[\d.]+)', open(f'{dft}/uks_sp.out', errors='replace').read())
            rows.append(dict(model=m, rxn=rx, group=r['group'], partner=r['partner'], nfod=float(r['nfod']),
                             s2=float(s2[-1]) if s2 else np.nan, f_dft=float(np.sqrt((F ** 2).sum(1)).max()),
                             f_model=float(np.sqrt((Fm ** 2).sum(1)).max()),
                             f_err_max=float(np.sqrt(((F - Fm) ** 2).sum(1)).max()), f_err_mae=float(np.abs(F - Fm).mean())))
    for name, R in (('d37_forces', rows), ('d37_forces_missing', missing)):
        with open(f'{RES}/{name}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(R[0].keys()) if R else ['model', 'rxn', 'group', 'reason'])
            w.writeheader()
            for r in R:
                w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
    print(len(rows), 'structures; missing', len(missing), [(x['model'], x['rxn'], x['group'], x['reason']) for x in missing])
    rng = np.random.default_rng(0)
    summ = []
    for m in M:
        R = [r for r in rows if r['model'] == m]
        e = lambda sel: [r['f_err_mae'] for r in R if sel(r)]   # noqa: E731
        base = e(lambda r: r['group'] == 'baseline')
        out = dict(model=m, n=len(R), base_med=np.median(base))
        for k, sel in (('high', lambda r: r['group'] == 'high'),
                       ('cs_mr', lambda r: r['group'] == 'closed' and r['nfod'] >= 0.5),
                       ('bs', lambda r: r['group'] == 'bs')):
            v = e(sel)
            ratio, lo, hi = boot_ratio(v, base, rng)
            out.update({f'{k}_n': len(v), f'{k}_med': np.median(v), f'{k}_ratio': ratio, f'{k}_lo': lo, f'{k}_hi': hi})
        by = {r['rxn']: r['f_err_mae'] for r in R}
        p = [by[r['rxn']] / by[r['partner']] for r in R if r['group'] == 'bs' and r['partner'] in by]
        pr = [np.median(rng.choice(p, len(p))) for _ in range(2000)]
        out.update(bs_pairs=len(p), bs_twin_ratio=np.median(p), bs_twin_lo=np.percentile(pr, 2.5), bs_twin_hi=np.percentile(pr, 97.5))
        for g in ('baseline', 'closed', 'bs', 'high'):
            fd = [r['f_dft'] for r in R if r['group'] == g]
            out[f'{g}_dft_above_005'] = np.mean(np.array(fd) > 0.05) if fd else np.nan
        summ.append(out)
        print('%-9s n=%3d  baseline %.4f | high %.4f x%.2f [%.2f, %.2f] | CS N_FOD>=0.5 x%.2f [%.2f, %.2f] | BS x%.2f [%.2f, %.2f] | BS/twin x%.2f [%.2f, %.2f] (%d pairs)' % (
            m, len(R), out['base_med'], out['high_med'], out['high_ratio'], out['high_lo'], out['high_hi'],
            out['cs_mr_ratio'], out['cs_mr_lo'], out['cs_mr_hi'], out['bs_ratio'], out['bs_lo'], out['bs_hi'],
            out['bs_twin_ratio'], out['bs_twin_lo'], out['bs_twin_hi'], len(p)))
    general = sum(s['high_lo'] > 1 for s in summ)
    print(f'MR effect (high vs baseline) interval above 1 on {general} of 4 models -> general by the D37 rule: {general >= 3}')
    with open(f'{RES}/d37_summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        for s in summ:
            w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--table', action='store_true')
    a = ap.parse_args()
    if a.model:
        model_forces(a.model)
    if a.table:
        table()
