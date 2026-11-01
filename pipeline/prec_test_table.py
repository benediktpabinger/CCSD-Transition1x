"""Why do repeated NEBs of the same model and reaction end at different
transition states? Evaluation of the numerics test (docs/paper_decisions.md D21).

Compares the transition states of paired runs at 6 reactions (rxn0346, rxn4113,
rxn1147, rxn4518 broken-symmetry, rxn0101, rxn0000 closed-shell):

    sp_cpu_gpu         single precision, CPU against GPU (the original mismatch)
    dbl_cpu_gpu        double precision, CPU against GPU
    tight_cpu_gpu      both bands continued to f = 0.01 eV/A on CPU, started from
                       the CPU and the GPU band
    stop_shift_cpu     CPU band at 0.05 against the same band continued to 0.01:
                       how far the 0.05 stopping point is from the tighter one
    mace_default_cpu_gpu, mace_float32_cpu_gpu   MACE-OMol as a control
    mace_stop_shift    the stopping shift for MACE-OMol
    repeat_cpu, repeat_gpu   identical setup run twice, rxn0346
    threads_dbl        UMA-S double precision, 3 workers against 1 worker

Output results/prec_test.csv, one row per comparison, model and reaction:
    rmsd_ts_A, de_ts_mev, dbarrier_mev, rmsd_r_A, rmsd_p_A, steps_a, steps_b,
    f_a, f_b, met_a, met_b, agree

agree = RMSD <= 0.003 A and |dE_TS| <= 0.5 meV, fixed in D21 before the runs.
The summary counts the agreeing pairs at the broken-symmetry reactions.

    python pipeline/prec_test_table.py
"""
import collections
import csv
import json
import os

import numpy as np
from ase.io import read

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
T, P = f'{H}/neb_test', f'{H}/neb_prec'
BS = ['rxn0346', 'rxn4113', 'rxn1147', 'rxn4518']
CS = ['rxn0101', 'rxn0000']
RXNS = BS + CS
FAIR = ['uma-s', 'uma-m', 'esen']
TOL_RMSD, TOL_E = 0.003, 0.5

# name, model list, reactions, root a, root b
COMPARE = [
    ('sp_cpu_gpu', FAIR, RXNS, f'{T}/cpu3x8', f'{T}/gpu'),
    ('dbl_cpu_gpu', FAIR, RXNS, f'{P}/dbl_cpu', f'{P}/dbl_gpu'),
    ('tight_cpu_gpu', FAIR, RXNS, f'{P}/tight_fromcpu', f'{P}/tight_fromgpu'),
    ('stop_shift_cpu', FAIR, RXNS, f'{T}/cpu3x8', f'{P}/tight_fromcpu'),
    ('mace_default_cpu_gpu', ['mace-omol'], RXNS, f'{T}/cpu3x8', f'{T}/gpu'),
    ('mace_float32_cpu_gpu', ['mace-omol'], RXNS, f'{P}/mace32_cpu', f'{P}/mace32_gpu'),
    ('mace_stop_shift', ['mace-omol'], RXNS, f'{T}/cpu3x8', f'{P}/tight_fromcpu'),
    ('repeat_cpu', FAIR, ['rxn0346'], f'{T}/cpu3x8', f'{P}/rep_cpu'),
    ('repeat_gpu', FAIR, ['rxn0346'], f'{T}/gpu', f'{P}/rep_gpu'),
    ('threads_dbl', ['uma-s'], ['rxn0346', 'rxn0101', 'rxn0000'], f'{P}/dbl_cpu', f'{P}/dbl_1x24'),
]


def rmsd(a, b):
    A = a.get_positions() - a.get_positions().mean(0)
    B = b.get_positions() - b.get_positions().mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((A @ R - B) ** 2).sum(1).mean()))


def load(root, m, rx):
    d = f'{root}/{m}/{rx}'
    p = f'{d}/result.json'
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    if r['status'] != 'done':
        return None
    return r, {n: read(f'{d}/{n}.xyz') for n in ('reactant', 'transition_state', 'product')}


rows, missing = [], collections.Counter()
for name, models, rxns, ra, rb in COMPARE:
    for m in models:
        for rx in rxns:
            a, b = load(ra, m, rx), load(rb, m, rx)
            if a is None or b is None:
                missing[name] += 1
                continue
            (x, gx), (y, gy) = a, b
            r_ts = rmsd(gx['transition_state'], gy['transition_state'])
            de = (x['e_ts_ev'] - y['e_ts_ev']) * 1e3
            rows.append(dict(
                comparison=name, model=m, rxn=rx, group='bs' if rx in BS else 'closed',
                rmsd_ts_A=r_ts, de_ts_mev=de,
                dbarrier_mev=(x['barrier_ev'] - y['barrier_ev']) * 1e3,
                rmsd_r_A=rmsd(gx['reactant'], gy['reactant']),
                rmsd_p_A=rmsd(gx['product'], gy['product']),
                steps_a=x['n_steps'], steps_b=y['n_steps'], f_a=x['f_band_final'], f_b=y['f_band_final'],
                met_a=int(bool(x['criterion_met'])), met_b=int(bool(y['criterion_met'])),
                agree=int(r_ts <= TOL_RMSD and abs(de) <= TOL_E)))

print('%-22s %-7s %5s %6s   %s' % ('comparison', 'group', 'agree', 'pairs', 'RMSD TS max [A], |dE_TS| max [meV], unconverged'))
for name, *_ in COMPARE:
    for g in ('bs', 'closed'):
        S = [r for r in rows if r['comparison'] == name and r['group'] == g]
        if S:
            print('%-22s %-7s %5d %6d   %.4f   %.2f   %d'
                  % (name, g, sum(r['agree'] for r in S), len(S), max(r['rmsd_ts_A'] for r in S),
                     max(abs(r['de_ts_mev']) for r in S),
                     sum(1 for r in S if not (r['met_a'] and r['met_b']))))
if missing:
    print('missing pairs:', dict(missing))

out = os.path.join(REPO, 'results', 'prec_test.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
print(f'results/prec_test.csv   {len(rows)} rows')
