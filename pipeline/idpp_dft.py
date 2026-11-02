"""DFT at the transition states from the Transition1x start and the interpolated start.

ORCA 6.0.1 stored-TS protocol (~/orca6_idpp, rxn4113 reference data in ~/orca6_r4113).
Barriers are against each model's own reactant. Output results/idpp_dft.csv.

    python pipeline/idpp_dft.py
"""
import csv
import json
import os
import re

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
HA, EVA = 27.211386245988, 51.42208
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')


def run(p):
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return None
    G = []
    for line in t[t.find('CARTESIAN GRADIENT'):].split('\n')[3:]:
        f = line.split()
        if len(f) < 6:
            break
        try:
            G.append([float(v) for v in f[3:6]])
        except ValueError:
            break
    s = S_RE.findall(t)
    return dict(e=float(E_RE.findall(t)[-1]) * HA, s2=float(s[-1]) if s else 0.0,
                fmax=float(np.abs(np.array(G) * EVA).max()) if G else None)


rows = []
for rx in ('rxn0000', 'rxn1147', 'rxn4518', 'rxn4113'):
    for m in ('uma-s', 'uma-m', 'esen', 'mace-omol'):
        base = f'{H}/orca6_r4113/{rx}_{m}_' if rx == 'rxn4113' else f'{H}/orca6_idpp/{rx}_{m}_'
        r = run(base + 'r/uks_engrad.out')
        for start, d, root in (('t1x', base + ('ts05' if rx == 'rxn4113' else 'tst1x'), f'{H}/neb_test/cpu3x8'),
                               ('idpp', f'{H}/orca6_idpp/{rx}_{m}_tsidpp', f'{H}/neb_prec/idpp')):
            u, k = run(d + '/uks_engrad.out'), run(d + '/rks_sp.out')
            res = json.load(open(f'{root}/{m}/{rx}/result.json'))
            if u is None or k is None or r is None:
                print('missing', rx, m, start)
                continue
            rows.append(dict(rxn=rx, model=m, start=start, barrier_dft_ev=u['e'] - r['e'], barrier_model_ev=res['barrier_ev'],
                             s2=u['s2'], depth_mev=(k['e'] - u['e']) * 1e3, f_dft_max=u['fmax'],
                             criterion_met=int(bool(res['criterion_met'])), n_steps=res['n_steps']))
            x = rows[-1]
            print('%-8s %-10s %-5s DFT %.3f  model %.3f  S2 %.3f  depth %8.1f  f %.3f  met %d'
                  % (rx, m, start, x['barrier_dft_ev'], x['barrier_model_ev'], x['s2'], x['depth_mev'], x['f_dft_max'], x['criterion_met']))
with open(os.path.join(REPO, 'results', 'idpp_dft.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
