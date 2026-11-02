"""DFT at the two rxn4113 transition states of each model (decision D21).

For each of the 4 models: the transition state where its band met 0.05 eV/A
(ts05), the one after continuing the band (tstight), and its reactant, each with
the stored-TS protocol in ORCA 6.0.1 (~/orca6_r4113, RKS + EnGrad, UKS +
STABPerform, EnGrad on the UKS orbitals). Barriers are against the model's own
reactant, which is closed-shell.

Output results/rxn4113_dft.csv.

    python pipeline/rxn4113_dft.py
"""
import csv
import json
import os
import re

import numpy as np
from ase.io import read

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
HA, EVA = 27.211386245988, 51.42208
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')


def run(p):
    t = open(p, errors='replace').read()
    assert 'ORCA TERMINATED NORMALLY' in t, p
    G = []
    i = t.find('CARTESIAN GRADIENT')
    if i >= 0:
        for line in t[i:].split('\n')[3:]:
            f = line.split()
            if len(f) < 6:
                break
            try:
                G.append([float(v) for v in f[3:6]])
            except ValueError:
                break
    s = S_RE.findall(t)
    return dict(e=float(E_RE.findall(t)[-1]) * HA, s2=float(s[-1]) if s else None,
                F=-np.array(G) * EVA if G else None)


rows = []
for m in ('uma-s', 'uma-m', 'esen', 'mace-omol'):
    d = f'{H}/orca6_r4113/rxn4113_{m}_'
    ru, rr = run(d + 'r/uks_engrad.out'), run(d + 'r/rks_sp.out')
    for tag, root in (('ts05', f'{H}/neb_test/cpu3x8'), ('tstight', f'{H}/neb_prec/tight_fromcpu')):
        u, k = run(f'{d}{tag}/uks_engrad.out'), run(f'{d}{tag}/rks_sp.out')
        res = json.load(open(f'{root}/{m}/rxn4113/result.json'))
        ts = read(f'{root}/{m}/rxn4113/final_band.extxyz', ':')[res['ts_image']]
        dF = ts.get_forces() - u['F']
        rows.append(dict(model=m, structure=tag, barrier_dft_uks_ev=u['e'] - ru['e'],
                         barrier_dft_rks_ev=k['e'] - rr['e'], barrier_model_ev=res['barrier_ev'],
                         s2=u['s2'], depth_mev=(k['e'] - u['e']) * 1e3,
                         f_dft_max=np.abs(u['F']).max(), f_err_max=np.abs(dF).max(),
                         f_err_mae=np.abs(dF).mean()))
        print(rows[-1])
with open(os.path.join(REPO, 'results', 'rxn4113_dft.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
