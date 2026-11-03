"""Structures and starting Hessians for the DFT refinement pilot of D32.

25 MACE-OMol freezing-string structures, seed 0 (the model of the 96.6 % of Marks et al.): 10 failing and 10 passing high-MR structures
(groups closed, bs and high with N_FOD >= 0.5; pass = class route under the D23 check)
and 5 passing baseline structures. For each, the model Hessian in double precision
(central differences, 0.005 A, not projected) is written as an ORCA .hess file, so the
DFT saddle optimisation starts from the model curvature, as Marks et al. start from an
analytic Hessian. Output ~/marks_pilot/<rxn>/, ~/marks_pilot/tasks.txt

    python pipeline/marks_pilot_prep.py
"""
import csv
import glob
import json
import os
import random
import sys

import numpy as np
from ase import Atoms
from ase.data import atomic_masses
from ase.io import read, write
from ase.units import Bohr, Hartree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omol_neb import make_calc_factory  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
S = {r['rxn']: r for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{REPO}/results/{n}.csv'))}
MODEL = 'mace-omol'
J = [json.load(open(f)) for f in glob.glob(f'{H}/neb_fsm/check_{MODEL}/{MODEL}/*_fsm.json')]
hard = lambda j: S[j['rxn']]['group'] in ('closed', 'bs', 'high') and float(S[j['rxn']]['nfod']) >= 0.5   # noqa: E731
rng = random.Random(0)
pick = []
for kind, pool, n in (('fail', [j for j in J if hard(j) and j['cls'] != 'route'], 10),
                      ('pass', [j for j in J if hard(j) and j['cls'] == 'route'], 10),
                      ('baseline', [j for j in J if S[j['rxn']]['group'] == 'baseline' and j['cls'] == 'route'], 5)):
    pick += [(kind, j) for j in rng.sample(sorted(pool, key=lambda j: j['rxn']), n)]
calc = make_calc_factory(MODEL, 'cpu', 'float64')()


def grad(Z, x):
    a = Atoms(numbers=Z, positions=x, info={'charge': 0, 'spin': 1})
    a.calc = calc
    return -a.get_forces().ravel()


def write_hess(p, Z, x, Hm):
    n = 3 * len(Z)
    Hb = Hm * Bohr ** 2 / Hartree          # eV/A^2 -> Eh/bohr^2
    L = ['', '$orca_hessian_file', '', '$act_atom', '  0', '', '$act_coord', '  0', '', '$act_energy', '        0.000000', '',
         '$hessian', str(n)]
    for c0 in range(0, n, 5):
        cols = range(c0, min(c0 + 5, n))
        L.append('        ' + ''.join('%19d' % c for c in cols))
        for i in range(n):
            L.append('%6d     ' % i + ''.join('%19.10E' % Hb[i, c] for c in cols))
    L += ['', '$atoms', str(len(Z))]
    from ase.data import chemical_symbols
    for z, (a, b, c) in zip(Z, x / Bohr):
        L.append(' %-2s %13.5f %18.12f %18.12f %18.12f' % (chemical_symbols[z], atomic_masses[z], a, b, c))
    L += ['', '$end', '']
    open(p, 'w').write('\n'.join(L))


lines = []
for kind, j in pick:
    rx = j['rxn']
    d = f'{H}/neb_fsm/fsm/{MODEL}/{rx}'
    ts = read(f'{d}/transition_state.xyz')
    Z, x = ts.numbers, ts.positions.copy()
    n = x.size
    Hm = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = 0.005
        Hm[i] = (grad(Z, x + dx.reshape(-1, 3)) - grad(Z, x - dx.reshape(-1, 3))) / 0.01
    Hm = 0.5 * (Hm + Hm.T)
    out = f'{H}/marks_pilot/{rx}'
    os.makedirs(out, exist_ok=True)
    write(f'{out}/ts_model.xyz', Atoms(numbers=Z, positions=x), format='xyz')
    for name in ('reactant.xyz', 'product.xyz'):
        write(f'{out}/{name}', read(f'{d}/{name}'), format='xyz')
    write_hess(f'{out}/model.hess', Z, x, Hm)
    json.dump(dict(rxn=rx, kind=kind, group=S[rx]['group'], nfod=float(S[rx]['nfod']), cls_model=j['cls'],
                   barrier_model=json.load(open(f'{d}/result.json'))['barrier_ev']), open(f'{out}/model.json', 'w'))
    lines.append(f'{rx}:{out}/ts_model.xyz:{int(Z.sum())}')
    print(kind, rx, S[rx]['group'], j['cls'], flush=True)
open(f'{H}/marks_pilot/tasks.txt', 'w').write('\n'.join(lines) + '\n')
# the five structures that also start from the model Hessian: 3 failing, 2 passing high-MR (seed 0)
kinds = {ln.split(':')[0]: k for (k, j), ln in zip(pick, lines)}
sub = rng.sample([ln for ln in lines if kinds[ln.split(':')[0]] == 'fail'], 3) + \
    rng.sample([ln for ln in lines if kinds[ln.split(':')[0]] == 'pass'], 2)
open(f'{H}/marks_pilot/tasks_model.txt', 'w').write('\n'.join(sub) + '\n')
print(len(lines), 'structures')
