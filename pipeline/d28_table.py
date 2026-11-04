"""Generalisation test D28: DFT at the freezing-string structures on eSEN.

Model forces at the refined structures (saved to model_forces.npy, not kept by the search),
DFT force on the lowest UKS solution, distance to the nearest Transition1x frame of the
reaction, barrier against the reactant DFT of D24 where it exists.

Output results/d28_forces.csv

    python pipeline/d28_table.py
"""
import csv
import json
import os
import re
import sys

import h5py
import numpy as np
from ase import Atoms
from ase.io import read
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omol_neb import make_calc_factory  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
HA, EVA = 27.211386245988, 51.42208
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S = {r['rxn']: r for r in csv.DictReader(open(f'{RES}/d24_set.csv'))}
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'force_vs_path_distance.py')).read()
exec(src[src.index('def nearest'):src.index('paths = {}')])


def forces(p):
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    if 'CARTESIAN GRADIENT' not in t:
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
    return -np.array(G) * EVA


def energy(p):
    if not os.path.exists(p):
        return None
    e = E_RE.findall(open(p, errors='replace').read())
    return float(e[-1]) * HA if e else None


paths = {}
with h5py.File(f'{H}/data/Transition1x.h5', 'r') as f:
    for split in f:
        for formula in f[split]:
            for rx in f[split][formula]:
                if rx in S and rx not in paths:
                    paths[rx] = np.array(f[split][formula][rx]['positions'])
factory = make_calc_factory('esen', 'cpu')
calc = factory()
rows, missing = [], []
for rx, r in S.items():
    d = f'{H}/neb_fsm/fsm/esen/{rx}'
    if not os.path.exists(f'{d}/result.json') or not json.load(open(f'{d}/result.json'))['criterion_met']:
        continue
    F = forces(f'{H}/orca6_fsm/{rx}_ts/uks_engrad.out')
    if F is None:
        missing.append((rx, r['group']))
        continue
    res = json.load(open(f'{d}/result.json'))
    x = read(f'{d}/transition_state.xyz')
    if os.path.exists(f'{d}/model_forces.npy'):
        Fm = np.load(f'{d}/model_forces.npy')
    else:
        a = Atoms(numbers=x.numbers, positions=x.positions, info={'charge': 0, 'spin': 1})
        a.calc = calc
        Fm = a.get_forces()
        np.save(f'{d}/model_forces.npy', Fm)
    # DFT and model forces share the frame of transition_state.xyz, which ORCA does not reorient for gradients
    et, er = energy(f'{H}/orca6_fsm/{rx}_ts/uks_engrad.out'), energy(f'{H}/orca6_d24/{rx}_r/uks_engrad.out')
    rows.append(dict(rxn=rx, group=r['group'], bin=r['bin'], partner=r['partner'], nfod=float(r['nfod']),
                     dist=nearest(x.positions, paths[rx]), f_dft=float(np.sqrt((F ** 2).sum(1)).max()),
                     f_model=float(np.sqrt((Fm ** 2).sum(1)).max()), f_err_max=float(np.sqrt(((F - Fm) ** 2).sum(1)).max()),
                     f_err_mae=float(np.abs(F - Fm).mean()), barrier_dft=et - er if et and er else np.nan,
                     barrier_model=res['barrier_ev']))
print(len(rows), 'structures with DFT; missing (no DFT gradient):', len(missing), missing)
with open(f'{RES}/d28_forces_missing.csv', 'w') as fh:
    fh.write('rxn,group\n' + ''.join('%s,%s\n' % m for m in missing))
with open(f'{RES}/d28_forces.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
d = np.array([r['dist'] for r in rows])
e = np.array([r['f_err_mae'] for r in rows])
m = np.array([r['f_err_max'] for r in rows])
n = np.array([r['nfod'] for r in rows])
g = np.array([r['group'] for r in rows])
print('distance median %.3f, above 0.1 A %d, above 0.3 A %d' % (np.median(d), (d > .1).sum(), (d > .3).sum()))
print('Spearman mean error with distance %.2f (p=%.2g), with N_FOD %.2f (p=%.2g)' % (*spearmanr(d, e), *spearmanr(n, e)))
for lo, hi in ((0, .02), (.02, .1), (.1, .3), (.3, 9)):
    s = (d >= lo) & (d < hi)
    b = np.array([abs(r['barrier_model'] - r['barrier_dft']) for r in rows])[s]
    b = b[np.isfinite(b)] * 1e3
    print(f'dist {lo}-{hi}: n={s.sum():3d} mean err {np.median(e[s]):.4f} largest {np.median(m[s]):.3f}  barrier err n={len(b)} median {np.median(b) if len(b) else 0:.1f} meV, >43 meV {100*(b>43).mean() if len(b) else 0:.0f} %')
for lab, sel in (('far above 0.1 A', d > .1), ('near below 0.1 A', d <= .1)):
    hi, lo = sel & (n >= .5) & (g != 'bs'), sel & (n < .5) & (g != 'bs')
    if hi.sum() >= 4 and lo.sum() >= 4:
        print(f'{lab}, closed-shell: N_FOD>=0.5 {np.median(e[hi]):.4f} (n={hi.sum()}), <0.5 {np.median(e[lo]):.4f} (n={lo.sum()}), ratio {np.median(e[hi])/np.median(e[lo]):.2f}, p={mannwhitneyu(e[hi],e[lo]).pvalue:.2g}')
for k in ('baseline', 'closed', 'bs'):
    s = g == k
    if s.sum():
        print(f'{k:9s} n={s.sum():3d} mean err {np.median(e[s]):.4f} largest {np.median(m[s]):.3f} F_DFT {np.median([r["f_dft"] for r,t in zip(rows,s) if t]):.3f}')
