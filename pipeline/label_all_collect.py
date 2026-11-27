"""Label check at every Transition1x transition state (D46): our ORCA 6.0.1 DFT against the OMol25 label.

Our side, one folder per reaction with uks_sp.out (UKS + STABPerform) and uks_engrad.out (EnGrad on
the uks_sp orbitals; energy, forces and <S^2> are taken from here):
  ~/orca6_label_all/<rxn>                  7,519 TS, pipeline/job_label_all.sh
  ~/orca6_{s2pos,cshigh,twins,base}/<rxn>  2,490 TS, the earlier label check
OMol25 side: ~/omol25_ts_all.jsonl, the entry whose source is the one in results/omol25_ts_all.csv
(225 reactions have two entries at the same geometry, which agree).

  d_e_mev       E(ours) - E(OMol25); negative = our solution lower
  f_mae, f_max  mean / largest absolute difference of force components (eV/A)
  class_*       broken-symmetry if <S^2> > 0.05
  status        ok, or the reason our run is missing. Our geometry is the Transition1x TS, which
                differs from the OMol25 positions by up to 1.2e-3 A in 7 cases (energy within 0.08 meV)

Output results/label_all.csv

    python pipeline/label_all_collect.py
"""
import csv
import json
import os
import re

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
HA_EV = 27.211386245988
EVA = 51.42208
DIRS = [f'{H}/orca6_label_all'] + [f'{H}/orca6_{g}' for g in ('s2pos', 'cshigh', 'twins', 'base')]
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')
L_RE = re.compile(r'E\(\s*0\)\s*=\s*([-\d.]+)\s*Eh')


def read(p):
    return open(p, errors='replace').read() if os.path.exists(p) else None


def forces(t):
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


def positions(p):
    return np.array([[float(v) for v in l.split()[1:4]] for l in open(p).read().split('\n')[2:] if l.strip()])


src = {r['rxn']: r['source'] for r in csv.DictReader(open(f'{RES}/omol25_ts_all.csv')) if r['found'] == '1'}
split = {r['rxn']: r['split'] for r in csv.DictReader(open(f'{RES}/omol25_ts_all.csv'))}
label = {}
for line in open(f'{H}/omol25_ts_all.jsonl'):
    d = json.loads(line)
    if src.get(d['rxn']) == d['source']:
        label[d['rxn']] = d

rows = []
for rxn in sorted(label):
    L = label[rxn]
    r = dict(rxn=rxn, split=split[rxn], n_atoms=L['num_atoms'], s2_omol=L['s_squared'],
             class_omol='bs' if L['s_squared'] > 0.05 else 'cs')
    d = next((d for d in DIRS if os.path.exists(f'{d}/{rxn}/uks_sp.out')), None)
    sp = read(f'{d}/{rxn}/uks_sp.out') if d else None
    eg = read(f'{d}/{rxn}/uks_engrad.out') if d else None
    if sp is None:
        r['status'] = 'not run'
    elif 'ORCA TERMINATED NORMALLY' not in sp:
        r['status'] = 'stab not converged' if 'Stability equations did not converge' in sp else 'scf failed'
    elif eg is None or 'CARTESIAN GRADIENT' not in eg:
        r['status'] = 'no gradient'
    else:
        F, Fl = forces(eg), np.array(L['forces']).reshape(-1, 3)
        dx = np.abs(positions(f'{d}/{rxn}/ts.xyz') - np.array(L['positions']).reshape(-1, 3)).max()
        s2 = float(S2_RE.findall(eg)[-1])
        lam = L_RE.findall(sp)
        r.update(status='ok' if dx < 2e-3 else 'geometry differs', folder=os.path.basename(d),
                 s2_ours=s2, class_ours='bs' if s2 > 0.05 else 'cs',
                 lambda_min=float(lam[0]) if lam else '',
                 d_e_mev=round((float(E_RE.findall(eg)[-1]) * HA_EV - L['energy']) * 1000, 3),
                 f_mae=round(float(np.abs(F - Fl).mean()), 5), f_max=round(float(np.abs(F - Fl).max()), 5))
    rows.append(r)

keys = ['rxn', 'split', 'n_atoms', 'status', 'folder', 's2_omol', 's2_ours', 'class_omol', 'class_ours',
        'lambda_min', 'd_e_mev', 'f_mae', 'f_max']
with open(f'{RES}/label_all.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

ok = [r for r in rows if r['status'] == 'ok']
de = np.array([r['d_e_mev'] for r in ok])
print(f'{len(rows)} labels, {len(ok)} ok;', {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}})
print(f'|dE| <= 1 meV: {np.sum(np.abs(de) <= 1)}, ours lower > 1 meV: {np.sum(de < -1)}, ours higher > 1 meV: {np.sum(de > 1)}')
for a in ('cs', 'bs'):
    for b in ('cs', 'bs'):
        print(f'OMol25 {a} -> ours {b}: {sum(r["class_omol"] == a and r["class_ours"] == b for r in ok)}')
agree = [r for r in ok if abs(r['d_e_mev']) <= 1]
print('agreeing: median f_mae %.5f, max f_max %.4f eV/A' % (np.median([r['f_mae'] for r in agree]),
                                                          max(r['f_max'] for r in agree)))
print('results/label_all.csv')
