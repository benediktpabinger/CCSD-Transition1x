"""Are the OMol25 labels noisy where symmetry breaking starts?

For the low-MR twin pairs (results/twins_selection.csv) the OMol25 label at the
stored transition state is compared with our own ORCA 6.0.1 force on the lowest
solution at the same geometry (uks_engrad.out in ~/orca6_s2pos, ~/orca6_twins),
and the model force error is scored against both references. If the rise of the
model error near the onset were label noise, it would shrink against our DFT.

Limit: one frame per reaction, the transition-state frame. Noise in the other
frames of a path, which the models also learned from, is not tested here.

Output results/labels_vs_dft.csv, by model and depth bin.

    python pipeline/labels_vs_dft.py
"""
import csv
import json
import os
import re

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
EVA = 51.42208
HA_MEV = 27211.386245988
BINS = [('<1', -1e9, 1), ('1-10', 1, 10), ('10-43', 10, 43), ('43-100', 43, 100), ('>100', 100, 1e9)]
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')


def text(p):
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    return t if 'ORCA TERMINATED NORMALLY' in t else None


def forces(t):
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


tw = {r['rxn']: r['partner_bs'] for r in csv.DictReader(open(f'{RES}/twins_selection.csv'))}
want = set(tw) | set(tw.values())
label = {}
for line in open(f'{H}/omol25_ts_all.jsonl'):
    d = json.loads(line)
    if d['rxn'] in want and (d['rxn'] not in label or d['energy'] < label[d['rxn']]['energy']):
        label[d['rxn']] = d
ours, depth = {}, {}
for c, b in tw.items():
    for rx, d in ((b, f'{H}/orca6_s2pos/{b}'), (c, f'{H}/orca6_twins/{c}')):
        tu, tr = text(f'{d}/uks_engrad.out'), text(f'{d}/rks_sp.out')
        if tu and tr:
            ours[rx] = forces(tu)
            depth[rx] = (float(E_RE.findall(tr)[-1]) - float(E_RE.findall(tu)[-1])) * HA_MEV


def fl(rx):
    return np.array(label[rx]['forces']).reshape(-1, 3)


rows = []
for m in ('uma-s', 'uma-m', 'esen', 'mace-omol'):
    F = {}
    for line in open(f'{H}/model_sp_ts/all_{m}_spin1.jsonl'):
        d = json.loads(line)
        if d['rxn'] in want:
            F[d['rxn']] = np.array(d['forces_ev_a'])
    for name, lo, hi in BINS:
        P = [(c, b) for c, b in tw.items() if b in ours and c in ours and b in label and c in label
             and b in F and c in F and lo <= depth[b] < hi]
        mae = lambda a, x: float(np.abs(a - x).mean())   # noqa: E731
        rows.append(dict(
            model=m, depth_mev=name, n_pairs=len(P),
            label_vs_dft_bs=np.median([mae(fl(b), ours[b]) for c, b in P]),
            label_vs_dft_cs=np.median([mae(fl(c), ours[c]) for c, b in P]),
            ratio_vs_label=np.median([mae(F[b], fl(b)) for c, b in P]) / np.median([mae(F[c], fl(c)) for c, b in P]),
            ratio_vs_dft=np.median([mae(F[b], ours[b]) for c, b in P]) / np.median([mae(F[c], ours[c]) for c, b in P])))
        r = rows[-1]
        print('%-10s %-7s n=%3d  label-DFT bs %.5f cs %.5f  ratio vs label %.2f  vs DFT %.2f'
              % (m, name, len(P), r['label_vs_dft_bs'], r['label_vs_dft_cs'], r['ratio_vs_label'], r['ratio_vs_dft']))
with open(f'{RES}/labels_vs_dft.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
print('results/labels_vs_dft.csv')
