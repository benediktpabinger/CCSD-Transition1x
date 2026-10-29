"""Zufallsstichprobe der Reaktionen, deren OMol25-Label am gespeicherten
Transition1x-TS (praktisch) KEINE Symmetriebrechung zeigt, s2 <= 1e-3.

Frage (docs/paper_decisions.md, offener Punkt nach D13): Wie oft hat die
20-Grad-Rotation von OMol25 eine tiefe broken-symmetry-Loesung verfehlt, wo
das Label <S^2> = 0 traegt? In der Grenzzone (1e-3 < s2 <= 0.1) waren es 7 von
213. Die Stichprobe laeuft durch dasselbe Protokoll wie die Grenzzone
(pipeline/job_border_zone.sh mit TASKFILE/OUTROOT, Sammler
pipeline/border_zone_table.py mit BZ_SET=zero).

Auswahl: N Reaktionen gleichverteilt zufaellig aus allen gefundenen mit
s2 <= 1e-3, fester Seed, damit die Auswahl reproduzierbar ist.

Ausgaben
    ~/t1x_ts_zero/<rxn>.xyz
    ~/t1x_ts_zero_tasks.txt
    results/zero_sample_selection.csv

    python pipeline/mk_zero_sample_tasks.py [--n 200] [--seed 0]
"""
import argparse
import csv
import json
import os
import random

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
S2_MAX = 1e-3
XYZ_DIR = f'{H}/t1x_ts_zero'
TASKFILE = f'{H}/t1x_ts_zero_tasks.txt'
SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

pool = [r for r in csv.DictReader(open(os.path.join(REPO, 'results', 'omol25_ts_all.csv')))
        if r['found'] == '1' and float(r['s2']) <= S2_MAX]
pool.sort(key=lambda r: r['rxn'])
rng = random.Random(a.seed)
rows = sorted(rng.sample(pool, a.n), key=lambda r: r['rxn'])
ts = json.load(open(f'{H}/t1x_ts_all.json'))

os.makedirs(XYZ_DIR, exist_ok=True)
sel = []
with open(TASKFILE, 'w') as tf:
    for r in rows:
        t = ts[r['rxn']]
        Z, R = t['numbers'], t['positions']
        assert len(Z) == int(r['n_atoms']) == len(R)
        nel = sum(Z)
        assert nel % 2 == 0, r['rxn']
        xyz = f"{XYZ_DIR}/{r['rxn']}.xyz"
        with open(xyz, 'w') as fh:
            fh.write(f'{len(Z)}\n')
            fh.write(f"rxn={r['rxn']} split={r['split']} stored Transition1x TS "
                     f"(wB97x/6-31G(d)) s2_omol={float(r['s2']):.6f}\n")
            for z, (x, y, w) in zip(Z, R):
                fh.write(f'{SYM[z]:2s} {x:16.8f} {y:16.8f} {w:16.8f}\n')
        tf.write(f"{r['rxn']}:{xyz}:{nel}\n")
        sel.append(dict(rxn=r['rxn'], split=r['split'], formula=r['formula'],
                        n_atoms=r['n_atoms'], s2_omol=r['s2'],
                        e_omol_ev=r['e_omol_ev'], fmax_omol=r['fmax_omol']))

out = os.path.join(REPO, 'results', 'zero_sample_selection.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(sel[0].keys()))
    w.writeheader()
    w.writerows(sel)
print(f'{len(sel)} von {len(pool)} Reaktionen mit s2 <= {S2_MAX:g}, Seed {a.seed}')
print('   exakt 0: %d, 0 < s2 <= 1e-3: %d' % (sum(float(r["s2_omol"]) == 0 for r in sel),
                                              sum(float(r["s2_omol"]) > 0 for r in sel)))
for s in ('train', 'val', 'test'):
    print(f'   {s:<6s} {sum(r["split"] == s for r in sel)}')
print(f'{TASKFILE}\n{XYZ_DIR}/\n{os.path.relpath(out, REPO)}')
