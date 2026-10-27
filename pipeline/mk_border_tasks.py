"""Taskliste fuer die Grenzzonen-Pruefung (docs/paper_decisions.md, offener
Punkt zur <S^2>-Schwelle).

Waehlt aus results/omol25_ts_all.csv die Reaktionen, deren OMol25-Label am
gespeicherten Transition1x-Uebergangszustand ein kleines, aber nicht
verschwindendes <S^2> traegt (S2_LO < s2 <= S2_HI), und schreibt fuer jede
die Geometrie aus ~/t1x_ts_all.json als xyz plus eine Taskdatei im Format
von job_hinge_t1x.sh (rxn:xyz-Pfad:Elektronenzahl).

An diesen Geometrien rechnet pipeline/job_border_zone.sh dann unser eigenes
Protokoll (RKS + EnGrad, UKS + STABPerform, EnGrad auf der UKS-Loesung),
dasselbe wie an den 45 Thesis-Reaktionen in ~/orca_hinge_t1x. Das sagt, ob
die kleinen OMol25-Werte echte flache broken-symmetry-Faelle sind oder
closed-shell mit Rauschen, und wo die Schwelle sinnvoll liegt.

Ausgaben
    ~/t1x_ts_border/<rxn>.xyz
    ~/t1x_ts_border_tasks.txt
    results/border_zone_selection.csv   die Auswahl, reproduzierbar in git

    python pipeline/mk_border_tasks.py
"""
import csv
import json
import os

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
S2_LO, S2_HI = 1e-3, 0.1
XYZ_DIR = f'{H}/t1x_ts_border'
TASKFILE = f'{H}/t1x_ts_border_tasks.txt'

SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}

rows = [r for r in csv.DictReader(open(os.path.join(REPO, 'results', 'omol25_ts_all.csv')))
        if r['found'] == '1' and S2_LO < float(r['s2']) <= S2_HI]
rows.sort(key=lambda r: r['rxn'])
ts = json.load(open(f'{H}/t1x_ts_all.json'))

os.makedirs(XYZ_DIR, exist_ok=True)
sel = []
with open(TASKFILE, 'w') as tf:
    for r in rows:
        t = ts[r['rxn']]
        Z, R = t['numbers'], t['positions']
        assert len(Z) == int(r['n_atoms']) == len(R)
        nel = sum(Z)                      # neutral
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

out = os.path.join(REPO, 'results', 'border_zone_selection.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(sel[0].keys()))
    w.writeheader()
    w.writerows(sel)

print(f'{len(sel)} Reaktionen mit {S2_LO:g} < s2 <= {S2_HI:g}')
for s in ('train', 'val', 'test'):
    print(f'   {s:<6s} {sum(r["split"] == s for r in sel)}')
print(f'Atome {min(int(r["n_atoms"]) for r in sel)}..{max(int(r["n_atoms"]) for r in sel)}')
print(f'{TASKFILE}\n{XYZ_DIR}/\n{os.path.relpath(out, REPO)}')
