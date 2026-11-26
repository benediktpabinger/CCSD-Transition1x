"""Task list for the label check at every Transition1x transition state (D46).

All transition states with an OMol25 label (results/omol25_ts_all.csv, found = 1) that are not
yet in the four ORCA 6.0.1 label-check files (results/stab_{s2pos,cshigh,twins,base}_orca6.csv).
Geometry from ~/t1x_ts_all.json, task format of job_border_zone.sh (rxn:xyz:n_electrons).

Output
    ~/t1x_ts_label_all/<rxn>.xyz
    ~/t1x_ts_label_all_tasks.txt
    results/label_all_selection.csv

    python pipeline/mk_label_all_tasks.py
"""
import csv
import json
import os

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
XYZ_DIR = f'{H}/t1x_ts_label_all'
TASKFILE = f'{H}/t1x_ts_label_all_tasks.txt'
SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}

done = set()
for g in ('s2pos', 'cshigh', 'twins', 'base'):
    done |= {r['rxn'] for r in csv.DictReader(open(f'{RES}/stab_{g}_orca6.csv'))}
rows = sorted((r for r in csv.DictReader(open(f'{RES}/omol25_ts_all.csv'))
               if r['found'] == '1' and r['rxn'] not in done), key=lambda r: r['rxn'])
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
        sel.append(dict(rxn=r['rxn'], split=r['split'], formula=r['formula'], n_atoms=r['n_atoms'],
                        s2_omol=r['s2'], e_omol_ev=r['e_omol_ev'], fmax_omol=r['fmax_omol']))

with open(f'{RES}/label_all_selection.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(sel[0].keys()))
    w.writeheader()
    w.writerows(sel)
print(f'{len(done)} already done, {len(sel)} new tasks -> {TASKFILE}')
print(f"OMol25 BS among new: {sum(float(r['s2_omol']) > 0.05 for r in sel)}")
