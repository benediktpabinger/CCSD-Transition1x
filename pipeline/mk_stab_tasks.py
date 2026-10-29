"""Taskliste fuer die Stabilitaetsanalyse an gespeicherten Transition1x-
Uebergangszustaenden, nach OMol25-<S^2> ausgewaehlt.

Verallgemeinert pipeline/mk_border_tasks.py: statt der festen Grenzzone
(1e-3 < s2 <= 0.1) ein beliebiges Fenster --min-s2 < s2 <= --max-s2 ueber
alle in OMol25 gefundenen Reaktionen (results/omol25_ts_all.csv), gleiche
Ausgabeformate (xyz je Reaktion, Taskdatei rxn:xyz:nel, Auswahl-CSV in
results/). Standard: alle 1278 Reaktionen mit s2 > 1e-3, der Satz, fuer den
die Breaking Depth gebraucht wird (docs/paper_decisions.md, offener Punkt
"Gruppen nach Tiefe"), gerechnet mit ORCA 6.0.1 (D16) durch
pipeline/job_border_zone.sh mit TASKFILE/OUTROOT.

    python pipeline/mk_stab_tasks.py --name s2pos --min-s2 1e-3 --max-s2 2
      -> ~/t1x_ts_s2pos/<rxn>.xyz, ~/t1x_ts_s2pos_tasks.txt, results/s2pos_selection.csv
"""
import argparse
import csv
import json
import os

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='s2pos')
ap.add_argument('--min-s2', type=float, default=1e-3)
ap.add_argument('--max-s2', type=float, default=2.0)
a = ap.parse_args()
XYZ_DIR = f'{H}/t1x_ts_{a.name}'
TASKFILE = f'{H}/t1x_ts_{a.name}_tasks.txt'

rows = [r for r in csv.DictReader(open(os.path.join(REPO, 'results', 'omol25_ts_all.csv')))
        if r['found'] == '1' and a.min_s2 < float(r['s2']) <= a.max_s2]
rows.sort(key=lambda r: r['rxn'])
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

out = os.path.join(REPO, 'results', f'{a.name}_selection.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(sel[0].keys()))
    w.writeheader()
    w.writerows(sel)
print(f'{len(sel)} Reaktionen mit {a.min_s2:g} < s2 <= {a.max_s2:g}')
for lo, hi in ((a.min_s2, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, a.max_s2)):
    print('   %g..%g: %d' % (lo, hi, sum(lo < float(r['s2_omol']) <= hi for r in sel)))
print(f'Atome {min(int(r["n_atoms"]) for r in sel)}..{max(int(r["n_atoms"]) for r in sel)}')
print(f'{TASKFILE}\n{XYZ_DIR}/\n{os.path.relpath(out, REPO)}')
