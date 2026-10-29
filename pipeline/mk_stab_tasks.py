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
ap.add_argument('--min-s2', type=float, default=1e-3, help='s2 > min-s2 (OMol25-Label)')
ap.add_argument('--max-s2', type=float, default=2.0, help='s2 <= max-s2')
ap.add_argument('--min-fod', type=float, default=-1.0, help='N_FOD > min-fod (results/fod_ts_all.csv)')
ap.add_argument('--max-fod', type=float, default=99.0, help='N_FOD <= max-fod')
ap.add_argument('--random', type=int, default=0, help='Zufallsstichprobe dieser Groesse aus der Auswahl')
ap.add_argument('--match-hist', help='Auswahl-CSV (z. B. results/s2pos_selection.csv): Stichprobe '
                'mit derselben N_FOD-Verteilung wie deren Reaktionen im Fenster, Bins von 0.1')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
XYZ_DIR = f'{H}/t1x_ts_{a.name}'
TASKFILE = f'{H}/t1x_ts_{a.name}_tasks.txt'

# Die 2x2-Zellen des Papers (docs/paper_decisions.md D18), MR-Schnitt N_FOD = 0.5:
#   s2pos   --min-s2 1e-3                      (broken-symmetry-Seite, beide MR-Zeilen)
#   cshigh  --max-s2 1e-3 --min-fod 0.5        (closed-shell, hohe MR, vollstaendig)
#   cslow   --max-s2 1e-3 --max-fod 0.5 --match-hist results/s2pos_selection.csv
#           (closed-shell, niedrige MR: Stichprobe mit derselben N_FOD-Verteilung
#            wie die broken-symmetry-Reaktionen unter 0.5, also ab deren
#            niedrigstem N_FOD 0.134, nicht aus dem ganzen Bereich ab 0)
fod = {r['rxn']: float(r['nfod'])
       for r in csv.DictReader(open(os.path.join(REPO, 'results', 'fod_ts_all.csv')))}
rows = [r for r in csv.DictReader(open(os.path.join(REPO, 'results', 'omol25_ts_all.csv')))
        if r['found'] == '1' and a.min_s2 < float(r['s2']) <= a.max_s2
        and a.min_fod < fod[r['rxn']] <= a.max_fod]
rows.sort(key=lambda r: r['rxn'])
import random  # noqa: E402
if a.random:
    rows = sorted(random.Random(a.seed).sample(rows, a.random), key=lambda r: r['rxn'])
if a.match_hist:
    # Zielverteilung: die Reaktionen der Referenz-Auswahl im selben N_FOD-Fenster
    ref = [fod[r['rxn']] for r in csv.DictReader(open(a.match_hist))
           if a.min_fod < fod[r['rxn']] <= a.max_fod]
    rng = random.Random(a.seed)
    picked = []
    b = 0.0
    while b < max(ref):
        want = sum(b < x <= b + 0.1 for x in ref)
        have = [r for r in rows if b < fod[r['rxn']] <= b + 0.1]
        assert len(have) >= want, f'Bin {b:.1f}: {want} gewuenscht, {len(have)} vorhanden'
        picked += rng.sample(have, want)
        b += 0.1
    rows = sorted(picked, key=lambda r: r['rxn'])
    print(f'Stichprobe mit der N_FOD-Verteilung von {a.match_hist} ({len(ref)} Referenzreaktionen)')
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
                        n_atoms=r['n_atoms'], s2_omol=r['s2'], nfod=fod[r['rxn']],
                        e_omol_ev=r['e_omol_ev'], fmax_omol=r['fmax_omol']))

out = os.path.join(REPO, 'results', f'{a.name}_selection.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(sel[0].keys()))
    w.writeheader()
    w.writerows(sel)
print(f'{len(sel)} Reaktionen mit {a.min_s2:g} < s2 <= {a.max_s2:g}, {a.min_fod:g} < N_FOD <= {a.max_fod:g}'
      + (f', Zufallsstichprobe Seed {a.seed}' if a.random else ''))
print('   N_FOD Median %.3f, < 0.5: %d, > 0.5: %d'
      % (sorted(r['nfod'] for r in sel)[len(sel) // 2], sum(r['nfod'] < 0.5 for r in sel),
         sum(r['nfod'] >= 0.5 for r in sel)))
for lo, hi in ((a.min_s2, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, a.max_s2)):
    print('   %g..%g: %d' % (lo, hi, sum(lo < float(r['s2_omol']) <= hi for r in sel)))
print(f'Atome {min(int(r["n_atoms"]) for r in sel)}..{max(int(r["n_atoms"]) for r in sel)}')
print(f'{TASKFILE}\n{XYZ_DIR}/\n{os.path.relpath(out, REPO)}')
