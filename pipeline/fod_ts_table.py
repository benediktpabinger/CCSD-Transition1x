"""Eine Tabelle: N_FOD am gespeicherten Transition1x-Uebergangszustand JEDER
Reaktion, neben dem OMol25-Label derselben Geometrie.

Eingaben
    ~/fod_ts_all/<rxn>.json        aus pipeline/fod_all_ts.py (Job 11072061):
                                   PBE/def2-SVP, RKS, Fermi 5000 K, wie
                                   pipeline/screen_fod.py in der Thesis
    results/omol25_ts_all.csv      s2 und Gruppe des OMol25-Labels

Ausgabe results/fod_ts_all.csv, eine Zeile je Reaktion im H5 (10073):
    rxn, split, formula, n_atoms, n_elec
    nfod            N_FOD am gespeicherten TS
    e_fod_ha        PBE/def2-SVP-Energie dieser Rechnung (nur zur Kontrolle)
    seconds
    found, s2_omol, group_omol   aus omol25_ts_all.csv (leer, wenn nicht gefunden)

Pruefungen
    eine JSON je Reaktion, keine mit Fehler
    nfod == nfod_check (die beiden Summenformeln) auf 1e-5
    n_atoms wie im H5
    Zeilenzahl 10073

    python pipeline/fod_ts_table.py
"""
import csv
import json
import os
import sys

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


omol = {r['rxn']: r for r in csv.DictReader(open(os.path.join(REPO, 'results',
                                                               'omol25_ts_all.csv')))}
rows, bad, missing, mism = [], [], [], []
for rx in sorted(omol):
    p = f'{H}/fod_ts_all/{rx}.json'
    if not os.path.exists(p):
        missing.append(rx)
        continue
    d = json.load(open(p))
    o = omol[rx]
    if 'error' in d:
        bad.append(rx)
        continue
    if abs(d['nfod'] - d['nfod_check']) > 1e-5:
        mism.append(rx)
    rows.append(dict(rxn=rx, split=o['split'], formula=o['formula'],
                     n_atoms=int(o['n_atoms']), n_elec=d['n_elec'],
                     nfod=d['nfod'], e_fod_ha=d['energy_Ha'], seconds=d['seconds'],
                     found=o['found'], s2_omol=o['s2'], group_omol=o['group_omol'],
                     n_atoms_fod=d['n_atoms']))

print('%d Reaktionen, %d mit N_FOD' % (len(omol), len(rows)))
F = np.array([r['nfod'] for r in rows])
print('N_FOD: min %.3f  Median %.3f  Mittel %.3f  max %.3f' % (F.min(), np.median(F), F.mean(), F.max()))
for q in (0.1, 0.3, 0.5, 0.68, 1.0):
    print('   N_FOD > %.2f: %5d  (%.1f %%)' % (q, (F > q).sum(), 100 * (F > q).mean()))
print('nach OMol25-Gruppe (gefundene Reaktionen):')
for g in ('bs', 'closed'):
    S = np.array([r['nfod'] for r in rows if r['group_omol'] == g])
    if len(S):
        print('   %-7s n=%5d  Median %.3f  Mittel %.3f  10/90 %% %.3f/%.3f'
              % (g, len(S), np.median(S), S.mean(), *np.percentile(S, [10, 90])))
S2 = np.array([float(r['s2_omol']) for r in rows if r['found'] == '1'])
FF = np.array([r['nfod'] for r in rows if r['found'] == '1'])
from scipy.stats import spearmanr  # noqa: E402
rho = spearmanr(S2, FF).correlation
print('Spearman rho(N_FOD, s2_omol) ueber %d gefundene: %.3f' % (len(FF), rho))
# Wieviel MR-Charakter haben closed-shell-Reaktionen im N_FOD-Bereich der BS?
bsF = np.array([r['nfod'] for r in rows if r['group_omol'] == 'bs'])
csF = np.array([r['nfod'] for r in rows if r['group_omol'] == 'closed'])
lo, hi = np.percentile(bsF, [10, 90])
print('closed-shell im 10-90 %%-Fenster der BS (N_FOD %.2f..%.2f): %d von %d'
      % (lo, hi, ((csF >= lo) & (csF <= hi)).sum(), len(csF)))
print('Rechenzeit: Median %.0f s, Summe %.1f Kern-h (4 Threads)'
      % (np.median([r['seconds'] for r in rows]), 4 * sum(r['seconds'] for r in rows) / 3600))

print()
print('Pruefungen')
check(not missing, 'eine JSON je Reaktion' + ('' if not missing else ': fehlen %d' % len(missing)))
check(not bad, 'keine Rechnung mit Fehler' + ('' if not bad else ': %s' % bad[:10]))
check(not mism, 'nfod == nfod_check' + ('' if not mism else ': %s' % mism[:10]))
check(all(r['n_atoms_fod'] == r['n_atoms'] for r in rows), 'Atomzahl wie im H5')
check(len(rows) == 10073, '10073 Zeilen (%d)' % len(rows))
if fails:
    sys.exit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))

COLS = ['rxn', 'split', 'formula', 'n_atoms', 'n_elec', 'nfod', 'e_fod_ha', 'seconds',
        'found', 's2_omol', 'group_omol']
FMT = {'nfod': '%.6f', 'e_fod_ha': '%.8f'}
out = os.path.join(REPO, 'results', 'fod_ts_all.csv')
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in rows:
        w.writerow([('' if r.get(c) in (None, '') else (FMT[c] % r[c] if c in FMT else r[c]))
                    for c in COLS])
print('\n%s   %d Zeilen' % (os.path.relpath(out, REPO), len(rows)))
