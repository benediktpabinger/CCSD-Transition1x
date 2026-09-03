"""Was jeder Schritt tatsaechlich gekostet hat, aus den Logs statt geschaetzt.

results/cost_hours.txt   eine Zeile je Vorgang, Stunden Wandzeit

Vorgaenge:
  praediktor    RKS + Gradient + externe Stabilitaetsanalyse am RKS-TS
                (stab_pipeline, Eintrag RKS-ref, Feld elapsed_s)
  routeA        neues NEB-CI-Band + SP + TS-Opt am Zielniveau
  routeB        vorhandenes Climbing Image + SP + TS-Opt
  routeC_orca   dreistufige ORCA-Bewertung der UMA-M-Geometrie
  routeC_pyscf  die vorgelagerte BS-TS-Optimierung in PySCF

Wichtig zum Praediktor: fuer eine STABILE Reaktion endet der Eintrag nach der
Stabilitaetsanalyse -- das ist genau der Aufwand, den die Vorhersage kostet.
Fuer eine instabile haengt die Datei die BS-Suche an, die die Vorhersage nicht
braucht; deren elapsed_s ist daher eine obere Schranke, kein Praediktorpreis.
Beide werden getrennt ausgewiesen.
"""
import csv
import glob
import json
import os
import re

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
PAT = re.compile(r'TOTAL RUN TIME:\s*(\d+) days (\d+) hours (\d+) minutes '
                 r'(\d+) seconds')


def orca_hours(path):
    if not os.path.exists(path):
        return None
    m = PAT.findall(open(path, errors='replace').read())
    if not m:
        return None                      # kein TOTAL RUN TIME = Abbruch
    d, h, mi, s = map(int, m[-1])
    return d * 24 + h + mi / 60 + s / 3600


rowsout = []

# ------------------------------------------------------------- Praediktor
for p in sorted(glob.glob(f'{H}/stab_pipeline/rxn*/result.json')):
    rx = os.path.basename(os.path.dirname(p))
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    if not g or g.get('elapsed_s') is None or g.get('ext_stable') is None:
        continue
    rowsout.append({'schritt': 'praediktor', 'rxn': rx,
                    'variante': 'stabil' if g['ext_stable'] else 'instabil',
                    'stunden': g['elapsed_s'] / 3600})

# ------------------------------------------------------------- Startpunkte
for root, files, tag in (
        ('bs_uks_nebci_prod', ('neb.out', 'bs.out', 'tsopt.out', 'tsopt2.out'), 'routeA'),
        ('sep_step23', ('bs.out', 'tsopt.out', 'tsopt2.out'), 'routeB')):
    for d in sorted(glob.glob(f'{H}/{root}/rxn*')):
        rx = os.path.basename(d)
        if '_' in rx:                    # Varianten wie rxn6196_maxiter50
            continue
        got = {f: orca_hours(f'{d}/{f}') for f in files}
        # Ein abgebrochener tsopt.out, der spaeter als tsopt2 nachgezogen wurde,
        # darf die Reaktion nicht aus der Statistik werfen -- gezaehlt wird,
        # was gelaufen ist.  Verlangt wird nur: die Bandphase bzw. der
        # Einzelpunkt steht, und mindestens eine TS-Optimierung ist fertig.
        first = files[0]
        tsopt = [got[f] for f in files if f.startswith('tsopt')]
        if got[first] is None or not any(h is not None for h in tsopt):
            continue
        v = [h for h in got.values() if h is not None]
        rowsout.append({'schritt': tag, 'rxn': rx, 'variante': '',
                        'stunden': float(sum(v))})

for d in sorted(glob.glob(f'{H}/orca_freq/tsopt_rxn*_UMA-M')):
    rx = os.path.basename(d).split('_')[1]
    v = [orca_hours(f'{d}/{f}') for f in ('bs_sp.out', 'engrad.out', 'numfreq.out')]
    v = [h for h in v if h is not None]
    if v:
        rowsout.append({'schritt': 'routeC_orca', 'rxn': rx, 'variante': '',
                        'stunden': float(sum(v))})

for p in sorted(glob.glob(f'{H}/bs_tsopt_umam/rxn*/result.json')):
    d = json.load(open(p))
    if d.get('elapsed_s'):
        rowsout.append({'schritt': 'routeC_pyscf',
                        'rxn': os.path.basename(os.path.dirname(p)),
                        'variante': d.get('status', ''),
                        'stunden': d['elapsed_s'] / 3600})

os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/cost_hours.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['schritt', 'rxn', 'variante', 'stunden'])
    for r in rowsout:
        w.writerow([r['schritt'], r['rxn'], r['variante'], '%.4f' % r['stunden']])

print('%-14s %4s %9s %9s %9s' % ('Schritt', 'n', 'Median', 'Min', 'Max'))
print('-' * 48)
for tag in ('praediktor', 'routeA', 'routeB', 'routeC_orca', 'routeC_pyscf'):
    v = np.array([r['stunden'] for r in rowsout if r['schritt'] == tag])
    if len(v):
        print('%-14s %4d %9.2f %9.2f %9.2f'
              % (tag, len(v), np.median(v), v.min(), v.max()))
for var in ('stabil', 'instabil'):
    v = np.array([r['stunden'] for r in rowsout
                  if r['schritt'] == 'praediktor' and r['variante'] == var])
    print('  praediktor/%-9s %4d %6.2f %9.2f %9.2f'
          % (var, len(v), np.median(v), v.min(), v.max()))
print()
print('geschrieben: results/cost_hours.csv (%d Zeilen)' % len(rowsout))
