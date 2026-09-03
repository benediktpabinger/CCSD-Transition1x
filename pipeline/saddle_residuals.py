"""Restgradient an den bestaetigten Sattelpunkten dieser Arbeit.

Damit wird die Stufe-1-Schwelle kalibriert: sie muss klar oberhalb dessen
liegen, was eine konvergierte TS-Optimierung hinterlaesst, sonst verwirft sie
gute Strukturen als "nicht stationaer".

Die Zahl 0.006-0.011, die im Docstring von model_saddle_stats.py stand und von
dort ins Kapitel wanderte, stammt aus der Phase vor den Produktionslaeufen.
Hier wird sie am Zielniveau neu gemessen.

results/saddle_residuals.csv   eine Zeile je Lauf
"""
import csv
import glob
import os
import re

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
EVA = 51.42208
# Laeufe, die keinen gueltigen Sattelpunkt geliefert haben -- sie duerfen die
# Kalibrierung nicht beeinflussen.  Begruendung je Fall in Kapitel §8.
# Ausgeschlossen wird nur, was nicht stationaer ist -- ob es der RICHTIGE
# Sattelpunkt ist (Stufe 2 und 3), spielt fuer diese Kalibrierung keine Rolle.
# Ein Punkt, der Stufe 3 verfehlt, ist trotzdem ein Stationaerpunkt und sagt
# damit genauso viel darueber, was eine konvergierte TS-Opt hinterlaesst.
EXCLUDE = {
    ('B', 'rxn1283'): 'Optimierung nicht konvergiert (Iterationslimit)',
    ('B', 'rxn6196_maxiter50'): 'abgebrochene Variante mit MaxIter 50',
    ('C', 'rxn7060'): 'faellt durch Stufe 1, also nicht stationaer (1.71 eV/A)',
}


def last_maxg(path):
    """Letzter MAX-Gradient aus ORCAs Konvergenztabelle, Eh/Bohr -> eV/A."""
    if not os.path.exists(path):
        return None
    m = re.findall(r'MAX gradient\s+([\d.]+)\s+([\d.]+)',
                   open(path, errors='replace').read())
    return float(m[-1][0]) * EVA if m else None


def engrad_max(path):
    if not os.path.exists(path):
        return None
    t = open(path, errors='replace').read()
    i = t.find('CARTESIAN GRADIENT')
    if i < 0:
        return None
    G = []
    for line in t[i:].split('\n')[3:]:
        f = line.split()
        if len(f) < 6:
            break
        try:
            G.append([float(v) for v in f[3:6]])
        except ValueError:
            break
    return float(np.abs(np.array(G) * EVA).max()) if G else None


rows = []
for d in sorted(glob.glob(f'{H}/bs_uks_nebci_prod/rxn*')):
    for f in ('tsopt2.out', 'tsopt.out'):
        g = last_maxg(f'{d}/{f}')
        if g is not None:
            rows.append(('A', os.path.basename(d), f, g))
            break
for d in sorted(glob.glob(f'{H}/sep_step23/rxn*')):
    for f in ('tsopt2.out', 'tsopt.out'):
        g = last_maxg(f'{d}/{f}')
        if g is not None:
            rows.append(('B', os.path.basename(d), f, g))
            break
for d in sorted(glob.glob(f'{H}/orca_freq/tsopt_rxn*_UMA-M')):
    g = engrad_max(f'{d}/engrad.out')
    if g is not None:
        rows.append(('C', os.path.basename(d).split('_')[1], 'engrad.out', g))

os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/saddle_residuals.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['route', 'rxn', 'quelle', 'maxgrad_evang', 'gueltig', 'grund'])
    for r, rx, f, g in rows:
        key = (r, rx)
        w.writerow([r, rx, f, '%.4f' % g, 0 if key in EXCLUDE else 1,
                    EXCLUDE.get(key, '')])

ok = np.array([g for r, rx, f, g in rows if (r, rx) not in EXCLUDE])
print('Restgradient konvergierter TS-Optimierungen  [eV/A]')
print('  n = %d   Median %.4f   Spanne %.4f bis %.4f'
      % (len(ok), np.median(ok), ok.min(), ok.max()))
print('  Stufe-1-Schwelle 0.15 liegt %.0f-fach ueber dem Median '
      'und %.1f-fach ueber dem ungünstigsten Fall'
      % (0.15 / np.median(ok), 0.15 / ok.max()))
print()
for r in ('A', 'B', 'C'):
    v = np.array([g for rr, rx, f, g in rows
                  if rr == r and (rr, rx) not in EXCLUDE])
    print('  Route %s  n=%2d  Median %.4f  %.4f bis %.4f'
          % (r, len(v), np.median(v), v.min(), v.max()))
print()
print('  ausgeschlossen:')
for (r, rx), why in sorted(EXCLUDE.items()):
    hit = [g for rr, x, f, g in rows if rr == r and x == rx]
    print('    %s %-18s %s  (%s)'
          % (r, rx, ('%.4f' % hit[0]) if hit else '-', why))
print()
print('geschrieben: results/saddle_residuals.csv (%d Zeilen)' % len(rows))
