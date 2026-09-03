# -*- coding: utf-8 -*-
"""Wie weit streut die DFT-Barriere einer Reaktion ueber die drei Modell-TS?

Bricht ab, wenn eine Pruefung fehlschlaegt.

DIE FRAGE
    Jede der 45 Reaktionen hat drei Uebergangszustaende -- einen je MLIP. An
    jedem davon steht eine DFT-Barriere, alle drei auf demselben Niveau und
    aus derselben Rechnung. Sie unterscheiden sich also nur darin, WO der
    Punkt liegt, nicht wie er bewertet wurde. Die Spannweite max - min ueber
    diese drei Zahlen ist damit der reine Geometrieeffekt: was die
    Verschiebung des Sattels allein an der Barriere anrichtet.

    Gegenstueck zu results/model_ts_rmsd.csv, das dieselben drei Punkte rein
    geometrisch vergleicht. Speist Panel b der Barrieren-Spannweiten-Figur
    (fig9_5, pipeline/plot_omol25_figs.py).

QUELLEN
    spread_mev   results/omol25_model_geoms.csv, Spalte barr_dft [eV], je
                 Reaktion ueber die drei Modellzeilen. In meV umgerechnet.
    group_rxn    results/paper_reactions.csv, Spalte group_rxn. UEBERNOMMEN,
                 nicht neu abgeleitet -- es ist das Reaktionslabel (unstable,
                 wenn mindestens einer der drei Modell-TS unstable_ts = 1
                 hat), und genau so soll es hier stehen.

    Join-Schluessel ist rxn, in beiden Dateien. Die Pruefung verlangt, dass
    die Schluesselmengen deckungsgleich sind und jede Reaktion in der Master
    genau drei Modellzeilen hat -- ein Join ueber unvollstaendige Gruppen
    wuerde eine zu kleine Spannweite melden, ohne dass es auffiele.

DIE SCHWELLE 43 meV
    Chemische Genauigkeit, 1 kcal/mol. Sie wird hier nur ausgezaehlt, nicht
    als Kriterium verwendet.

results/barrier_spread.csv
"""
import csv
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
CHEM_ACC = 43.0            # meV, 1 kcal/mol
ERR_ACC = 0.043            # eV, dieselbe Schwelle fuer |err_barr|

# --- eingefrorene Kennzahlen, Stand 26.08.2026 ------------------------------
# Aendert sich eine dieser Zahlen, bricht der Lauf ab. Sie stehen so im
# README und in der Figurenbeschriftung; beides darf nicht stillschweigend
# von der Datei abweichen.
FROZEN = {
    'n': 45,
    'n_stable': 27,
    'n_unstable': 18,
    'med_stable': 0.335,          # meV
    'med_unstable': 10.705,       # meV
    'over_stable': 2,             # Reaktionen mit spread_mev > 43
    'over_unstable': 5,
    'max_rxn': 'rxn8837',
    'max_spread': 4434.359,       # meV
    'un_rows_within': 49,         # unstable-Zeilen mit |err_barr| < 0.043 eV
    'un_rows_total': 53,
}
TOL_MEV = 5e-4                    # Mediane sind auf drei Nachkommastellen frei

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


def load(name):
    with open(os.path.join(RES, name), encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


M = load('omol25_model_geoms.csv')
P = load('paper_reactions.csv')

GROUP = {r['rxn']: r['group_rxn'] for r in P}
by = defaultdict(list)
for r in M:
    by[r['rxn']].append(r)

incomplete = sorted(rx for rx, v in by.items() if len(v) != 3)
missing_b = sorted(r['rxn'] for r in M if r['barr_dft'] == '')
only_master = sorted(set(by) - set(GROUP))
only_paper = sorted(set(GROUP) - set(by))

rows = []
for rx in sorted(by):
    b = [float(x['barr_dft']) for x in by[rx] if x['barr_dft'] != '']
    rows.append({
        'rxn': rx,
        'group_rxn': GROUP.get(rx, 'NOT FOUND'),
        'spread_mev': None if len(b) != 3 else (max(b) - min(b)) * 1000.0,
    })

with open(os.path.join(RES, 'barrier_spread.csv'), 'w', newline='',
          encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'group_rxn', 'spread_mev'])
    for r in rows:
        w.writerow([r['rxn'], r['group_rxn'],
                    'NOT FOUND' if r['spread_mev'] is None
                    else '%.3f' % r['spread_mev']])

# ------------------------------------------------------------------ Bericht
val = [r for r in rows if r['spread_mev'] is not None]
grp = {g: [r for r in val if r['group_rxn'] == g]
       for g in ('stable', 'unstable')}

print('SPANNWEITE DER DFT-BARRIERE UEBER DIE DREI MODELLGEOMETRIEN')
print('=' * 88)
print('%d Reaktionen.  nach group_rxn (uebernommen): %d stable / %d unstable'
      % (len(val), len(grp['stable']), len(grp['unstable'])))
print()
print('%-10s %4s %12s %10s %12s %14s'
      % ('Gruppe', 'n', 'Median/meV', '> 43 meV', 'min/meV', 'max/meV'))
print('-' * 88)
for g in ('stable', 'unstable'):
    s = [r['spread_mev'] for r in grp[g]]
    print('%-10s %4d %12.3f %10d %12.3f %14.3f'
          % (g, len(s), st.median(s), sum(1 for v in s if v > CHEM_ACC),
             min(s), max(s)))
alls = [r['spread_mev'] for r in val]
print('%-10s %4d %12.3f %10d %12.3f %14.3f'
      % ('alle', len(alls), st.median(alls),
         sum(1 for v in alls if v > CHEM_ACC), min(alls), max(alls)))

mx = max(val, key=lambda r: r['spread_mev'])
print()
print('Maximum: %s   %.3f meV   (%s)'
      % (mx['rxn'], mx['spread_mev'], mx['group_rxn']))
print()
print('die %d Reaktionen ueber %.0f meV:' % (sum(1 for v in alls
                                                 if v > CHEM_ACC), CHEM_ACC))
for r in sorted((r for r in val if r['spread_mev'] > CHEM_ACC),
                key=lambda r: -r['spread_mev']):
    print('   %-9s %-9s %11.3f meV' % (r['rxn'], r['group_rxn'],
                                       r['spread_mev']))

un = [r for r in M if r['unstable_ts'] == '1' and r['err_barr'] != '']
within = [r for r in un if abs(float(r['err_barr'])) < ERR_ACC]
print()
print('AUS DER MASTER, zum Vergleich')
print('-' * 88)
print('   unstable-Zeilen mit |err_barr| < %.3f eV: %d von %d  (%.0f %%)'
      % (ERR_ACC, len(within), len(un), 100.0 * len(within) / len(un)))
print('   Die Barriere an einer instabilen Geometrie ist also meist genau;')
print('   die Streuung sitzt darin, WELCHE Geometrie es ist.')

# ---------------------------------------------------------------- Pruefungen
print()
print('Pruefungen')
check(not only_master and not only_paper,
      'Schluesselmengen deckungsgleich (Join ueber rxn)'
      + ('' if not (only_master or only_paper)
         else ' -- nur Master: %s   nur paper_reactions: %s'
         % (only_master, only_paper)))
check(not incomplete, 'jede Reaktion mit genau drei Modellzeilen'
      + ('' if not incomplete else ' -- Ausnahmen: %s' % incomplete))
check(not missing_b, 'barr_dft in jeder Modellzeile belegt'
      + ('' if not missing_b else ' -- fehlt bei %s' % missing_b))
check(all(r['group_rxn'] != 'NOT FOUND' for r in rows),
      'group_rxn fuer jede Zeile aus paper_reactions.csv')
check(all(r['spread_mev'] is not None and r['spread_mev'] >= 0 for r in rows),
      'spread_mev fuer jede Zeile, nicht negativ')

got = {
    'n': len(val),
    'n_stable': len(grp['stable']),
    'n_unstable': len(grp['unstable']),
    'med_stable': st.median([r['spread_mev'] for r in grp['stable']]),
    'med_unstable': st.median([r['spread_mev'] for r in grp['unstable']]),
    'over_stable': sum(1 for r in grp['stable']
                       if r['spread_mev'] > CHEM_ACC),
    'over_unstable': sum(1 for r in grp['unstable']
                         if r['spread_mev'] > CHEM_ACC),
    'max_rxn': mx['rxn'],
    'max_spread': mx['spread_mev'],
    'un_rows_within': len(within),
    'un_rows_total': len(un),
}
drift = []
for k, exp in FROZEN.items():
    v = got[k]
    if isinstance(exp, str):
        if v != exp:
            drift.append('%s: %s statt %s' % (k, v, exp))
    elif isinstance(exp, int):
        if v != exp:
            drift.append('%s: %d statt %d' % (k, v, exp))
    elif abs(v - exp) > TOL_MEV:
        drift.append('%s: %.3f statt %.3f' % (k, v, exp))
check(not drift, 'Kennzahlen unveraendert gegen den eingefrorenen Stand'
      + ('' if not drift else ' -- %s' % drift))

print()
print('geschrieben: results/barrier_spread.csv (%d Zeilen)' % len(rows))
if fails:
    raise SystemExit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))
