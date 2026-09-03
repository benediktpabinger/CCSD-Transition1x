"""Die 135 Modell-NEB-Laeufe: was der Lauf behauptet, und was gemessen ist.

Bricht ab, wenn eine Pruefung fehlschlaegt.

DIE ZWEI SPALTEN, DIE NICHT DASSELBE SIND
    converged_marker  Existenz der Datei <modeldir>/<rxn>/converged. Sie wird
                      angelegt, wenn relax_neb.run(fmax=cineb_fmax) True
                      zurueckgibt (pipeline/uma_neb.py, Zeilen 159 bis 163).
                      Dieser Rueckgabewert ist NICHT gleichbedeutend mit
                      erreichter Toleranz: NEBOptimizer.run_ode gibt True
                      zurueck, sofern ode12r keine OptimizerConvergenceError
                      wirft, und ode12r faellt am Ende seiner Schleife
                      'for nit in range(1, steps + 1)' ohne raise heraus, wenn
                      das Schrittbudget aufgebraucht ist. run() setzt
                      max_steps = steps, die CI-Phase bekommt also erneut die
                      vollen 500 Versuche; gezaehlt werden Versuche,
                      protokolliert nur angenommene Schritte.
    criterion_met     f_band_final <= 0.05, also das tatsaechlich erreichte
                      Bandkriterium. Die Logspalte hat vier Nachkommastellen,
                      deshalb wird mit TOL = 5e-5 verglichen: ein Lauf, der bei
                      0.049996 anhielt, steht im Log als 0.0500.

Die Diskrepanz zwischen beiden ist ein Befund, keine Panne, und wird hier
festgehalten statt geglaettet. Die 21 betroffenen Zeilen sind namentlich
eingefroren; die Pruefung schlaegt an, sobald sich die Menge aendert.

WEITERE SPALTEN
    f_band_final   fmax der letzten Logzeile [eV/A]. ASE-Konvention, also die
                   groesste Kraftnorm je Atom auf dem Band -- und zwar die
                   PROJIZIERTE Bandkraft, nicht die rohe Kraft am TS-Bild.
    n_steps        Zahl der Logzeilen ueber beide Phasen, nicht der letzte
                   Schrittindex.

QUELLEN
    <modeldir>/<rxn>/neb.log, <modeldir>/<rxn>/converged
    modeldir: uma_neb_results, uma_m_neb_results, esen_neb_results
    Klassenspalte und Kraefte fuer den Report: results/omol25_model_geoms.csv

results/neb_runs.csv
"""
import collections
import csv
import os
import re

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
MD = {'uma-s': 'uma_neb_results', 'uma-m': 'uma_m_neb_results',
      'esen': 'esen_neb_results'}
CINEB = 0.05
TOL = 5e-5                      # das Log rundet auf vier Nachkommastellen
LINE = re.compile(r'^NEBOptimizer\[\w+\]:\s+(\d+)\s+\S+\s+([-\d.eE+]+)\s*$')

# Ohne Marker, aus dem Erstlauf bekannt.
ERWARTET_OHNE_MARKER = {('rxn0894', 'uma-s'), ('rxn8837', 'esen')}

# Marker ja, Kriterium nein. Eingefroren am 24.08.2026.
ERWARTETE_DISKREPANZ = {
    ('rxn0894', 'esen'), ('rxn0894', 'uma-m'),
    ('rxn1061', 'esen'), ('rxn1061', 'uma-m'), ('rxn1061', 'uma-s'),
    ('rxn1154', 'esen'), ('rxn1154', 'uma-m'), ('rxn1154', 'uma-s'),
    ('rxn4004', 'esen'), ('rxn4004', 'uma-m'), ('rxn4004', 'uma-s'),
    ('rxn7937', 'esen'), ('rxn7937', 'uma-s'),
    ('rxn7949', 'esen'), ('rxn7949', 'uma-m'), ('rxn7949', 'uma-s'),
    ('rxn8837', 'uma-m'), ('rxn8837', 'uma-s'),
    ('rxn8885', 'esen'), ('rxn8885', 'uma-m'), ('rxn8885', 'uma-s'),
}

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


def read_log(p):
    if not os.path.exists(p):
        return None, None
    st = []
    for line in open(p, errors='replace'):
        m = LINE.match(line.strip())
        if m:
            st.append(float(m.group(2)))
    return (st[-1] if st else None), len(st)


geo = {(r['rxn'], r['model']): r for r in
       csv.DictReader(open(f'{OUT}/omol25_model_geoms.csv'))}
rxns = sorted({k[0] for k in geo})

rows, nolog = [], []
for rx in rxns:
    for m, d in MD.items():
        base = f'{H}/{d}/{rx}'
        f, ns = read_log(f'{base}/neb.log')
        if f is None:
            nolog.append('%s/%s' % (rx, m))
        rows.append({
            'rxn': rx, 'model': m,
            'converged_marker': int(os.path.exists(f'{base}/converged')),
            'criterion_met': (None if f is None else int(f <= CINEB + TOL)),
            'f_band_final': f, 'n_steps': ns})

os.makedirs(OUT, exist_ok=True)
rows.sort(key=lambda r: (r['rxn'], r['model']))
with open(f'{OUT}/neb_runs.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'model', 'converged_marker', 'criterion_met',
                'f_band_final', 'n_steps'])
    for r in rows:
        w.writerow([r['rxn'], r['model'], r['converged_marker'],
                    '' if r['criterion_met'] is None else r['criterion_met'],
                    '' if r['f_band_final'] is None else '%.4f' % r['f_band_final'],
                    '' if r['n_steps'] is None else r['n_steps']])

key = lambda r: (r['rxn'], r['model'])
marker = {key(r) for r in rows if r['converged_marker'] == 1}
met = {key(r) for r in rows if r['criterion_met'] == 1}
disk = marker - met
ohne = {key(r) for r in rows if r['converged_marker'] == 0}
D = {key(r): r for r in rows}

print('DIE 135 MODELL-NEB-LAEUFE')
print('=' * 90)
print('%d Zeilen.  Marker gesetzt: %d.  Kriterium erfuellt: %d.  '
      'Diskrepanz: %d.  Ohne Marker: %d.'
      % (len(rows), len(marker), len(met), len(disk), len(ohne)))
print()

print('DISKREPANZ  --  Marker ja, Kriterium nein')
print('-' * 90)
print('%-9s %-6s %12s %8s   %s' % ('rxn', 'model', 'f_band_final', 'n_steps',
                                   'Klasse am TS'))
for k in sorted(disk, key=lambda k: -D[k]['f_band_final']):
    print('%-9s %-6s %12.4f %8d   %s'
          % (k[0], k[1], D[k]['f_band_final'], D[k]['n_steps'],
             'instabil' if geo[k]['unstable_ts'] == '1' else 'stabil'))
print()
print('OHNE MARKER')
print('-' * 90)
for k in sorted(ohne, key=lambda k: -D[k]['f_band_final']):
    print('%-9s %-6s %12.4f %8d   %s'
          % (k[0], k[1], D[k]['f_band_final'], D[k]['n_steps'],
             'instabil' if geo[k]['unstable_ts'] == '1' else 'stabil'))
print()

# ------------------------------------------------------------ Kreuztabelle
print('STATUSKLASSE GEGEN RKS-STABILITAET AM MODELL-TS')
print('-' * 90)
KL = [('Marker + Kriterium', met & marker),
      ('Marker, Kriterium nein', disk),
      ('kein Marker', ohne)]
print('%-24s %8s %10s %10s' % ('', 'gesamt', 'stabil', 'instabil'))
for nm, s in KL:
    u = sum(1 for k in s if geo[k]['unstable_ts'] == '1')
    print('%-24s %8d %10d %10d' % (nm, len(s), len(s) - u, u))
tot_u = sum(1 for k in geo if geo[k]['unstable_ts'] == '1')
print('%-24s %8d %10d %10d' % ('alle', len(geo), len(geo) - tot_u, tot_u))
print()

# ------------------------------------------------------------- Robustheit
print('ROBUSTHEIT  --  aendert der Ausschluss der 23 Zeilen die Trennung?')
print('-' * 90)


def med(keys, col, unst):
    v = [float(geo[k][col]) for k in keys
         if (geo[k]['unstable_ts'] == '1') == unst]
    return float(np.median(v)), len(v)


print('%-22s %-14s %10s %10s %10s' % ('Teilmenge', 'Groesse', 'stabil',
                                      'instabil', 'Faktor'))
for nm, keys in (('alle 135', set(geo)), ('nur criterion_met', met)):
    for col in ('f_model_max', 'f_dft_max'):
        a, na = med(keys, col, False)
        b, nb = med(keys, col, True)
        print('%-22s %-14s %10.4f %10.4f %10.2f'
              % (nm, '%s (%d/%d)' % (col, na, nb), a, b, b / a))
fa = med(set(geo), 'f_dft_max', True)[0] / med(set(geo), 'f_dft_max', False)[0]
fb = med(met, 'f_dft_max', True)[0] / med(met, 'f_dft_max', False)[0]
print()
print('   Die Trennung der DFT-Restkraft zwischen stabil und instabil geht '
      'von Faktor %.2f auf %.2f,' % (fa, fb))
print('   wenn nur die %d Zeilen mit erfuelltem Kriterium betrachtet werden. '
      'Die Aussage der Figuren' % len(met))
print('   haengt damit %s an den %d ausgeschlossenen Zeilen.'
      % ('nicht' if abs(fa - fb) / fa < 0.15 else 'sehr wohl', len(geo) - len(met)))
print()

# ------------------------------------------------------------ Pruefungen
print('Pruefungen')
check(len(rows) == 135, 'n = 135 (%d)' % len(rows))
check(not nolog, 'neb.log fuer alle Zeilen gefunden'
      + ('' if not nolog else ': fehlt bei ' + ', '.join(nolog)))
check(len(marker) == 133, 'converged_marker = 1 in genau 133 Zeilen (%d)'
      % len(marker))
check(len(met) == 112, 'criterion_met = 1 in genau 112 Zeilen (%d)' % len(met))
check(disk == ERWARTETE_DISKREPANZ,
      'die %d Diskrepanz-Zeilen sind unveraendert' % len(ERWARTETE_DISKREPANZ)
      + ('' if disk == ERWARTETE_DISKREPANZ else
         '  --  neu: %s   weggefallen: %s'
         % (sorted(disk - ERWARTETE_DISKREPANZ),
            sorted(ERWARTETE_DISKREPANZ - disk))))
check(ohne == ERWARTET_OHNE_MARKER,
      'ohne Marker sind unveraendert %s' % sorted(ERWARTET_OHNE_MARKER))
check(met <= marker,
      'kein Lauf erfuellt das Kriterium ohne Marker')

print()
print('geschrieben: results/neb_runs.csv (%d Zeilen)' % len(rows))
if fails:
    raise SystemExit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))
