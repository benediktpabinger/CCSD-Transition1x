"""Die Label-Uebergangszustaende der 45 Reaktionen aus Transition1x.h5 als xyz.

Das ist die Geometrie, auf der die Modelle trainiert sind: der
Uebergangszustand des Transition1x-NEB auf wB97x/6-31G(d), so wie er im
Datensatz steht. Kein eigener NEB, keine Nachoptimierung.

WICHTIG, welche Gruppe gelesen wird
    Jede Reaktion im H5 hat ein Feld 'positions' -- das ist NICHT das
    konvergierte Band, sondern die gesamte Optimierungshistorie (im Testsplit
    138 bis 4274 Bilder). Das Energiemaximum darueber ist ein fruehes,
    unrelaxiertes Bild und liegt teils Elektronenvolt daneben. Der
    Uebergangszustand steht in der eigenen Gruppe 'transition_state'; die wird
    hier gelesen.

Der xyz-Kommentar traegt Energie und groesste Kraftkomponente des Datensatzes
selbst. Damit ist ohne neue Rechnung bekannt, wie stationaer der Label-TS auf
SEINEM EIGENEN Niveau ist -- die Nullreferenz fuer die Frage, wieviel der
spaeter gemessenen Restkraft der Niveauwechsel ist.

Schreibt ~/t1x_ts/<rxn>.xyz und ~/t1x_ts_tasks.txt (rxn:pfad:NEL).
Bricht ab, wenn eine Reaktion fehlt oder nicht neutral-geschlossenschalig ist.
"""
import csv
import os

import h5py
import numpy as np

H = '/home/energy/s242862'
H5 = f'{H}/data/Transition1x.h5'
OUT = f'{H}/t1x_ts'
SPLIT = 'test'
Z2S = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl'}
EVA_REF = 1.0            # der Datensatz speichert Kraefte bereits in eV/A

want = [r['rxn'] for r in
        csv.DictReader(open(f'{H}/results/paper_reactions.csv'))]
os.makedirs(OUT, exist_ok=True)

f = h5py.File(H5, 'r')
found, lines, bad = {}, [], []
for formula in f[SPLIT]:
    for rxn in f[SPLIT][formula]:
        if rxn in want:
            found[rxn] = f[SPLIT][formula][rxn]

missing = [r for r in want if r not in found]

for rxn in want:
    if rxn not in found:
        continue
    g = found[rxn]
    if 'transition_state' not in g:
        bad.append((rxn, 'keine transition_state-Gruppe'))
        continue
    ts = g['transition_state']
    z = np.asarray(ts['atomic_numbers'])
    x = np.asarray(ts['positions'])[0]
    e = float(np.asarray(ts['wB97x_6-31G(d).energy'])[0])
    fo = np.asarray(ts['wB97x_6-31G(d).forces'])[0]
    nel = int(z.sum())
    if nel % 2:
        bad.append((rxn, 'ungerade Elektronenzahl %d' % nel))
        continue
    unknown = sorted({int(v) for v in z} - set(Z2S))
    if unknown:
        bad.append((rxn, 'unbekannte Ordnungszahl %s' % unknown))
        continue
    p = f'{OUT}/{rxn}.xyz'
    with open(p, 'w') as fh:
        fh.write('%d\n' % len(z))
        fh.write('rxn=%s split=%s level=wB97x/6-31G(d) E_ref_eV=%.6f '
                 'maxcomp_ref_eV_A=%.4f maxnorm_ref_eV_A=%.4f\n'
                 % (rxn, SPLIT, e, float(np.abs(fo).max()),
                    float(np.linalg.norm(fo, axis=1).max())))
        for zi, xi in zip(z, x):
            fh.write('%-2s %14.8f %14.8f %14.8f\n'
                     % (Z2S[int(zi)], xi[0], xi[1], xi[2]))
    lines.append('%s:%s:%d' % (rxn, p, nel))

open(f'{H}/t1x_ts_tasks.txt', 'w').write('\n'.join(lines) + '\n')

print('DIE 45 LABEL-UEBERGANGSZUSTAENDE')
print('=' * 70)
print('gesucht %d, im Split %s gefunden %d, geschrieben %d'
      % (len(want), SPLIT, len(found), len(lines)))
if missing:
    print('   NICHT GEFUNDEN: %s' % missing)
if bad:
    print('   NICHT VERWENDBAR:')
    for r, why in bad:
        print('      %-9s %s' % (r, why))
print('   Taskdatei: t1x_ts_tasks.txt')
if missing or bad or len(lines) != len(want):
    raise SystemExit('ABBRUCH: %d fehlen, %d unbrauchbar'
                     % (len(missing), len(bad)))
