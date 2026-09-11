"""Die 45 Label-Geometrien als eine JSON-Datei, fuer den Abgleich mit OMol25.

Liest ~/t1x_ts/<rxn>.xyz -- die Uebergangszustaende aus Transition1x selbst,
Gruppe 'transition_state', extrahiert von pipeline/extract_t1x_ts.py -- und
schreibt Kernladungen, Koordinaten und Summenformel nach ~/t1x_ts_45.json.

Die Summenformel in derselben Schreibweise wie OMol25 sie in
row['data']['composition'] fuehrt: Elemente alphabetisch, jedes mit Anzahl,
auch der Anzahl eins. Also C3H5N1O2, nicht C3H5NO2. Darauf filtert
pipeline/omol25_scan_t1x.py.

    python pipeline/omol25_dump_ts45.py
"""
import glob
import json
import os

from ase.data import chemical_symbols

H = '/home/energy/s242862'

out = {}
for p in sorted(glob.glob(f'{H}/t1x_ts/rxn*.xyz')):
    rx = os.path.basename(p)[:-4]
    L = open(p).read().split('\n')
    n = int(L[0])
    Z, R = [], []
    for line in L[2:2 + n]:
        f = line.split()
        Z.append(chemical_symbols.index(f[0]))
        R.append([float(x) for x in f[1:4]])
    cnt = {}
    for z in Z:
        cnt[chemical_symbols[z]] = cnt.get(chemical_symbols[z], 0) + 1
    out[rx] = dict(numbers=Z, positions=R, n=n,
                   formula=''.join(f'{s}{cnt[s]}' for s in sorted(cnt)),
                   comment=L[1])

json.dump(out, open(f'{H}/t1x_ts_45.json', 'w'))
print('%d Reaktionen -> %s/t1x_ts_45.json' % (len(out), H))
forms = sorted({v['formula'] for v in out.values()})
print('%d Summenformeln:' % len(forms))
for f in forms:
    print('   %-12s %s' % (f, ' '.join(k for k, v in sorted(out.items())
                                       if v['formula'] == f)))
print('Atomzahlen: %s' % sorted({v['n'] for v in out.values()}))
