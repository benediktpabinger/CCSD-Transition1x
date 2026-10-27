"""Der gespeicherte Uebergangszustand JEDER Transition1x-Reaktion als eine JSON-Datei.

Verallgemeinert pipeline/omol25_dump_ts45.py (45 Reaktionen des Testsplits)
auf alle Reaktionen aller drei Splits. Gelesen wird wie in
pipeline/extract_t1x_ts.py die Gruppe 'transition_state' jeder Reaktion --
NICHT das Feld 'positions', das die ganze Optimierungshistorie enthaelt.

Je Reaktion: Split, Summenformel (H5-Gruppenname), Kernladungen, Koordinaten,
Energie und groesste Kraftkomponente auf wB97x/6-31G(d) aus dem Datensatz
selbst (die Nullreferenz der Leiter, Tabelle 4.3 Stufe 1).

Schreibt ~/t1x_ts_all.json. Gehoert zu pipeline/omol25_stream_scan_ts.py.
Laeuft im SLURM-Job pipeline/job_omol25_ts_scan.sh, nicht auf dem Login-Knoten.

    python pipeline/omol25_dump_ts_all.py
"""
import collections
import json
import os
import sys

import h5py
import numpy as np

H = os.path.expanduser('~')
H5 = f'{H}/data/Transition1x.h5'
OUT = f'{H}/t1x_ts_all.json'

f = h5py.File(H5, 'r')
out = {}
missing = []
per_split = collections.Counter()
# Die Gruppe 'data' ist nur die Vereinigung von train/val/test (10073
# Reaktionen) und wuerde jede Reaktion doppelt liefern.
for split in ('train', 'val', 'test'):
    for formula in f[split]:
        for rxn in f[split][formula]:
            g = f[split][formula][rxn]
            if rxn in out:
                sys.exit('ABBRUCH: %s kommt doppelt vor (%s und %s)'
                         % (rxn, out[rxn]['split'], split))
            if 'transition_state' not in g:
                missing.append((split, rxn))
                continue
            ts = g['transition_state']
            z = np.asarray(ts['atomic_numbers']).astype(int)
            x = np.asarray(ts['positions'])[0]
            e = float(np.asarray(ts['wB97x_6-31G(d).energy'])[0])
            fo = np.asarray(ts['wB97x_6-31G(d).forces'])[0]
            out[rxn] = dict(split=split, formula=formula, n=int(len(z)),
                            numbers=z.tolist(), positions=x.tolist(),
                            e_ref=e, fmax_ref=float(np.abs(fo).max()))
            per_split[split] += 1

json.dump(out, open(OUT, 'w'))
print('%d Reaktionen -> %s' % (len(out), OUT))
for s, n in sorted(per_split.items()):
    print('   %-6s %6d' % (s, n))
if missing:
    print('%d Reaktionen ohne transition_state-Gruppe:' % len(missing))
    for s, r in missing[:20]:
        print('   %s/%s' % (s, r))
