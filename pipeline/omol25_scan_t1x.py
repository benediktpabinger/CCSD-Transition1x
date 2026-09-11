"""Sucht die 45 Label-Geometrien in den freigegebenen OMol25-Splits.

OMol25 legt jede Struktur als zlib-komprimiertes JSON in einer aselmdb ab
(ase-db-backends, LMDBDatabase). Der Datensatz-Herkunftsschluessel steht in
row['data']['data_id'], die Summenformel in row['data']['composition'] --
gleiche Schreibweise wie hier gebaut: Elemente alphabetisch, jeweils mit
Anzahl, also C3H5N1O2.

Transition1x heisst in OMol25 data_id 'trans1x', und die Herkunft traegt die
Reaktion im Klartext:

    trans1x/t1x_rxn9839_1010_943_0_1/orca.tar.zst

also t1x_<rxn>_<zwei Zaehlern>_<Ladung>_<Spin>. Damit ist jede unserer 45
Reaktionen namentlich zu finden; welcher der Rahmen der Uebergangszustand ist,
entscheidet danach der RMSD.

Der Scan laeuft in zwei Stufen, weil ein voller JSON-Parse aller 100M Zeilen
zu teuer waere:
  1  dekomprimieren und auf zwei Muster testen -- eine der fuenf
     Summenformeln, oder einer der 45 Reaktionsnamen (reine Substringsuche,
     rund 20000 Zeilen/s je Kern)
  2  nur die Treffer vollstaendig parsen und mit Geometrie, Energie, Kraft
     und Metadaten nach ~/omol25_hits_<split>.jsonl schreiben

Nebenbei faellt die Verteilung der data_id an; sie beantwortet die Frage, ob
Transition1x in dem jeweiligen Split ueberhaupt vertreten ist.

    python pipeline/omol25_scan_t1x.py <split-verzeichnis> [...]
"""
import collections
import glob
import json
import os
import re
import sys
import time
import zlib

import lmdb

H = '/home/energy/s242862'
SKIP = {b'nextid', b'metadata', b'deleted_ids'}
KEEP = ('source', 'reference_source', 'data_id', 'charge', 'spin',
        'num_atoms', 'num_electrons', 'n_scf_steps', 'unrestricted',
        's_squared', 's_squared_dev', 'homo_lumo_gap', 'composition',
        'nl_energy')

ID_RE = re.compile(rb'"data_id": "([^"]*)"')
SRC_RE = re.compile(rb'"source": "([^"/]*)/([^"]*)"')
COMP_RE = re.compile(rb'"composition": "([^"]*)"')
RXN_RE = re.compile(rb't1x_(rxn\d+)_')

targets = json.load(open(f'{H}/t1x_ts_45.json'))
FORM = sorted({v['formula'] for v in targets.values()})
FORMSET = {f.encode() for f in FORM}
RXN = sorted(targets)
RXNSET = set(RXN)
print('gesucht: %d Summenformeln (%s), %d Reaktionsnamen'
      % (len(FORM), ', '.join(FORM), len(RXN)))

dirs = sys.argv[1:]
if not dirs:
    sys.exit('ABBRUCH: kein Split-Verzeichnis angegeben')

files = []
for d in dirs:
    files += sorted(glob.glob(os.path.join(d, '**', '*.aselmdb'),
                              recursive=True))
print('%d aselmdb-Dateien' % len(files))
if not files:
    sys.exit('ABBRUCH: keine aselmdb gefunden')

census = collections.Counter()
hits = []
n = 0
t0 = time.time()
for j, f in enumerate(files):
    try:
        env = lmdb.open(f, subdir=False, readonly=True, lock=False)
    except lmdb.Error as e:
        print('  uebersprungen (nicht lesbar): %s  %s' % (f, e))
        continue
    with env.begin() as txn:
        for k, v in txn.cursor():
            if k in SKIP:
                continue
            try:
                s = zlib.decompress(v)
            except zlib.error:
                continue
            n += 1
            i = ID_RE.search(s)
            m = SRC_RE.search(s)
            census[i.group(1).decode() if i else '?'] += 1
            c = COMP_RE.search(s)
            byform = bool(c) and c.group(1) in FORMSET
            byname = False
            if m is not None:
                r = RXN_RE.search(m.group(2))
                byname = bool(r) and r.group(1).decode() in RXNSET
            if not (byform or byname):
                continue
            d = json.loads(s)
            dat = d.get('data', {})
            rec = {q: dat.get(q) for q in KEEP}
            rec['by_name'] = int(byname)
            rec['file'] = os.path.relpath(f, os.path.dirname(dirs[0]))
            rec['key'] = k.decode()
            rec['energy'] = d.get('energy')
            for q in ('numbers', 'positions', 'forces'):
                a = d.get(q)
                rec[q] = a['__ndarray__'][2] if isinstance(a, dict) else a
            hits.append(rec)
    env.close()
    if (j + 1) % 25 == 0 or j + 1 == len(files):
        print('  %4d/%d  %10d Zeilen  %6d Treffer  %5.0f s'
              % (j + 1, len(files), n, len(hits), time.time() - t0), flush=True)

tag = '_'.join(os.path.basename(os.path.normpath(d)) for d in dirs)
out = f'{H}/omol25_hits_{tag}.jsonl'
with open(out, 'w') as fh:
    for r in hits:
        fh.write(json.dumps(r) + '\n')
json.dump(dict(census), open(f'{H}/omol25_census_{tag}.json', 'w'), indent=1)

print()
print('%d Zeilen gelesen, %d Treffer' % (n, len(hits)))
byname = collections.Counter()
for r in hits:
    if r.get('by_name'):
        for q in RXN:
            if 't1x_%s_' % q in (r['source'] or ''):
                byname[q] += 1
print('%d der 45 Reaktionen namentlich in diesem Split, %d Rahmen insgesamt'
      % (len(byname), sum(byname.values())))
if byname:
    print('   ' + ', '.join('%s:%d' % kv for kv in sorted(byname.items())))
print(out)
print()
print('data_id im Split %s' % tag)
for k, v in census.most_common():
    print('   %-28s %10d' % (k, v))
print()
print('data_id der Treffer')
for k, v in collections.Counter(r['data_id'] for r in hits).most_common():
    print('   %-28s %10d' % (k, v))
