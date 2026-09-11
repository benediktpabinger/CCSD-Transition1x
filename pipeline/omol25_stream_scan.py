"""Scannt entpackte aselmdb-Dateien, waehrend sie ankommen, und loescht sie.

Gehoert zu pipeline/omol25_stream_train.sh. tar entpackt der Reihe nach; eine
Datei ist fertig, sobald eine spaetere aufgetaucht ist. Die jeweils zwei
juengsten Dateien bleiben deshalb liegen, bis der Download beendet ist. Danach
liegt nie mehr als eine Handvoll Dateien gleichzeitig auf Platte -- der volle
Trainingssplit hat 456 GB, die Quote gibt das nicht her.

    python omol25_stream_scan.py <verzeichnis> <rc-datei>

DER FILTER  Pro Zeile faellt nur an: einmal zlib entpacken, zwei kurze
Regex-Suchen (data_id, source) und ein Vergleich der Summenformel gegen eine
Menge. Die 45 Reaktionsnamen werden nicht einzeln gesucht -- der Name steht in
source, und ein Regex zieht ihn in einem Schritt heraus. Eine Schleife ueber
50 Muster je Zeile waere bei 100 Millionen Zeilen zehnmal zu langsam.

Vier Prozesse arbeiten je eine ganze Datei ab; das haelt den Scan vor dem
Download, sonst laufen die Dateien auf.

Schreibt ~/omol25_hits_train.jsonl und ~/omol25_census_train.json, gleiche
Felder wie pipeline/omol25_scan_t1x.py.
"""
import collections
import glob
import json
import os
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import lmdb

H = '/home/energy/s242862'
SKIP = {b'nextid', b'metadata', b'deleted_ids'}
KEEP = ('source', 'reference_source', 'data_id', 'charge', 'spin',
        'num_atoms', 'num_electrons', 'n_scf_steps', 'unrestricted',
        's_squared', 's_squared_dev', 'homo_lumo_gap', 'composition',
        'nl_energy')
NWORK = 4

ID_RE = re.compile(rb'"data_id": "([^"]*)"')
SRC_RE = re.compile(rb'"source": "([^"/]*)/([^"]*)"')
COMP_RE = re.compile(rb'"composition": "([^"]*)"')
RXN_RE = re.compile(rb't1x_(rxn\d+)_')

targets = json.load(open(f'{H}/t1x_ts_45.json'))
FORMSET = {v['formula'].encode() for v in targets.values()}
RXNSET = set(targets)


def scan_file(path):
    """Eine aselmdb: Zeilenzahl, data_id-Verteilung, Treffer."""
    n = 0
    census = collections.Counter()
    hits = []
    try:
        env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    except lmdb.Error as e:
        return 0, census, hits, 'nicht lesbar: %s' % e
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
            census[(i.group(1).decode() if i else '?',
                    m.group(1).decode() if m else '?')] += 1
            c = COMP_RE.search(s)
            byform = bool(c) and c.group(1) in FORMSET
            byname = False
            if m is not None:
                r = RXN_RE.search(m.group(2))
                byname = bool(r) and r.group(1).decode() in RXNSET
            if not (byform or byname):
                continue
            dd = json.loads(s)
            dat = dd.get('data', {})
            rec = {q: dat.get(q) for q in KEEP}
            rec['by_name'] = int(byname)
            rec['file'] = os.path.basename(path)
            rec['key'] = k.decode()
            rec['energy'] = dd.get('energy')
            for q in ('numbers', 'positions', 'forces'):
                a = dd.get(q)
                rec[q] = a['__ndarray__'][2] if isinstance(a, dict) else a
            hits.append(rec)
    env.close()
    return n, census, hits, None


def main():
    d, rcfile = sys.argv[1], sys.argv[2]
    print('gesucht: %d Summenformeln, %d Reaktionsnamen, %d Prozesse'
          % (len(FORMSET), len(RXNSET), NWORK), flush=True)

    census = collections.Counter()
    nhit = 0
    n = 0
    seen = set()
    pending = {}
    t0 = time.time()
    fh = open(f'{H}/omol25_hits_train.jsonl', 'w')
    ex = ProcessPoolExecutor(max_workers=NWORK)

    while True:
        running = not os.path.exists(rcfile)
        files = sorted((p for p in glob.glob(os.path.join(d, '**', '*.aselmdb'),
                                             recursive=True) if p not in seen),
                       key=lambda p: os.path.getmtime(p))
        ready = files[:-2] if running else files
        for p in ready:
            seen.add(p)
            pending[p] = ex.submit(scan_file, p)

        for p, fut in list(pending.items()):
            if not fut.done():
                continue
            del pending[p]
            cn, cc, hs, err = fut.result()
            if err:
                print('  %s  %s' % (os.path.basename(p), err), flush=True)
            n += cn
            census.update(cc)
            nhit += len(hs)
            for r in hs:
                fh.write(json.dumps(r) + '\n')
            fh.flush()
            for q in (p, p + '-lock'):
                try:
                    os.remove(q)
                except OSError:
                    pass
            print('  %-20s %4d Dateien  %11d Zeilen  %5d Treffer  %6.0f s'
                  % (os.path.basename(p), len(seen), n, nhit,
                     time.time() - t0), flush=True)

        if not running and not pending and not ready:
            break
        time.sleep(5)

    ex.shutdown()
    fh.close()
    json.dump({'%s|%s' % k: v for k, v in census.items()},
              open(f'{H}/omol25_census_train.json', 'w'), indent=1)

    print()
    print('%d Dateien, %d Zeilen, %d Treffer' % (len(seen), n, nhit))
    print()
    print('%-20s %-24s %12s' % ('data_id', 'source[0]', 'n'))
    for (a, b), v in sorted(census.items(), key=lambda x: -x[1])[:40]:
        print('%-20s %-24s %12d' % (a, b, v))


if __name__ == '__main__':
    main()
