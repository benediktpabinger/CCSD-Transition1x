"""Scannt den OMol25-Trainingssplit nach dem Uebergangszustand JEDER
Transition1x-Reaktion, waehrend die aselmdb-Dateien ankommen, und loescht die
heruntergeladenen Dateien danach wieder (nur die; nichts Eigenes).

Verallgemeinert pipeline/omol25_stream_scan.py (45 Reaktionen, Filter ueber
Summenformel oder Name) auf alle Reaktionen. Der Filter ist jetzt die
Geometrie selbst: eine Zeile wird nur behalten, wenn sie trans1x ist UND ihre
Koordinaten mit dem gespeicherten Uebergangszustand ihrer Reaktion
(~/t1x_ts_all.json, aus pipeline/omol25_dump_ts_all.py) uebereinstimmen,
RMSD < 1e-5 A. Alle anderen ~9.5 Millionen Transition1x-Bilder werden nur
gezaehlt. Das Ergebnis sind rund 10 000 Zeilen, eine je Reaktion, mit <S^2>,
Energie und Kraeften des OMol25-Labels an genau der Geometrie, auf der die
Modelle trainiert wurden.

Was das liefert, ohne eine einzige DFT-Rechnung:
    s_squared   closed-shell oder broken-symmetry an jedem Label-TS
    forces      die Restkraft auf der unrestricted Flaeche am Label-TS,
                also Stufe 3 der Leiter (Tabelle 4.3) fuer den ganzen Datensatz

Zuordnung wie in pipeline/omol25_label_compare.py: Reaktionsname aus source
(trans1x/t1x_<rxn>_..., ein Regex), dann direkter RMSD bei gleicher
Atomreihenfolge; nur wenn der nicht passt, ein billiger rotationsinvarianter
Vortest (sortierte Abstaende zum Schwerpunkt) und erst dann Kabsch. OMol25 hat
die Koordinaten unveraendert uebernommen; der exakte Rahmen liegt bei ~5e-7 A
(Speichergenauigkeit), der naechste andere NEB-Rahmen bei >= 5e-5 A (Vorprobe,
Job 11011077). Eine Schwelle von 1e-3 wie in omol25_label_compare.py haette
bis zu 28 Nachbarbilder je Reaktion mitgenommen.

Diagnose fuer nicht gefundene Reaktionen: je Reaktion wird der kleinste RMSD
aller ihrer Bilder zum gespeicherten TS mitgefuehrt, auch wenn keines die
Schwelle unterschreitet. So laesst sich nach dem Lauf unterscheiden, ob der
Rahmen in OMol25 ganz fehlt (bester RMSD gross) oder nur knapp ueber der
Schwelle liegt -- ohne den Download zu wiederholen.

    python omol25_stream_scan_ts.py <verzeichnis> <rc-datei>
    python omol25_stream_scan_ts.py --test ~/omol25_hits_train.jsonl

--test wendet nur die Zuordnung auf eine vorhandene Trefferdatei des alten
Scanners an (alle Bilder der 45 Reaktionen) und prueft sie gegen
results/omol25_label_compare.csv: dieselben 44 Reaktionen muessen gefunden
werden. Der Test ist dazu da, den Filter zu pruefen, BEVOR der fuenfstuendige
Download laeuft.

Schreibt ~/omol25_ts_all.jsonl (Treffer) und ~/omol25_census_ts_all.json
(je Reaktion: n_frames, best_rmsd_A).
"""
import collections
import csv
import glob
import json
import os
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

H = os.path.expanduser('~')
TS_JSON = f'{H}/t1x_ts_all.json'
OUT_HITS = f'{H}/omol25_ts_all.jsonl'
OUT_CENSUS = f'{H}/omol25_census_ts_all.json'
RMSD_MATCH = 1e-5        # A, siehe Docstring
PRE_TOL = 0.01           # A, Vortest vor Kabsch
NWORK = 8
SKIP = {b'nextid', b'metadata', b'deleted_ids'}
KEEP = ('source', 'reference_source', 'data_id', 'charge', 'spin',
        'num_atoms', 'num_electrons', 'n_scf_steps', 'unrestricted',
        's_squared', 's_squared_dev', 'homo_lumo_gap', 'composition',
        'nl_energy')

SRC_RE = re.compile(rb'"source": "([^"/]*)/([^"]*)"')
RXN_RE = re.compile(rb't1x_(rxn\d+)_')

TS = {rx: (np.array(v['numbers']), np.array(v['positions']))
      for rx, v in json.load(open(TS_JSON)).items()}


def kabsch_rmsd(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    V, S, W = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(V @ W))
    R = V @ np.diag([1.0, 1.0, d]) @ W
    return float(np.sqrt((((A @ R) - B) ** 2).sum(1).mean()))


def match(rx, Z, R):
    """(Treffer-RMSD oder None, RMSD dieses Bildes zum TS der Reaktion).

    Der zweite Wert wird auch fuer Nicht-Treffer geliefert, damit eine nicht
    gefundene Reaktion spaeter unterscheidbar ist: Rahmen fehlt ganz oder
    liegt knapp ueber der Schwelle."""
    t = TS.get(rx)
    if t is None:
        return None, None
    Zt, Rt = t
    if Z.shape != Zt.shape or not (Z == Zt).all():
        return None, None
    d = float(np.sqrt(((R - Rt) ** 2).sum(1).mean()))
    if d < RMSD_MATCH:
        return d, d
    a = np.sort(np.linalg.norm(R - R.mean(0), axis=1))
    b = np.sort(np.linalg.norm(Rt - Rt.mean(0), axis=1))
    if np.abs(a - b).max() > PRE_TOL:
        return None, d
    k = kabsch_rmsd(R, Rt)
    return (k if k < RMSD_MATCH else None), min(d, k)


def arr(a):
    return a['__ndarray__'][2] if isinstance(a, dict) else a


def record(dd, rx, d, where):
    dat = dd.get('data', {})
    rec = {q: dat.get(q) for q in KEEP}
    rec['rxn'] = rx
    rec['rmsd_A'] = d
    rec.update(where)
    rec['energy'] = dd.get('energy')
    for q in ('numbers', 'positions', 'forces'):
        rec[q] = arr(dd.get(q))
    return rec


def scan_file(path):
    """Eine aselmdb: Zeilen, trans1x-Zeilen je Reaktion, bester RMSD, Treffer."""
    import lmdb
    n = 0
    per_rxn = collections.Counter()
    best = {}
    hits = []
    try:
        env = lmdb.open(path, subdir=False, readonly=True, lock=False)
    except lmdb.Error as e:
        return 0, per_rxn, best, hits, 'nicht lesbar: %s' % e
    with env.begin() as txn:
        for k, v in txn.cursor():
            if k in SKIP:
                continue
            try:
                s = zlib.decompress(v)
            except zlib.error:
                continue
            n += 1
            m = SRC_RE.search(s)
            if m is None or m.group(1) != b'trans1x':
                continue
            r = RXN_RE.search(m.group(2))
            rx = r.group(1).decode() if r else '?'
            per_rxn[rx] += 1
            if rx not in TS:
                continue
            dd = json.loads(s)
            Z = np.array(arr(dd.get('numbers')))
            R = np.array(arr(dd.get('positions'))).reshape(-1, 3)
            d, b = match(rx, Z, R)
            if b is not None and b < best.get(rx, np.inf):
                best[rx] = b
            if d is None:
                continue
            hits.append(record(dd, rx, d,
                               dict(file=os.path.basename(path),
                                    key=k.decode())))
    env.close()
    return n, per_rxn, best, hits, None


def test(jsonl):
    """Die Zuordnung an der alten Trefferdatei der 45 Reaktionen pruefen."""
    ref = {r['rxn']: float(r['rmsd_A']) for r in csv.DictReader(
        open(os.path.join(os.path.dirname(__file__), '..', 'results',
                          'omol25_label_compare.csv')))}
    found = collections.defaultdict(list)
    seen = collections.Counter()
    best = {}
    n = 0
    t0 = time.time()
    for line in open(jsonl):
        h = json.loads(line)
        n += 1
        m = RXN_RE.search((h.get('source') or '').encode())
        if not m:
            continue
        rx = m.group(1).decode()
        seen[rx] += 1
        Z = np.array(h['numbers'])
        R = np.array(h['positions']).reshape(-1, 3)
        d, b = match(rx, Z, R)
        if b is not None and b < best.get(rx, np.inf):
            best[rx] = b
        if d is not None:
            found[rx].append((d, h.get('s_squared')))
    print('%d Zeilen gelesen in %.0f s, %d Reaktionen mit trans1x-Zeilen'
          % (n, time.time() - t0, len(seen)))
    bad = []
    for rx in sorted(ref):
        f = found.get(rx, [])
        print('   %-9s %5d Bilder  %d Treffer  rmsd %s  <S^2> %s  bester %.1e'
              % (rx, seen[rx], len(f),
                 ' '.join('%.1e' % d for d, _ in f) or '-',
                 ' '.join('%.4f' % s for _, s in f) or '-',
                 best.get(rx, np.inf)))
        if not f:
            bad.append(rx)
    extra = sorted(set(found) - set(ref))
    print()
    print('%d von %d Referenzreaktionen gefunden' % (len(ref) - len(bad), len(ref)))
    if bad:
        print('   nicht gefunden: %s' % ', '.join(bad))
    if extra:
        print('   zusaetzlich gefunden (nicht in der Referenz): %d Reaktionen'
              % len(extra))
    multi = {rx: len(f) for rx, f in found.items() if len(f) > 1}
    if multi:
        print('   mehrere Treffer je Reaktion: %s' % multi)
    sys.exit(1 if bad else 0)


def main():
    if sys.argv[1] == '--test':
        test(sys.argv[2])
    d, rcfile = sys.argv[1], sys.argv[2]
    print('%d Uebergangszustaende geladen, %d Prozesse' % (len(TS), NWORK),
          flush=True)

    per_rxn = collections.Counter()
    best = {}
    nhit = 0
    n = 0
    seen = set()
    pending = {}
    t0 = time.time()
    fh = open(OUT_HITS, 'w')
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
            cn, cc, cb, hs, err = fut.result()
            if err:
                print('  %s  %s' % (os.path.basename(p), err), flush=True)
            n += cn
            per_rxn.update(cc)
            for rx, b in cb.items():
                if b < best.get(rx, np.inf):
                    best[rx] = b
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
    json.dump({rx: dict(n_frames=c, best_rmsd_A=best.get(rx))
               for rx, c in per_rxn.items()},
              open(OUT_CENSUS, 'w'), indent=1)

    hit_rxn = set()
    for line in open(OUT_HITS):
        hit_rxn.add(json.loads(line)['rxn'])
    print()
    print('%d Dateien, %d Zeilen, %d trans1x-Zeilen in %d Reaktionen'
          % (len(seen), n, sum(per_rxn.values()), len(per_rxn)))
    print('%d Treffer, %d von %d Uebergangszustaenden gefunden'
          % (nhit, len(hit_rxn), len(TS)))
    print('%d Reaktionen von OMol25 ohne Gegenstueck im H5'
          % sum(1 for rx in per_rxn if rx not in TS))
    miss = sorted((best.get(rx, np.inf), rx) for rx in TS if rx not in hit_rxn)
    if miss:
        print('nicht gefunden: %d Reaktionen, bester RMSD von %.1e bis %.1e A'
              % (len(miss), miss[0][0], miss[-1][0]))


if __name__ == '__main__':
    main()
