"""N_FOD am gespeicherten Transition1x-Uebergangszustand JEDER Reaktion.

Dieselbe Rechnung wie pipeline/screen_fod.py (PBE/def2-SVP, RKS, Fermi-
Schmierung bei T_el = 5000 K, N_FOD = sum |n_i - n_i^0|), nur fuer alle 10073
Reaktionen aus ~/t1x_ts_all.json statt fuer die 279 Test-Referenzen. Die
Rechenfunktion wird aus screen_fod.py importiert, damit die Einstellungen
nicht auseinanderlaufen (docs/paper_decisions.md D05, D07).

Ein Aufruf bearbeitet einen Block von Reaktionen (Array-Task), mit einem
Prozess-Pool im Knoten; je Reaktion eine JSON-Datei in OUT_DIR, vorhandene
werden uebersprungen (Neustart-sicher).

    python pipeline/fod_all_ts.py --start 0 --count 250 --workers 6
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

H = os.path.expanduser('~')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_fod import compute_fod, T_EL  # noqa: E402

SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
TS_JSON = f'{H}/t1x_ts_all.json'
XYZ_DIR = f'{H}/t1x_ts_xyz_all'
OUT_DIR = f'{H}/fod_ts_all'


def write_xyz(rx, t):
    p = f'{XYZ_DIR}/{rx}.xyz'
    if not os.path.exists(p):
        with open(p, 'w') as fh:
            fh.write(f"{t['n']}\n")
            fh.write(f"rxn={rx} split={t['split']} stored Transition1x TS wB97x/6-31G(d)\n")
            for z, (x, y, w) in zip(t['numbers'], t['positions']):
                fh.write(f'{SYM[z]:2s} {x:16.8f} {y:16.8f} {w:16.8f}\n')
    return p


def run_one(args):
    rx, t = args
    out = f'{OUT_DIR}/{rx}.json'
    if os.path.exists(out):
        return rx, 'vorhanden'
    t0 = time.time()
    try:
        res, err = compute_fod(write_xyz(rx, t))
    except Exception as e:          # ein Fehler darf den Block nicht stoppen
        res, err = None, f'{type(e).__name__}: {e}'
    d = dict(rxn=rx, split=t['split'], formula=t['formula'],
             geometry='stored Transition1x TS', level='PBE/def2-SVP RKS',
             T_el_K=T_EL, seconds=round(time.time() - t0, 1))
    d.update(res if res else dict(error=err))
    with open(out, 'w') as fh:
        json.dump(d, fh, indent=1)
    return rx, ('FEHLER ' + err) if err else 'N_FOD = %.4f  (%.0f s)' % (res['nfod'], d['seconds'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--count', type=int, required=True)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(XYZ_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = json.load(open(TS_JSON))
    keys = sorted(ts)
    block = [(rx, ts[rx]) for rx in keys[a.start:a.start + a.count]]
    print(f'Block {a.start}..{a.start + len(block) - 1} von {len(keys)}: '
          f'{len(block)} Reaktionen, {a.workers} Prozesse, '
          f'OMP {os.environ.get("OMP_NUM_THREADS")}', flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for rx, msg in pool.imap_unordered(run_one, block):
            print(f'  {rx}: {msg}', flush=True)
    print(f'ENDE Block {a.start}: {len(block)} Reaktionen in {(time.time() - t0) / 60:.1f} min',
          flush=True)
