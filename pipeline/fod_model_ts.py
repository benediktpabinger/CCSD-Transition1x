"""N_FOD at the transition state each model found (freezing string + Sella), D45.

The same calculation as pipeline/screen_fod.py and pipeline/fod_all_ts.py (PBE/def2-SVP, RKS,
Fermi smearing at T_el = 5000 K, N_FOD = sum |n_i - n_i^0|), at every structure with a D37 force
row: ~/neb_fsm/fsm/<model>/<rxn>/transition_state.xyz. Carries out D07 (N_FOD at the model TS)
for the paper data, so errors and N_FOD are measured at the same structure.

One call works through a block of the sorted (model, rxn) list with a process pool; one JSON per
structure in ~/fod_model_ts/<model>/, existing ones are skipped (safe to restart).
--table collects all JSONs into results/fod_model_ts.csv and prints the missing ones.

    python pipeline/fod_model_ts.py --start 0 --count 250 --workers 6
    python pipeline/fod_model_ts.py --table
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_fod import compute_fod, T_EL  # noqa: E402

OUT_DIR = f'{H}/fod_model_ts'


def tasks():
    return sorted((r['model'], r['rxn']) for r in csv.DictReader(open(f'{RES}/d37_forces.csv')))


def run_one(task):
    m, rx = task
    out = f'{OUT_DIR}/{m}/{rx}.json'
    if os.path.exists(out):
        return task, 'exists'
    xyz = f'{H}/neb_fsm/fsm/{m}/{rx}/transition_state.xyz'
    t0 = time.time()
    try:
        res, err = compute_fod(xyz)
    except Exception as e:          # one failure must not stop the block
        res, err = None, f'{type(e).__name__}: {e}'
    d = dict(model=m, rxn=rx, geometry=xyz, level='PBE/def2-SVP RKS', T_el_K=T_EL,
             seconds=round(time.time() - t0, 1))
    d.update(res if res else dict(error=err))
    with open(out, 'w') as fh:
        json.dump(d, fh, indent=1)
    return task, ('ERROR ' + err) if err else 'N_FOD = %.4f  (%.0f s)' % (res['nfod'], d['seconds'])


def table():
    rows, missing = [], []
    for m, rx in tasks():
        p = f'{OUT_DIR}/{m}/{rx}.json'
        d = json.load(open(p)) if os.path.exists(p) else {}
        if 'nfod' in d:
            rows.append(dict(model=m, rxn=rx, nfod_model_ts=d['nfod']))
        else:
            missing.append((m, rx, d.get('error', 'not run')))
    with open(f'{RES}/fod_model_ts.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['model', 'rxn', 'nfod_model_ts'])
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} structures written to results/fod_model_ts.csv, {len(missing)} missing')
    for x in missing:
        print('  missing', *x)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int)
    ap.add_argument('--count', type=int)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--table', action='store_true')
    a = ap.parse_args()
    if a.table:
        table()
        sys.exit()
    T = tasks()
    for m in {m for m, _ in T}:
        os.makedirs(f'{OUT_DIR}/{m}', exist_ok=True)
    block = T[a.start:a.start + a.count]
    print(f'Block {a.start}..{a.start + len(block) - 1} of {len(T)}: {len(block)} structures, '
          f'{a.workers} processes, OMP {os.environ.get("OMP_NUM_THREADS")}', flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for (m, rx), msg in pool.imap_unordered(run_one, block):
            print(f'  {m} {rx}: {msg}', flush=True)
    print(f'END block {a.start}: {len(block)} structures in {(time.time() - t0) / 60:.1f} min', flush=True)
