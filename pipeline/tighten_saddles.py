"""D43: are the freezing-string saddles converged on the model, or did the search stop too early?

For every converged freezing-string run (Sella criterion 0.05 eV/A met) of the 510 reactions,
Sella continues on the same model from transition_state.xyz with the settings of
fsm_search.py (order 1, internal coordinates) down to 0.01 eV/A, at most 1000 steps. Recorded:
convergence, steps, RMSD between the two saddles after Kabsch alignment, model energy change of
the saddle (the reactant is unchanged, so this is the change of the model barrier), largest
per-atom model force before and after. A saddle that moves more than 0.1 A is flagged as
moved elsewhere, not as a refined version of the same saddle. Reading fixed in D43.

Output ~/neb_fsm/tight/<model>/<rxn>.json and .xyz; table results/tighten_saddles.csv

    python pipeline/tighten_saddles.py --model esen     (workers of a job share the list)
    python pipeline/tighten_saddles.py --table
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from ase import Atoms
from ase.io import read, write

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
OUT = f'{H}/neb_fsm/tight'
M = ('esen', 'uma-s', 'uma-m', 'mace-omol')
S = {r['rxn']: r for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{RES}/{n}.csv'))}


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt((((a @ (u @ np.diag([1, 1, d]) @ vt)) - b) ** 2).sum(1).mean()))


def todo(m):
    for rx in S:
        d = f'{H}/neb_fsm/fsm/{m}/{rx}'
        if os.path.exists(f'{d}/result.json') and json.load(open(f'{d}/result.json'))['criterion_met']:
            yield rx, d


def run(m):
    from omol_neb import make_calc_factory
    from sella import Sella
    factory = make_calc_factory(m, 'cpu')
    job = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    os.makedirs(f'{OUT}/{m}', exist_ok=True)
    for rx, d in todo(m):
        p = f'{OUT}/{m}/{rx}'
        if os.path.exists(f'{p}.json'):
            continue
        try:
            os.close(os.open(f'{p}.claim.{job}', os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        x = read(f'{d}/transition_state.xyz')
        a = Atoms(numbers=x.numbers, positions=x.positions, info={'charge': 0, 'spin': 1})
        a.calc = factory()
        e0, f0 = float(a.get_potential_energy()), float(np.sqrt((a.get_forces() ** 2).sum(1)).max())
        out = dict(rxn=rx, model=m, group=S[rx]['group'], nfod=float(S[rx]['nfod']), e0_ev=e0, fmax0=f0)
        try:
            s = Sella(a, internal=True, order=1, logfile=None)
            out['converged'] = bool(s.run(fmax=0.01, steps=1000))
            out['steps'] = int(s.get_number_of_steps())
            out['e1_ev'] = float(a.get_potential_energy())
            out['fmax1'] = float(np.sqrt((a.get_forces() ** 2).sum(1)).max())
            out['rmsd_A'] = kabsch_rmsd(x.positions, a.positions)
            out['dbarrier_mev'] = 1000 * (out['e1_ev'] - e0)
            out['status'] = 'done'
            write(f'{p}.xyz', a, format='xyz')
        except Exception as exc:             # a Sella failure is a result, recorded per structure
            out.update(status='sella_error', error=str(exc)[:200])
        with open(f'{p}.tmp', 'w') as fh:
            json.dump(out, fh, indent=1)
        os.replace(f'{p}.tmp', f'{p}.json')
        print('  %s %-9s %-8s %s steps %s rmsd %s dE %s meV' % (rx, m, out['group'], out['status'], out.get('steps'),
              '%.3f' % out['rmsd_A'] if 'rmsd_A' in out else '-', '%.1f' % out['dbarrier_mev'] if 'dbarrier_mev' in out else '-'), flush=True)
    print('END', m, flush=True)


def table():
    rows = [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}/*/*.json'))]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f'{RES}/tighten_saddles.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval='')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('%.5f' % v if isinstance(v, float) else v) for k, v in r.items()})
    print('%-10s %-9s %4s %5s %6s %6s %8s %8s %6s' % ('model', 'group', 'n', 'conv%', 'moved', 'rmsd', '|dE|med', '>43meV', 'err'))
    for m in M:
        for g in ('baseline', 'closed', 'bs', 'high'):
            R = [r for r in rows if r['model'] == m and r['group'] == g]
            if not R:
                continue
            ok = [r for r in R if r['status'] == 'done']
            same = [r for r in ok if r['rmsd_A'] <= 0.1]
            print('%-10s %-9s %4d %5.0f %6d %6.3f %8.1f %7.0f%% %6d' % (
                m, g, len(R), 100 * np.mean([r['converged'] for r in ok]) if ok else 0, len(ok) - len(same),
                np.median([r['rmsd_A'] for r in same]) if same else np.nan,
                np.median([abs(r['dbarrier_mev']) for r in same]) if same else np.nan,
                100 * np.mean([abs(r['dbarrier_mev']) > 43 for r in ok]) if ok else 0, len(R) - len(ok)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--table', action='store_true')
    a = ap.parse_args()
    if a.model:
        run(a.model)
    if a.table:
        table()
