"""Freezing-string search with saddle refinement on the model, on the D24 reactions.

The ML-FSM workflow of Marks et al. (arXiv 2604.00405): 18 nodes, interpolation in
redundant internal coordinates, Cartesian L-BFGS-B with at most 3 line-search and
2 optimisation steps per node, step limit 0.05 A. The highest node goes to Sella
(first-order saddle, internal coordinates, 0.05 eV/A, at most 1000 steps).

Starts from the relaxed endpoints of the eSEN NEB of the same reaction, so nothing
comes from the Transition1x band. Precision as in the NEBs (D24).

Each run folder has the layout of omol_neb.py, with transition_state.xyz the refined
structure, so pipeline/pilot_check.py can check it with start = fsm.

Output ~/neb_fsm/fsm/<model>/<rxn>/

    python pipeline/fsm_search.py --model esen --device cpu
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from ase import Atoms
from ase.io import read, write
from mlfsm.cos import FreezingString
from mlfsm.geom import project_trans_rot
from mlfsm.opt import CartesianOptimizer
from sella import Sella

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omol_neb import make_calc_factory  # noqa: E402

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def load(p):
    a = read(p)
    return Atoms(numbers=a.numbers, positions=a.positions, info={'charge': 0, 'spin': 1})


def energy(a, factory):
    b = a.copy()
    b.calc = factory()
    return float(b.get_potential_energy()), float(np.sqrt((b.get_forces() ** 2).sum(1)).max())


def run_one(rx, src, out, factory):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    r, p = load(f'{src}/reactant.xyz'), load(f'{src}/product.xyz')
    xr, xp = project_trans_rot(r.get_positions(), p.get_positions())
    r.set_positions(xr.reshape(-1, 3))
    p.set_positions(xp.reshape(-1, 3))
    write(f'{out}/reactant.xyz', r, format='xyz')
    write(f'{out}/product.xyz', p, format='xyz')
    res = dict(rxn=rx, start='fsm', status='fsm_error', criterion_met=False, n_grad_fsm=0, n_steps_sella=0)
    try:
        string = FreezingString(r, p, 18, 'ric', 50, 0.0)
        opt = CartesianOptimizer(factory(), 'L-BFGS-B', 2, 3, 0.05)
        while string.growing:
            string.grow()
            string.optimize(opt)
    except Exception as e:
        res['error'] = repr(e)[:300]
        res['seconds'] = time.time() - t0
        return res
    res['n_grad_fsm'] = int(string.ngrad)
    nodes = string.r_string[1:] + string.p_string[1:][::-1]
    E = [energy(n, factory)[0] for n in nodes]
    g = nodes[int(np.argmax(E))]
    guess = Atoms(numbers=r.numbers, positions=g.get_positions(), info={'charge': 0, 'spin': 1})
    write(f'{out}/fsm_guess.xyz', guess, format='xyz')
    e_r, e_p = energy(r, factory)[0], energy(p, factory)[0]
    res.update(e_r_ev=e_r, e_p_ev=e_p, barrier_guess_ev=max(E) - e_r, n_nodes=len(nodes))
    a = guess.copy()
    a.calc = factory()
    try:
        s = Sella(a, internal=True, order=1, logfile=f'{out}/sella.log')
        res['criterion_met'] = bool(s.run(fmax=0.05, steps=1000))
        res['n_steps_sella'] = int(s.get_number_of_steps())
        res['status'] = 'done'
    except Exception as e:
        res['status'] = 'sella_error'
        res['error'] = repr(e)[:300]
    ts = Atoms(numbers=r.numbers, positions=a.get_positions(), info={'charge': 0, 'spin': 1})
    write(f'{out}/transition_state.xyz', ts, format='xyz')
    e_ts, f = energy(ts, factory)
    res.update(e_ts_ev=e_ts, f_band_final=f, barrier_ev=e_ts - e_r, seconds=time.time() - t0)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='esen')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--outroot', default=f'{H}/neb_fsm')
    ap.add_argument('--rxns')
    a = ap.parse_args()
    S = {r['rxn']: r for r in csv.DictReader(open(f'{REPO}/results/d24_set.csv'))}
    rxns = [x.strip() for x in open(a.rxns) if x.strip()] if a.rxns else list(S)
    base = f'{a.outroot}/fsm/{a.model}'
    os.makedirs(base, exist_ok=True)
    todo = [r for r in rxns if not os.path.exists(f'{base}/{r}/result.json')]
    JOB = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    print(f'{a.model}: {len(todo)} of {len(rxns)} reactions to do, OMP {os.environ.get("OMP_NUM_THREADS")}', flush=True)
    factory = make_calc_factory(a.model, a.device)
    for rx in todo:
        out = f'{base}/{rx}'
        os.makedirs(out, exist_ok=True)
        try:                                # workers of a job share the list, as in omol_neb.py
            os.close(os.open(f'{out}/claim.{JOB}', os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        src = f"{H}/{S[rx]['root']}/t1x/esen/{rx}"
        res = run_one(rx, src, out, factory)
        with open(f'{out}/result.tmp', 'w') as fh:
            json.dump(res, fh, indent=1)
        os.replace(f'{out}/result.tmp', f'{out}/result.json')
        print('  %s %-7s %-11s met %-5s grads %d sella %d barrier %s guess %s %.0f s %s'
              % (rx, S[rx]['group'], res['status'], res['criterion_met'], res['n_grad_fsm'], res['n_steps_sella'],
                 '%.3f' % res['barrier_ev'] if 'barrier_ev' in res else '-',
                 '%.3f' % res['barrier_guess_ev'] if 'barrier_guess_ev' in res else '-',
                 res.get('seconds', 0), res.get('error', '')[:80]), flush=True)
    print('END', a.model, flush=True)
