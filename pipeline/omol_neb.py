"""CI-NEB with an OMol25 model for a list of Transition1x reactions.

One script for UMA-S, UMA-M, eSEN and MACE-OMol, replacing pipeline/uma_neb.py
and pipeline/esen_neb.py for the paper. The NEB itself is unchanged from the
thesis (Section "Transition-State Searches"):

    start     final wB97X/6-31G(d) band from Transition1x.h5: R, last 8 interior
              images, P (positions[0], positions[-8:], positions[9])
    endpoints BFGS to 0.05 eV/A
    band      improved tangent, ASE default spring, NEBOptimizer (ode12r),
              NEB to 0.15, then climbing image to 0.05, at most 500 steps per phase

What differs from the thesis scripts, none of it touching the band dynamics:
    - the model is loaded once and the script works through a list of reactions
    - split and charge/spin are explicit (charge 0, spin 1 = multiplicity); the
      thesis relied on fairchem's default, which gives the same (checked for
      UMA-S and eSEN, 2026-10-29/31)
    - the band is not written to neb.db at every step; final_band.extxyz holds
      the last band with energies and forces. neb.log is kept, in the same
      format, because pipeline/neb_runs.py reads convergence from it
    - BFGS has a step limit of 2000 (the ASE default is effectively unlimited);
      in the thesis no relaxation came near it

Convergence: the return value of NEBOptimizer.run is NOT convergence, it is True
whenever the solver did not raise, also when the step budget ran out (thesis:
133 marked, 112 reached the criterion). result.json therefore records the last
value of neb.log as f_band_final and criterion_met = f_band_final <= 0.05 + 5e-5,
the rule of pipeline/neb_runs.py.

Output  <outroot>/<model>/<rxn>/
    reactant.xyz, product.xyz, transition_state.xyz (highest image), final_band.extxyz,
    neb.log, relax_r.log, relax_p.log, result.json
result.json is written last, so a reaction with one counts as finished and is
skipped on a restart; a job killed mid-reaction is simply repeated.

    python pipeline/omol_neb.py --model uma-s --rxns list.txt --start 0 --count 50 \
        --worker 0 --nworkers 4 --device cpu --outroot ~/neb_omol
"""
import argparse
import copy
import json
import os
import re
import time
import traceback

import h5py
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.mep import NEB
from ase.mep.neb import NEBOptimizer
from ase.optimize.bfgs import BFGS

H = os.path.expanduser('~')
CHECKPOINTS = {'uma-s': f'{H}/checkpoints/uma-s-1p2.pt',
               'uma-m': f'{H}/checkpoints/uma-m-1p1.pt',
               'esen': f'{H}/checkpoints/esen_sm_conserving_all.pt',
               'mace-omol': f'{H}/.cache/mace/MACE-omol-0-extra-large-1024.model'}
H5 = f'{H}/data/Transition1x.h5'
CINEB_TOL = 5e-5        # neb.log rounds to four decimals, as in pipeline/neb_runs.py
LOGLINE = re.compile(r'^NEBOptimizer\[\w+\]:\s+(\d+)\s+\S+\s+([-\d.eE+]+)\s*$')


def make_calc_factory(model, device, precision='default'):
    """Return a function that gives each NEB image its own calculator,
    sharing one loaded model (ASE NEB refuses a shared calculator).
    precision 'default' is single for UMA and eSEN (fairchem's default, also
    the thesis) and double for MACE-OMol (recommended for optimisation)."""
    if model == 'mace-omol':
        from mace.calculators import mace_omol
        base = mace_omol(model=CHECKPOINTS[model], device=device,
                         default_dtype='float32' if precision == 'float32' else 'float64')

        def factory():
            c = copy.copy(base)
            c.reset()
            return c
        return factory
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    kw = {}
    if precision == 'float64':
        import torch
        from fairchem.core.units.mlip_unit.api.inference import inference_settings_default
        st = inference_settings_default()
        st.base_precision_dtype = torch.float64
        kw['inference_settings'] = st
    unit = pretrained_mlip.load_predict_unit(CHECKPOINTS[model], device=device, **kw)
    return lambda: FAIRChemCalculator(unit, task_name='omol')


def software_versions():
    from importlib import metadata
    v = {}
    for p in ('fairchem-core', 'mace-torch', 'e3nn', 'torch', 'ase'):
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            v[p] = None
    import e3nn
    v['e3nn'] = e3nn.__version__        # the version actually on the path
    return v


def load_band(f, split, rxn):
    """The ten images the thesis starts from, identical to load_wB97x_images."""
    for formula in f[split]:
        if rxn in f[split][formula]:
            g = f[split][formula][rxn]
            P, Z = g['positions'], g['atomic_numbers'][:]
            frames = [P[0]] + list(P[-8:]) + [P[9]]
            return [Atoms(numbers=Z, positions=x, info={'charge': 0, 'spin': 1}) for x in frames]
    raise ValueError(f'{rxn} not in split {split}')


def read_log(p):
    """(last band force, number of logged steps), as pipeline/neb_runs.py."""
    f, n = None, 0
    if os.path.exists(p):
        for line in open(p):
            m = LOGLINE.match(line.strip())
            if m:
                f, n = float(m.group(2)), n + 1
    return f, n


def run_one(rxn, split, f5, factory, out, a):
    os.makedirs(out, exist_ok=True)
    for name in ('neb.log', 'relax_r.log', 'relax_p.log'):      # repeat after a kill
        if os.path.exists(f'{out}/{name}'):
            os.remove(f'{out}/{name}')
    t0 = time.time()
    res = dict(rxn=rxn, split=split, model=a.model, device=a.device, precision=a.precision,
               continue_from=a.continue_from, cineb_fmax=a.cineb_fmax, start_band=a.start_band, status='error',
               versions=VERSIONS)
    images = None
    try:
        if a.continue_from:
            # keep going from a finished band: endpoints stay fixed, climbing image on
            old = read(f'{a.continue_from}/{a.model}/{rxn}/final_band.extxyz', ':')
            images = [Atoms(numbers=x.numbers, positions=x.positions, info={'charge': 0, 'spin': 1})
                      for x in old]
            for im in images:
                im.calc = factory()
            write(f'{out}/reactant.xyz', images[0])
            write(f'{out}/product.xyz', images[-1])
            neb = NEB(images, climb=True, parallel=False, method='improvedtangent')
            opt = NEBOptimizer(neb, logfile=f'{out}/neb.log')
            res['solver_ok_cineb'] = bool(opt.run(fmax=a.cineb_fmax, steps=a.steps))
            res['status'] = 'done'
        else:
            images = load_band(f5, split, rxn)
            for im in images:
                im.calc = factory()

            r_opt = BFGS(images[0], logfile=f'{out}/relax_r.log')
            p_opt = BFGS(images[-1], logfile=f'{out}/relax_p.log')
            res['endpoint_r_ok'] = bool(r_opt.run(fmax=0.05, steps=2000))
            res['endpoint_p_ok'] = bool(p_opt.run(fmax=0.05, steps=2000))
            res['seconds_endpoints'] = round(time.time() - t0, 1)
            write(f'{out}/reactant.xyz', images[0])
            write(f'{out}/product.xyz', images[-1])

            neb = NEB(images, climb=False, parallel=False, method='improvedtangent')
            if a.start_band == 'idpp':
                # no Transition1x path: only the relaxed endpoints enter
                neb.interpolate(method='idpp')
            opt = NEBOptimizer(neb, logfile=f'{out}/neb.log')
            res['solver_ok_neb'] = bool(opt.run(fmax=a.neb_fmax, steps=a.steps))
            neb.climb = True
            res['solver_ok_cineb'] = bool(opt.run(fmax=a.cineb_fmax, steps=a.steps))
            res['status'] = 'done'
    except Exception as e:      # one reaction must not stop the block
        res['error'] = f'{type(e).__name__}: {e}'
        res['traceback'] = traceback.format_exc()[-800:]

    if images is not None and res['status'] == 'done':
        ts = max(range(len(images)), key=lambda i: images[i].get_potential_energy())
        write(f'{out}/transition_state.xyz', images[ts])
        write(f'{out}/final_band.extxyz', images, format='extxyz')
        E = np.array([im.get_potential_energy() for im in images])
        res.update(ts_image=int(ts), e_r_ev=float(E[0]), e_ts_ev=float(E[ts]), e_p_ev=float(E[-1]),
                   barrier_ev=float(E[ts] - E[0]))
    fband, nsteps = read_log(f'{out}/neb.log')
    res.update(f_band_final=fband, n_steps=nsteps,
               criterion_met=(None if fband is None else bool(fband <= a.cineb_fmax + CINEB_TOL)),
               seconds=round(time.time() - t0, 1))
    tmp = f'{out}/result.json.tmp'
    with open(tmp, 'w') as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, f'{out}/result.json')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(CHECKPOINTS))
    ap.add_argument('--rxns', required=True, help='file with one reaction per line')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=10**9)
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--nworkers', type=int, default=1)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--outroot', default=f'{H}/neb_omol')
    ap.add_argument('--neb-fmax', type=float, default=0.15)
    ap.add_argument('--cineb-fmax', type=float, default=0.05)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--precision', default='default', choices=['default', 'float32', 'float64'])
    ap.add_argument('--start-band', default='t1x', choices=['t1x', 'idpp'],
                    help='t1x: final Transition1x band (thesis); idpp: interpolation between the relaxed endpoints')
    ap.add_argument('--continue-from', default=None,
                    help='outroot of a finished run: continue its final bands with the climbing image')
    a = ap.parse_args()
    VERSIONS = software_versions()

    rxns = [l.strip() for l in open(a.rxns) if l.strip()]
    block = rxns[a.start:a.start + a.count][a.worker::a.nworkers]
    split = {k: v['split'] for k, v in json.load(open(f'{H}/t1x_ts_all.json')).items()}
    todo = [r for r in block if not os.path.exists(f'{a.outroot}/{a.model}/{r}/result.json')]
    print(f'{a.model} on {a.device}: {len(todo)} of {len(block)} reactions to do '
          f'(worker {a.worker}/{a.nworkers}, OMP {os.environ.get("OMP_NUM_THREADS")})', flush=True)
    if not todo:
        raise SystemExit(0)

    t0 = time.time()
    factory = make_calc_factory(a.model, a.device, a.precision)
    print(f'model loaded in {time.time() - t0:.1f} s', flush=True)
    with h5py.File(H5, 'r') as f5:
        for rx in todo:
            r = run_one(rx, split[rx], f5, factory, f'{a.outroot}/{a.model}/{rx}', a)
            print('  %s %-5s f_band %s  met %s  steps %s  %.0f s  %s'
                  % (rx, r['status'], r['f_band_final'], r['criterion_met'], r['n_steps'],
                     r['seconds'], r.get('error', '')), flush=True)
    print(f'END {a.model}: {len(todo)} reactions in {(time.time() - t0) / 60:.1f} min', flush=True)
