"""
NEB with a genuinely broken-symmetry UKS reference, OMol25 procedure.

New script. pipeline/orca_neb.py and pipeline/orca_neb_omol25.py are untouched.

Why this exists: ~/orca_neb_omol25/ was labelled UKS but its ORCA input carried
only `charge 0 mult 1` and no UKS/UHF keyword, so ORCA ran the restricted
formalism. That run is a basis/grid test of the RKS reference, not a
broken-symmetry calculation.

Level of theory (identical to orca_neb_omol25.py):
    wB97M-V / def2-TZVPD, def2/J, RIJCOSX, TightSCF, DEFGRID3,
    Thresh 1e-12, TCut 1e-13
Additions:
    (1) UKS explicitly in the simple-input line.
    (2) OMol25 symmetry-broken guess: 20 deg HOMO/LUMO rotation in the BETA
        space -> `%scf Rotate {nHOMO, nLUMO, 20, 1, 1} end`.
        ORCA operator index 1 = beta (0 = alpha). Verified in this project:
        for rxn1320 this gives <S^2> = 0.7918, matching the stability-following
        solution to 1e-8 Ha.
    (3) BS continuity: the rotation is applied ONCE per image. After that image
        has a converged BS solution its .gbw is kept and every later call for
        that image reads it back via `! MORead` + `%moinp`, so the broken
        symmetry is propagated across NEB steps instead of being re-guessed.
    (4) Scratch is NOT deleted. Per-image ORCA .inp/.out/.gbw are retained.

nelec is DERIVED from the structure (sum of atomic numbers - charge). It selects
which orbitals get rotated, and a wrong value would silently rotate the wrong
pair with no error, so --nelec is only an optional cross-check that aborts on
disagreement.

<S^2> is logged per phase (relax_r / relax_p / neb) and per image to
s2_log.json. The collapse flag is evaluated on the TS image only: reactant and
product legitimately have <S^2> ~ 0 (intact bonds, no BS solution), so a pooled
minimum over all images is meaningless.

Usage:
    python orca_neb_bs.py --h5file ~/data/Transition1x.h5 --reaction rxn1320 \
        --output ~/orca_neb_bs/rxn1320 --nprocs 12
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import ase.db
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read, write
from ase.mep import NEB, NEBTools
from ase.mep.neb import NEBOptimizer
from ase.optimize.bfgs import BFGS

LEVEL = 'UKS wB97M-V/def2-TZVPD, BS guess (20 deg beta HOMO-LUMO rotation)'
BASE_SIMPLE = 'UKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3 EnGrad'
ROT_ANGLE = 20
BS_THRESH = 0.1          # <S^2> below this counts as collapsed

S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')

# {phase: {image_index: [s2 per call]}}
S2LOG = {}
PHASE = {'name': 'init'}


class BSORCA(ORCA):
    """ORCA calculator that breaks spin symmetry once, then keeps it.

    First call for this image : UKS + `%scf Rotate {H,L,20,1,1} end`
    Every later call          : UKS + `! MORead` + `%moinp "<saved>.gbw"`
    """

    def __init__(self, *args, img=0, homo=0, lumo=1, nprocs=12, maxcore=3000,
                 bs_store=None, **kwargs):
        self._img = img
        self._homo = homo
        self._lumo = lumo
        self._nprocs = nprocs
        self._maxcore = maxcore
        self._bs_gbw = bs_store
        super().__init__(*args, **kwargs)

    def _build(self):
        common = f'%pal nprocs {self._nprocs} end\n%maxcore {self._maxcore}\n'
        if self._bs_gbw and os.path.exists(self._bs_gbw):
            simple = BASE_SIMPLE + ' MORead'
            blocks = (common
                      + f'%moinp "{self._bs_gbw}"\n'
                      + '%scf\n  maxiter 200\n  Thresh 1e-12\n  TCut   1e-13\nend')
        else:
            simple = BASE_SIMPLE
            blocks = (common
                      + '%scf\n'
                      + '  maxiter 200\n'
                      + '  Thresh 1e-12\n'
                      + '  TCut   1e-13\n'
                      + f'  Rotate {{{self._homo}, {self._lumo}, {ROT_ANGLE}, 1, 1}} end\n'
                      + 'end')
        self.parameters['orcasimpleinput'] = simple
        self.parameters['orcablocks'] = blocks

    def calculate(self, atoms=None, properties=None, system_changes=None):
        # ASE 3.28 ORCA is a GenericFileIOCalculator: no write_input /
        # read_results to override, the template consumes self.parameters.
        self._build()
        super().calculate(atoms, properties, system_changes)
        self._post()

    def _post(self):
        d = str(self.directory)
        outs = sorted(glob.glob(os.path.join(d, '*.out')), key=os.path.getmtime)
        if outs:
            hits = S2_RE.findall(open(outs[-1], errors='replace').read())
            if hits:
                (S2LOG.setdefault(PHASE['name'], {})
                      .setdefault(str(self._img), []).append(float(hits[-1])))
        gbws = sorted(glob.glob(os.path.join(d, '*.gbw')), key=os.path.getmtime)
        if gbws and self._bs_gbw:
            try:
                shutil.copyfile(gbws[-1], self._bs_gbw)
            except Exception as e:
                print(f'  [img {self._img}] gbw copy failed: {e}', flush=True)


# ---------------------------------------------------------------- fix 3
def load_wB97x_images(h5file, reaction, split):
    """Starting band, selection identical to pipeline/orca_neb.py:
       [0] + last 8 + [9].  The indices are printed and checked for
       duplicates/gaps, because that expression can silently misbehave for
       short trajectories (and the same expression is in the older scripts)."""
    with h5py.File(h5file, 'r') as f:
        for formula in f[split]:
            if reaction in f[split][formula]:
                g = f[split][formula][reaction]
                pos = g['positions'][:]
                z = g['atomic_numbers'][:]
                n = pos.shape[0]
                idx = [0] + list(range(n - 8, n)) + [9]
                print(f'H5 trajectory: {n} configurations')
                print(f'selected indices: {idx}')
                dup = len(idx) != len(set(idx))
                bad = [i for i in idx if i < 0 or i >= n]
                if dup:
                    print(f'  WARNING: duplicated indices -> {sorted(idx)}')
                    print(f'  (n={n}; last-8 window overlaps index 9 when n <= 17)')
                if bad:
                    print(f'  WARNING: out-of-range indices {bad} for n={n}')
                if not dup and not bad:
                    print(f'  index check OK: 10 distinct images, all in [0,{n-1}]')
                print('  NOTE: identical selection to orca_neb.py / '
                      'orca_neb_omol25.py, so the old runs behaved the same way.')
                sel = [pos[0]] + list(pos[-8:]) + [pos[9]]
                return [Atoms(numbers=z, positions=p) for p in sel]
    raise ValueError(f'{reaction} not in split {split}')


class DBWriter:
    def __init__(self, db_path, images):
        self.images, self.db_path = images, db_path

    def write(self):
        with ase.db.connect(self.db_path) as db:
            for a in self.images:
                if a.calc and a.calc.results:
                    db.write(a, data=a.calc.results)


# ---------------------------------------------------------------- fix 2
def dump_s2(output, quiet=False):
    json.dump(S2LOG, open(os.path.join(output, 's2_log.json'), 'w'), indent=1)
    if quiet:
        return
    ph = PHASE['name']
    per = S2LOG.get(ph, {})
    if not per:
        return
    parts = []
    for k in sorted(per, key=int):
        v = per[k]
        parts.append(f'img{int(k):02d}={v[-1]:.3f}(n={len(v)})')
    print(f'  <S^2> [{ph}] ' + '  '.join(parts), flush=True)


def collapse_flag(ts_img):
    """Evaluate BS retention on the TS image only, during the NEB phase.
    Endpoints legitimately sit at <S^2> ~ 0 and must not enter this check."""
    traj = S2LOG.get('neb', {}).get(str(ts_img), [])
    if not traj:
        return 'NO_DATA', None, traj
    first = next((i for i, v in enumerate(traj) if v > BS_THRESH), None)
    if first is None:
        return 'NO_BS', None, traj
    after = traj[first:]
    lost = next((first + i for i, v in enumerate(after) if v < BS_THRESH), None)
    if lost is not None:
        return 'BS_LOST', lost, traj
    return 'BS_MAINTAINED', None, traj


def main(a):
    os.makedirs(a.output, exist_ok=True)
    scratch = os.path.join(a.output, 'scratch')
    os.makedirs(scratch, exist_ok=True)

    orca_cmd = a.orca_cmd if os.path.isabs(a.orca_cmd) else shutil.which(a.orca_cmd)
    if not orca_cmd:
        print('ERROR: orca not found'); sys.exit(1)

    images = load_wB97x_images(a.h5file, a.reaction, a.split)

    # ------------------------------------------------------------ fix 1
    nelec = int(sum(images[0].get_atomic_numbers())) - a.charge
    if nelec % 2 != 0:
        print(f'ERROR: odd electron count {nelec} for charge {a.charge}; '
              'a closed-shell singlet guess is not defined'); sys.exit(1)
    if a.nelec is not None and a.nelec != nelec:
        print(f'ERROR: --nelec {a.nelec} disagrees with the value derived from '
              f'the structure ({nelec} = sum(Z) - charge {a.charge}). '
              'Refusing to rotate a possibly wrong orbital pair.')
        sys.exit(1)
    nalpha = nelec // 2
    homo, lumo = nalpha - 1, nalpha          # 0-based beta HOMO / LUMO

    print(f'{LEVEL}')
    print(f'ORCA {orca_cmd}  nprocs={a.nprocs}')
    print(f'formula {images[0].get_chemical_formula()}  charge {a.charge}')
    print(f'nelec (derived) = {nelec}   nalpha = {nalpha}')
    print(f'beta HOMO = {homo}   beta LUMO = {lumo}   '
          f'-> Rotate {{{homo}, {lumo}, {ROT_ANGLE}, 1, 1}}')

    json.dump({'level': LEVEL, 'simpleinput': BASE_SIMPLE,
               'rotate': f'{{{homo}, {lumo}, {ROT_ANGLE}, 1, 1}}  (op 1 = beta)',
               'nelec_derived': nelec, 'charge': a.charge,
               'beta_homo': homo, 'beta_lumo': lumo,
               'nprocs': a.nprocs, 'bs_thresh': BS_THRESH,
               'orca_version': '5.0.4 (OMol25 used 6.0.0)',
               'scratch_retained': True},
              open(os.path.join(a.output, 'level.json'), 'w'), indent=1)

    for i, at in enumerate(images):
        d = os.path.join(scratch, f'img{i:02d}')
        os.makedirs(d, exist_ok=True)
        at.calc = BSORCA(profile=OrcaProfile(command=orca_cmd),
                         charge=a.charge, mult=1, directory=d,
                         img=i, homo=homo, lumo=lumo,
                         nprocs=a.nprocs, maxcore=a.maxcore,
                         bs_store=os.path.join(scratch, f'bs_img{i:02d}.gbw'))

    r_xyz = os.path.join(a.output, 'reactant.xyz')
    p_xyz = os.path.join(a.output, 'product.xyz')

    PHASE['name'] = 'relax_r'
    print('\nRelaxing reactant (BS-UKS) ...', flush=True)
    BFGS(images[0], logfile=os.path.join(a.output, 'relax_r.log')).run(fmax=0.05)
    write(r_xyz, images[0]); dump_s2(a.output)

    PHASE['name'] = 'relax_p'
    print('Relaxing product (BS-UKS) ...', flush=True)
    BFGS(images[-1], logfile=os.path.join(a.output, 'relax_p.log')).run(fmax=0.05)
    write(p_xyz, images[-1]); dump_s2(a.output)

    for ph, lbl in (('relax_r', 'reactant'), ('relax_p', 'product')):
        v = [x for lst in S2LOG.get(ph, {}).values() for x in lst]
        if v:
            bs = 'BS' if v[-1] > BS_THRESH else 'closed-shell'
            print(f'  endpoint {lbl}: final <S^2> = {v[-1]:.4f}  ({bs})', flush=True)

    PHASE['name'] = 'neb'
    print('\nNEB ...', flush=True)
    neb = NEB(images, climb=False, parallel=False)
    tools = NEBTools(images)
    opt = NEBOptimizer(neb, logfile=os.path.join(a.output, 'neb.log'))
    dbw = DBWriter(os.path.join(a.output, 'neb.db'), images)
    fmaxs = []
    opt.attach(dbw.write)
    opt.attach(lambda: fmaxs.append(tools.get_fmax()))
    opt.attach(lambda: dump_s2(a.output))
    opt.run(fmax=a.neb_fmax, steps=a.steps)

    print('\nCI-NEB ...', flush=True)
    neb.climb = True
    converged = opt.run(fmax=a.cineb_fmax, steps=a.steps)
    if converged:
        open(os.path.join(a.output, 'converged'), 'w').close()
    print('converged' if converged else 'NOT converged', flush=True)

    json.dump(fmaxs, open(os.path.join(a.output, 'fmaxs.json'), 'w'))
    dump_s2(a.output)

    ts = max(images, key=lambda x: x.get_potential_energy())
    ts_i = images.index(ts)
    write(os.path.join(a.output, 'transition_state.xyz'), ts)
    write(r_xyz, images[0]); write(p_xyz, images[-1])

    flag, lost_at, traj = collapse_flag(ts_i)
    print(f'\nTS image {ts_i}: {flag}'
          + (f' (fell below {BS_THRESH} at NEB call {lost_at})' if lost_at is not None else ''))
    if traj:
        print(f'  TS <S^2> first={traj[0]:.4f} last={traj[-1]:.4f} '
              f'min={min(traj):.4f} max={max(traj):.4f} n={len(traj)}')

    json.dump({'ts_image': ts_i, 'bs_flag': flag, 'lost_at_call': lost_at,
               'ts_s2_trajectory': traj,
               'ts_s2_final': traj[-1] if traj else None},
              open(os.path.join(a.output, 'bs_summary.json'), 'w'), indent=1)

    tsd = os.path.join(scratch, f'img{ts_i:02d}')
    for pat in ('*.out', '*.inp', '*.gbw'):
        for f in glob.glob(os.path.join(tsd, pat)):
            shutil.copy(f, os.path.join(a.output, f'TS_{os.path.basename(f)}'))

    fit = NEBTools(images).get_fit()
    fig, ax = plt.subplots()
    ax.plot(fit.fit_path, fit.fit_energies, 'b-o', markersize=4,
            label=f'{LEVEL} (barrier {max(fit.fit_energies):.3f} eV)')
    ax.set_xlabel('Reaction Coordinate [Å]'); ax.set_ylabel('Energy [eV]')
    ax.legend(fontsize=7); fig.savefig(os.path.join(a.output, 'mep.png'))
    plt.close(fig)
    print(f'\nDone. {a.output}/  (TS image {ts_i}, scratch retained)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5file', required=True)
    p.add_argument('--reaction', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--output', required=True)
    p.add_argument('--charge', type=int, default=0)
    p.add_argument('--nelec', type=int, default=None,
                   help='optional cross-check; aborts if it disagrees with '
                        'sum(Z) - charge derived from the structure')
    p.add_argument('--orca-cmd', default='orca')
    p.add_argument('--nprocs', type=int, default=12)
    p.add_argument('--maxcore', type=int, default=3000)
    p.add_argument('--neb-fmax', type=float, default=0.15)
    p.add_argument('--cineb-fmax', type=float, default=0.05)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--dry-run', action='store_true',
                   help='write the ORCA input for image 0 and exit')
    a = p.parse_args()

    if a.dry_run:
        os.makedirs(a.output, exist_ok=True)
        d = os.path.join(a.output, 'dryrun')
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        imgs = load_wB97x_images(a.h5file, a.reaction, a.split)
        nelec = int(sum(imgs[0].get_atomic_numbers())) - a.charge
        if a.nelec is not None and a.nelec != nelec:
            print(f'ERROR: --nelec {a.nelec} != derived {nelec}'); sys.exit(1)
        nalpha = nelec // 2
        print(f'formula {imgs[0].get_chemical_formula()}  charge {a.charge}')
        print(f'nelec (derived) = {nelec}  -> beta HOMO {nalpha-1}, LUMO {nalpha}')
        c = BSORCA(profile=OrcaProfile(command=shutil.which('orca') or 'orca'),
                   charge=a.charge, mult=1, directory=d, img=0,
                   homo=nalpha - 1, lumo=nalpha,
                   nprocs=a.nprocs, maxcore=a.maxcore,
                   bs_store=os.path.join(d, 'bs.gbw'))
        c._build()
        c.template.write_input(profile=c.profile, directory=c.directory,
                               atoms=imgs[0], parameters=c.parameters,
                               properties=['energy'])
        inp = sorted(glob.glob(os.path.join(d, '*.inp')))
        txt = open(inp[0]).read() if inp else ''
        print('\n===== WRITTEN ORCA INPUT =====')
        print(txt if txt else 'NO INPUT WRITTEN')
        print('==============================')
        print('UKS present :', 'UKS' in txt)
        print('Rotate      :', 'Rotate' in txt)
        print('beta op (1) :',
              bool(re.search(r'Rotate\s*\{[^}]*,\s*1\s*,\s*1\s*\}', txt)))
        m = re.search(r'Rotate\s*\{\s*(\d+)\s*,\s*(\d+)', txt)
        print('rotated pair:', f'{m.group(1)} -> {m.group(2)}' if m else 'NONE',
              f'(expected {nalpha-1} -> {nalpha})')
        sys.exit(0)
    main(a)
