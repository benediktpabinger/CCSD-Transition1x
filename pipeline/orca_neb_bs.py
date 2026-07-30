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

<S^2> is parsed from every ORCA output and logged per image per step to
s2_log.json.

Usage:
    python orca_neb_bs.py --h5file ~/data/Transition1x.h5 --reaction rxn1320 \
        --output ~/orca_neb_bs/rxn1320 --nelec 46 --nprocs 12
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

S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')

# collected as {image_index: [s2, s2, ...]} across all calls
S2LOG = {}


def scf_common(maxcore, nprocs):
    return (f'%pal nprocs {nprocs} end\n'
            f'%maxcore {maxcore}\n')


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
        self._bs_gbw = bs_store          # absolute path of the retained gbw
        super().__init__(*args, **kwargs)

    def _build(self):
        common = scf_common(self._maxcore, self._nprocs)
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
        # ASE 3.28 ORCA is a GenericFileIOCalculator: there is no write_input /
        # read_results to override, the template consumes self.parameters. So
        # rebuild the input right before the run and post-process right after.
        self._build()
        super().calculate(atoms, properties, system_changes)
        self._post()

    def _post(self):
        d = str(self.directory)
        outs = sorted(glob.glob(os.path.join(d, '*.out')),
                      key=os.path.getmtime)
        if outs:
            txt = open(outs[-1], errors='replace').read()
            hits = S2_RE.findall(txt)
            if hits:
                S2LOG.setdefault(str(self._img), []).append(float(hits[-1]))
        # retain the converged BS orbitals for the next call on this image
        gbws = sorted(glob.glob(os.path.join(d, '*.gbw')), key=os.path.getmtime)
        if gbws and self._bs_gbw:
            try:
                shutil.copyfile(gbws[-1], self._bs_gbw)
            except Exception as e:
                print(f'  [img {self._img}] gbw copy failed: {e}', flush=True)


def load_wB97x_images(h5file, reaction, split):
    with h5py.File(h5file, 'r') as f:
        for formula in f[split]:
            if reaction in f[split][formula]:
                g = f[split][formula][reaction]
                pos = g['positions'][:]
                z = g['atomic_numbers'][:]
                sel = [pos[0]] + list(pos[-8:]) + [pos[9]]
                print(f'Loaded 10 wB97x images ({pos.shape[0]} total)')
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


def dump_s2(output):
    json.dump(S2LOG, open(os.path.join(output, 's2_log.json'), 'w'), indent=1)
    allv = [v for lst in S2LOG.values() for v in lst]
    if allv:
        print(f'  <S^2>: n={len(allv)} min={min(allv):.4f} max={max(allv):.4f}',
              flush=True)


def main(a):
    os.makedirs(a.output, exist_ok=True)
    scratch = os.path.join(a.output, 'scratch')     # retained, not /tmp
    os.makedirs(scratch, exist_ok=True)

    orca_cmd = a.orca_cmd if os.path.isabs(a.orca_cmd) else shutil.which(a.orca_cmd)
    if not orca_cmd:
        print('ERROR: orca not found'); sys.exit(1)

    nalpha = a.nelec // 2
    homo, lumo = nalpha - 1, nalpha        # 0-based beta HOMO / LUMO
    print(f'{LEVEL}\nORCA {orca_cmd} nprocs={a.nprocs}')
    print(f'nelec={a.nelec} -> beta HOMO={homo} LUMO={lumo}')

    json.dump({'level': LEVEL, 'simpleinput': BASE_SIMPLE,
               'rotate': f'{{{homo}, {lumo}, {ROT_ANGLE}, 1, 1}}  (1 = beta)',
               'nelec': a.nelec, 'nprocs': a.nprocs,
               'orca_version': '5.0.4 (OMol25 used 6.0.0)',
               'scratch_retained': True},
              open(os.path.join(a.output, 'level.json'), 'w'), indent=1)

    images = load_wB97x_images(a.h5file, a.reaction, a.split)
    for i, at in enumerate(images):
        d = os.path.join(scratch, f'img{i:02d}')
        os.makedirs(d, exist_ok=True)
        at.calc = BSORCA(profile=OrcaProfile(command=orca_cmd),
                         charge=0, mult=1, directory=d,
                         img=i, homo=homo, lumo=lumo,
                         nprocs=a.nprocs, maxcore=a.maxcore,
                         bs_store=os.path.join(scratch, f'bs_img{i:02d}.gbw'))

    r_xyz = os.path.join(a.output, 'reactant.xyz')
    p_xyz = os.path.join(a.output, 'product.xyz')
    print('Relaxing reactant (BS-UKS) ...', flush=True)
    BFGS(images[0], logfile=os.path.join(a.output, 'relax_r.log')).run(fmax=0.05)
    write(r_xyz, images[0]); dump_s2(a.output)
    print('Relaxing product (BS-UKS) ...', flush=True)
    BFGS(images[-1], logfile=os.path.join(a.output, 'relax_p.log')).run(fmax=0.05)
    write(p_xyz, images[-1]); dump_s2(a.output)

    print('NEB ...', flush=True)
    neb = NEB(images, climb=False, parallel=False)
    tools = NEBTools(images)
    opt = NEBOptimizer(neb, logfile=os.path.join(a.output, 'neb.log'))
    dbw = DBWriter(os.path.join(a.output, 'neb.db'), images)
    fmaxs = []
    opt.attach(dbw.write)
    opt.attach(lambda: fmaxs.append(tools.get_fmax()))
    opt.attach(lambda: dump_s2(a.output))
    opt.run(fmax=a.neb_fmax, steps=a.steps)

    print('CI-NEB ...', flush=True)
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

    # retain the TS single-point ORCA output
    tsd = os.path.join(scratch, f'img{ts_i:02d}')
    for pat in ('*.out', '*.inp', '*.gbw'):
        for f in glob.glob(os.path.join(tsd, pat)):
            shutil.copy(f, os.path.join(a.output, f'TS_{os.path.basename(f)}'))
    json.dump({'ts_image': ts_i}, open(os.path.join(a.output, 'ts_image.json'), 'w'))

    tools2 = NEBTools(images)
    fit = tools2.get_fit()
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
    p.add_argument('--nelec', type=int, required=True)
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
        os.makedirs(d, exist_ok=True)
        nalpha = a.nelec // 2
        c = BSORCA(profile=OrcaProfile(command=shutil.which('orca') or 'orca'),
                   charge=0, mult=1, directory=d, img=0,
                   homo=nalpha - 1, lumo=nalpha,
                   nprocs=a.nprocs, maxcore=a.maxcore,
                   bs_store=os.path.join(d, 'bs.gbw'))
        imgs = load_wB97x_images(a.h5file, a.reaction, a.split)
        c._build()
        c.template.write_input(profile=c.profile, directory=c.directory,
                               atoms=imgs[0], parameters=c.parameters,
                               properties=['energy'])
        inp = sorted(glob.glob(os.path.join(d, '*.inp')))
        print('\n===== WRITTEN ORCA INPUT =====')
        print(open(inp[0]).read() if inp else 'NO INPUT WRITTEN')
        print('==============================')
        txt = open(inp[0]).read() if inp else ''
        print('UKS present :', 'UKS' in txt)
        print('Rotate      :', 'Rotate' in txt)
        print('beta op (1) :', bool(re.search(r'Rotate\s*\{[^}]*,\s*1\s*,\s*1\s*\}', txt)))
        sys.exit(0)
    main(a)
