"""
Val delta SP (flip): wB97M-V/def2-TZVP single points + gradients on 50 uniformly
subsampled T1x geometries for a failed-NEB val reaction. The wB97X-D3/6-31G(d)
energy and forces are read directly from T1x.

Delta energy = E_wB97M - E_wB97X
Delta forces = F_wB97M - F_wB97X

Used for the 51 val reactions where NEB did not converge, so no neb.db exists.
T1x geometries span the full NEB optimization history — good PES coverage.

Output: ~/val_delta_sp_flip/{rxn}/results.json

Usage:
    python val_delta_sp_flip.py <rxn> [--n-images 50] [--n-threads 8]
"""
import argparse
import json
import os
import re
import subprocess

import h5py
import numpy as np
from ase import Atoms
from ase.io import write

HOME     = '/home/energy/s242862'
H5FILE   = f'{HOME}/data/Transition1x.h5'
OUT_BASE = f'{HOME}/val_delta_sp_flip'
ORCA_CMD = '/home/modules/software/ORCA/5.0.4-gompi-2023a/bin/orca'
HA_TO_EV          = 27.2114
HA_BOHR_TO_EV_ANG = 27.2114 / 0.529177  # gradient Ha/Bohr → force eV/Å

ENERGY_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')

ORCA_TEMPLATE = """\
! wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad

%pal nprocs 1 end
%maxcore 4000
%scf maxiter 200 end

* xyzfile 0 1 geom.xyz
"""


def load_t1x_reaction(h5file, rxn, split='val'):
    with h5py.File(h5file, 'r') as f:
        split_grp = f[split]
        for formula in split_grp:
            if rxn in split_grp[formula]:
                grp = split_grp[formula][rxn]
                positions      = grp['positions'][:]
                atomic_numbers = grp['atomic_numbers'][:]
                energies_wb97x = grp['wB97x_6-31G(d).energy'][:]
                forces_wb97x   = grp['wB97x_6-31G(d).forces'][:]
                return positions, atomic_numbers, energies_wb97x, forces_wb97x
    raise KeyError(f'{rxn} not found in {split} split')


def parse_forces(out_path):
    """Parse forces (eV/Å) from ORCA EnGrad output. Returns list of [fx,fy,fz] or None."""
    content = open(out_path).read()
    idx = content.find('CARTESIAN GRADIENT')
    if idx == -1:
        return None
    lines = content[idx:].split('\n')
    forces = []
    for line in lines[3:]:  # skip header, dashes, blank line
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit() and parts[2] == ':':
            gx, gy, gz = float(parts[3]), float(parts[4]), float(parts[5])
            forces.append([
                -gx * HA_BOHR_TO_EV_ANG,
                -gy * HA_BOHR_TO_EV_ANG,
                -gz * HA_BOHR_TO_EV_ANG,
            ])
        elif forces:
            break
    return forces if forces else None


def run_orca_sp(atoms, work_dir, nprocs):
    os.makedirs(work_dir, exist_ok=True)
    out_path = os.path.join(work_dir, 'sp.out')

    if os.path.exists(out_path):
        content = open(out_path).read()
        m      = ENERGY_RE.search(content)
        forces = parse_forces(out_path)
        if m and forces is not None:
            return float(m.group(1)) * HA_TO_EV, forces
        # Old energy-only result — rerun with EnGrad
        os.remove(out_path)

    write(os.path.join(work_dir, 'geom.xyz'), atoms)
    with open(os.path.join(work_dir, 'sp.inp'), 'w') as f:
        f.write(ORCA_TEMPLATE)

    with open(out_path, 'w') as fout:
        subprocess.run([ORCA_CMD, os.path.join(work_dir, 'sp.inp')],
                       stdout=fout, stderr=fout, cwd=work_dir)

    content = open(out_path).read()
    m      = ENERGY_RE.search(content)
    forces = parse_forces(out_path)
    if not m or forces is None:
        raise RuntimeError(f'ORCA SP failed in {work_dir}')
    return float(m.group(1)) * HA_TO_EV, forces


def main(args):
    rxn     = args.rxn
    out_dir = f'{OUT_BASE}/{rxn}'
    os.makedirs(out_dir, exist_ok=True)

    out_json = f'{out_dir}/results.json'
    if os.path.exists(out_json):
        existing = json.load(open(out_json))
        if existing.get('geometries') and 'forces_wb97m_eV_per_ang' in existing['geometries'][0]:
            print(f'{rxn}: already done with forces, skipping.')
            return
        os.remove(out_json)
        print(f'{rxn}: rerunning to add forces.')

    positions, atomic_numbers, energies_wb97x, forces_wb97x = load_t1x_reaction(H5FILE, rxn)

    n_total  = len(positions)
    n_sample = min(args.n_images, n_total)
    indices  = np.round(np.linspace(0, n_total - 1, n_sample)).astype(int).tolist()
    print(f'{rxn}: {n_total} T1x images, sampling {n_sample} uniformly', flush=True)

    results = []
    for geom_idx in indices:
        atoms   = Atoms(numbers=atomic_numbers, positions=positions[geom_idx])
        e_wb97x = float(energies_wb97x[geom_idx])
        f_wb97x = forces_wb97x[geom_idx].tolist()

        work_dir         = f'{out_dir}/geom_{geom_idx:04d}'
        e_wb97m, f_wb97m = run_orca_sp(atoms, work_dir, args.n_threads)

        delta_e = e_wb97m - e_wb97x
        delta_f = (np.array(f_wb97m) - np.array(f_wb97x)).tolist()

        print(f'  [{geom_idx}/{n_total-1}] wB97M={e_wb97m:.4f} wB97X={e_wb97x:.4f} '
              f'delta={delta_e:.4f} eV', flush=True)

        results.append({
            'geom_idx':                geom_idx,
            'e_wb97m_eV':              round(e_wb97m, 6),
            'e_wb97x_eV':              round(e_wb97x, 6),
            'delta_eV':                round(delta_e, 6),
            'forces_wb97m_eV_per_ang': [[round(v, 6) for v in row] for row in f_wb97m],
            'forces_wb97x_eV_per_ang': [[round(v, 6) for v in row] for row in f_wb97x],
            'delta_forces_eV_per_ang': [[round(v, 6) for v in row] for row in delta_f],
        })

    summary = {
        'rxn':        rxn,
        'n_total':    n_total,
        'n_sampled':  len(results),
        'geometries': results,
    }

    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    deltas = [r['delta_eV'] for r in results]
    print(f'\n{rxn}: delta mean={np.mean(deltas):.4f} std={np.std(deltas):.4f} eV')
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--n-images',  type=int, default=50)
    parser.add_argument('--n-threads', type=int, default=8)
    args = parser.parse_args()
    main(args)
