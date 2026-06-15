"""
Compute wB97M-V/def2-TZVP forces for the last 10 NEB images of Val A reactions.

Reads existing val_delta_sp/{rxn}/results.json (which already has wB97X-D3
forces), runs ORCA wB97M-V EnGrad on the last 10 geometries, and adds:
    forces_wb97m_eV_per_ang, delta_forces_eV_per_ang

Scratch goes to /home/scratch3 to avoid NFS IO load.

Usage:
    python compute_val_a_forces.py <rxn> [--nprocs 8]
"""
import argparse, json, os, re, subprocess
import numpy as np
import ase.db
from ase import Atoms
from ase.io import write

HOME     = '/home/energy/s242862'
NEB_BASE = f'{HOME}/orca_neb_val_results'
VAL_SP   = f'{HOME}/val_delta_sp'
SCRATCH  = '/home/scratch3/s242862/val_a_forces'
ORCA_CMD = '/home/modules/software/ORCA/5.0.4-gompi-2023a/bin/orca'

HA_TO_EV          = 27.2114
HA_BOHR_TO_EV_ANG = 27.2114 / 0.529177

ENERGY_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')

ORCA_TEMPLATE = """\
! wB97M-V def2-TZVP def2/J RIJCOSX TightSCF EnGrad

%pal nprocs {nprocs} end
%maxcore 4000
%scf maxiter 200 end

* xyzfile 0 1 geom.xyz
"""


def parse_forces(out_path):
    content = open(out_path).read()
    idx = content.find('CARTESIAN GRADIENT')
    if idx == -1:
        return None
    lines = content[idx:].split('\n')
    forces = []
    for line in lines[3:]:
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


def run_orca(atoms, work_dir, nprocs):
    os.makedirs(work_dir, exist_ok=True)
    out_path = os.path.join(work_dir, 'sp.out')

    if os.path.exists(out_path):
        content = open(out_path).read()
        m = ENERGY_RE.search(content)
        forces = parse_forces(out_path)
        if m and forces is not None:
            return float(m.group(1)) * HA_TO_EV, forces
        os.remove(out_path)

    write(os.path.join(work_dir, 'geom.xyz'), atoms)
    with open(os.path.join(work_dir, 'sp.inp'), 'w') as f:
        f.write(ORCA_TEMPLATE.format(nprocs=nprocs))

    with open(out_path, 'w') as fout:
        subprocess.run([ORCA_CMD, os.path.join(work_dir, 'sp.inp')],
                       stdout=fout, stderr=fout, cwd=work_dir)

    content = open(out_path).read()
    m = ENERGY_RE.search(content)
    forces = parse_forces(out_path)
    if not m or forces is None:
        raise RuntimeError(f'ORCA failed in {work_dir}')
    return float(m.group(1)) * HA_TO_EV, forces


def main(args):
    rxn = args.rxn
    json_path = f'{VAL_SP}/{rxn}/results.json'

    if not os.path.exists(json_path):
        print(f'{rxn}: no results.json, skipping')
        return

    data = json.load(open(json_path))
    geoms = data['geometries']

    last10 = geoms[-10:]
    need = [g for g in last10 if 'forces_wb97m_eV_per_ang' not in g]

    if not need:
        print(f'{rxn}: all last 10 geoms already have wB97M-V forces, skipping')
        return

    db_path = f'{NEB_BASE}/{rxn}/neb.db'
    with ase.db.connect(db_path) as db:
        rows = list(db.select())
    atomic_numbers = rows[0].toatoms().get_atomic_numbers()

    print(f'{rxn}: computing wB97M-V forces for {len(need)}/10 geoms')

    scratch_dir = f'{SCRATCH}/{rxn}'
    os.makedirs(scratch_dir, exist_ok=True)

    for g in need:
        idx = g['geom_idx']
        positions = rows[idx].toatoms().get_positions()
        atoms = Atoms(numbers=atomic_numbers, positions=positions)

        work_dir = f'{scratch_dir}/geom_{idx:04d}'
        e_wb97m, f_wb97m = run_orca(atoms, work_dir, args.nprocs)

        f_wb97x = np.array(g['forces_wb97x_eV_per_ang'])
        delta_f = (np.array(f_wb97m) - f_wb97x).tolist()

        g['forces_wb97m_eV_per_ang'] = [[round(v, 6) for v in row] for row in f_wb97m]
        g['delta_forces_eV_per_ang'] = [[round(v, 6) for v in row] for row in delta_f]

        print(f'  geom_{idx:04d}: e_wb97m={e_wb97m:.4f} eV', flush=True)

    json.dump(data, open(json_path, 'w'), indent=2)
    print(f'{rxn}: saved updated results.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--nprocs', type=int, default=8)
    main(parser.parse_args())
