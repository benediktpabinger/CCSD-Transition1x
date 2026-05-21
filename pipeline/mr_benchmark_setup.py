"""
MR benchmark — Step 1 setup: extract all NEB geometries and generate
wB97X-D3/6-31G(d) ORCA single-point inputs for the top 10 MR reactions.

Creates:
  ~/mr_benchmark/geometries/{rxn}/geom_{i:04d}.xyz   (raw geometries)
  ~/mr_benchmark/orca_sp/{rxn}/geom_{i:04d}/sp.inp   (ORCA inputs)

Usage:
    python mr_benchmark_setup.py
"""
import os
import ase.db
from ase.io import read, write

TOP10 = [
    'rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885',
    'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150',
]

NEB_DIR  = '/home/energy/s242862/orca_neb_results'
BASE     = '/home/energy/s242862/mr_benchmark'
GEOM_DIR = f'{BASE}/geometries'
SP_DIR   = f'{BASE}/orca_sp'

ORCA_TEMPLATE = """\
! wB97X-D3 6-31G(d) TightSCF

%pal nprocs 4 end
%maxcore 4000

* xyzfile 0 1 geom.xyz
"""

total_geoms = 0
total_inputs = 0

for rxn in TOP10:
    db_path = f'{NEB_DIR}/{rxn}/neb.db'
    geom_out = f'{GEOM_DIR}/{rxn}'
    sp_out   = f'{SP_DIR}/{rxn}'
    os.makedirs(geom_out, exist_ok=True)

    with ase.db.connect(db_path) as db:
        rows = list(db.select())

    for i, row in enumerate(rows):
        atoms = row.toatoms()
        atoms.info = {}

        xyz_path = f'{geom_out}/geom_{i:04d}.xyz'
        write(xyz_path, atoms)

        inp_dir = f'{sp_out}/geom_{i:04d}'
        os.makedirs(inp_dir, exist_ok=True)

        # copy geom
        write(f'{inp_dir}/geom.xyz', atoms)

        # skip if already done
        out_path = f'{inp_dir}/sp.out'
        if os.path.exists(out_path) and 'FINAL SINGLE POINT ENERGY' in open(out_path).read():
            continue

        with open(f'{inp_dir}/sp.inp', 'w') as f:
            f.write(ORCA_TEMPLATE)
        total_inputs += 1

    total_geoms += len(rows)
    print(f'{rxn}: {len(rows)} geometries → {sp_out}/')

print(f'\nTotal geometries : {total_geoms}')
print(f'ORCA inputs written: {total_inputs}')
