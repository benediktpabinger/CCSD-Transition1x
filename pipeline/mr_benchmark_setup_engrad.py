"""
MR benchmark — setup wB97X-D3/6-31G(d) EnGrad inputs for all 30 benchmark reactions.

Uses the last 10 NEB images from neb.db (matching what eval_delta_neb evaluates).
Writes inputs to ~/mr_benchmark/orca_engrad/{rxn}/geom_{i:04d}/

Run on cluster, then submit job_mr_sp_engrad.sh.

Usage:
    python mr_benchmark_setup_engrad.py
"""
import os
import ase.db
from ase.io import write

ALL30 = [
    # Top 10 (highest NFOD)
    'rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885',
    'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150',
    # Bottom 10 (lowest NFOD)
    'rxn9246', 'rxn4498', 'rxn1061', 'rxn4003', 'rxn4004',
    'rxn4063', 'rxn4114', 'rxn4060', 'rxn1961', 'rxn1962',
    # Middle 10 (evenly spaced)
    'rxn0896', 'rxn1154', 'rxn5690', 'rxn4513', 'rxn7955',
    'rxn4519', 'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155',
]

N_IMAGES = 10

NEB_DIR  = '/home/energy/s242862/orca_neb_results'
BASE     = '/home/energy/s242862/mr_benchmark'
OUT_DIR  = f'{BASE}/orca_engrad'

ORCA_TEMPLATE = """\
! wB97X-D3 6-31G(d) TightSCF EnGrad

%pal nprocs 4 end
%maxcore 4000

* xyzfile 0 1 geom.xyz
"""

total_inputs = 0

for rxn in ALL30:
    db_path = os.path.join(NEB_DIR, rxn, 'neb.db')
    if not os.path.exists(db_path):
        print(f'{rxn}: neb.db not found, skipping')
        continue

    with ase.db.connect(db_path) as db:
        rows = list(db.select())

    rows = rows[-N_IMAGES:]

    for i, row in enumerate(rows):
        atoms = row.toatoms()
        atoms.info = {}

        inp_dir = os.path.join(OUT_DIR, rxn, f'geom_{i:04d}')
        os.makedirs(inp_dir, exist_ok=True)

        write(os.path.join(inp_dir, 'geom.xyz'), atoms)

        out_path = os.path.join(inp_dir, 'sp.out')
        if os.path.exists(out_path) and 'FINAL SINGLE POINT ENERGY' in open(out_path).read():
            continue

        with open(os.path.join(inp_dir, 'sp.inp'), 'w') as f:
            f.write(ORCA_TEMPLATE)
        total_inputs += 1

    print(f'{rxn}: {len(rows)} images -> {OUT_DIR}/{rxn}/')

print(f'\nTotal ORCA inputs written: {total_inputs}')
print(f'Submit: sbatch --array=0-{(total_inputs + 1) // 2 - 1}%60 job_mr_sp_engrad.sh')
