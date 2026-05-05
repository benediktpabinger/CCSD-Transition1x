"""
Extract all geometries from neb.db for the 3 test reactions.
Writes one xyz file per geometry under:
  ~/delta_experiment/geometries/rxnXXXX/geom_NNN.xyz

Usage:
    python extract_neb_geometries.py
"""
import os
import ase.db
from ase.io import write

REACTIONS = ['rxn0103', 'rxn1319', 'rxn7952']
NEB_DIR   = '/home/energy/s242862/orca_neb_results'
OUT_BASE  = '/home/energy/s242862/delta_experiment/geometries'

for rxn in REACTIONS:
    db_path = os.path.join(NEB_DIR, rxn, 'neb.db')
    out_dir = os.path.join(OUT_BASE, rxn)
    os.makedirs(out_dir, exist_ok=True)

    with ase.db.connect(db_path) as db:
        rows = list(db.select())

    print(f'{rxn}: {len(rows)} geometries')
    for i, row in enumerate(rows):
        atoms = row.toatoms()
        atoms.info = {}  # strip ORCA calculator info
        write(os.path.join(out_dir, f'geom_{i:03d}.xyz'), atoms)

    print(f'  Written to {out_dir}/')
