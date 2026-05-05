"""
Generate ORCA single-point input files at wB97X-D3/6-31G(d) (T1x level)
for every geometry extracted from neb.db.

Creates:
  ~/delta_experiment/orca_sp/rxnXXXX/geom_NNN/sp.inp

Usage:
    python generate_orca_inputs.py
"""
import os
from ase.io import read

REACTIONS = ['rxn0103', 'rxn1319', 'rxn7952']
GEOM_BASE = '/home/energy/s242862/delta_experiment/geometries'
SP_BASE   = '/home/energy/s242862/delta_experiment/orca_sp'

ORCA_TEMPLATE = """\
! wB97X-D3 6-31G(d) TightSCF

%pal nprocs 4 end
%maxcore 4000

* xyzfile {charge} {mult} geom.xyz
"""

for rxn in REACTIONS:
    geom_dir = os.path.join(GEOM_BASE, rxn)
    geom_files = sorted(f for f in os.listdir(geom_dir) if f.endswith('.xyz'))

    for geom_file in geom_files:
        name = geom_file.replace('.xyz', '')
        out_dir = os.path.join(SP_BASE, rxn, name)
        os.makedirs(out_dir, exist_ok=True)

        # Copy geometry
        geom_src = os.path.join(geom_dir, geom_file)
        geom_dst = os.path.join(out_dir, 'geom.xyz')
        atoms = read(geom_src)
        from ase.io import write
        write(geom_dst, atoms, format='xyz')

        # Write ORCA input
        inp = ORCA_TEMPLATE.format(charge=0, mult=1)
        with open(os.path.join(out_dir, 'sp.inp'), 'w') as f:
            f.write(inp)

    n = len(geom_files)
    print(f'{rxn}: {n} input files written to {os.path.join(SP_BASE, rxn)}/')
