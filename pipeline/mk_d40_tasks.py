"""ORCA task list for D40: DFT at the reactant of every freezing-string run of all four models
that has a D37 transition-state gradient. Task lines rxn_<model>_r:xyz:nel for
pipeline/job_border_zone.sh, output ~/orca6_fsm_models.

    python pipeline/mk_d40_tasks.py   -> ~/d40_tasks.txt
"""
import os
import sys

from ase.io import read

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d37_table import M, dft_forces, runs  # noqa: E402

H = os.path.expanduser('~')
lines = []
for m in M:
    for rx, r, d, dft in runs(m):
        if dft_forces(f'{dft}/uks_engrad.out') is None:
            continue
        nel = int(read(f'{d}/reactant.xyz').numbers.sum())
        lines.append(f'{rx}_{m}_r:{d}/reactant.xyz:{nel}')
open(f'{H}/d40_tasks.txt', 'w').write('\n'.join(lines) + '\n')
print(len(lines), 'reactants')
