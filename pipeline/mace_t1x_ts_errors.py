"""The thesis MACE model (trained on Transition1x, wB97X/6-31G(d), restricted only) against its own
training labels at every Transition1x transition state, for comparison with the OMol25 models
against the OMol25 labels at the same structures (results/model_sp_ts_all.csv).

Per transition state: force MAE (mean absolute component, eV/A) and energy error (meV) of
~/mace_t1x_p10_compiled.model against the wB97x_6-31G(d) label stored in Transition1x.h5, with the
split (train / val / test) so that seen and unseen transition states can be told apart.

Output results/mace_t1x_ts_errors.csv

    python pipeline/mace_t1x_ts_errors.py
"""
import csv
import os

import h5py
import numpy as np
from ase import Atoms
from mace.calculators import MACECalculator

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MODEL = f'{H}/mace_t1x_p10_compiled.model'
H5 = f'{H}/data/Transition1x.h5'

calc = MACECalculator(model_paths=MODEL, device=os.environ.get('DEVICE', 'cpu'), default_dtype='float64')
rows = []
with h5py.File(H5, 'r') as f:
    for split in ('train', 'val', 'test'):
        for formula, grp in f[split].items():
            for rx, r in grp.items():
                t = r['transition_state']
                a = Atoms(numbers=t['atomic_numbers'][()], positions=t['positions'][0])
                a.calc = calc
                e, fm = a.get_potential_energy(), a.get_forces()
                e0, f0 = float(t['wB97x_6-31G(d).energy'][0]), t['wB97x_6-31G(d).forces'][0]
                rows.append(dict(rxn=rx, split=split, n_atoms=len(a), de_mev=round(1000 * (e - e0), 3),
                                 f_mae=round(float(np.abs(fm - f0).mean()), 6)))
        print(split, len(rows), flush=True)
with open(f'{REPO}/results/mace_t1x_ts_errors.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('wrote', len(rows))
