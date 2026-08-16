"""Do the stored model forces belong to the stored model geometry?

The force-error result rests on a comparison between two numbers taken from two
different places: the forces written into the model's transition_state.xyz, and
an ORCA gradient computed at the coordinates in that same file.  The ORCA side
is unambiguous -- it was computed from those coordinates.  The model side was
not recomputed; it was read out of the file.

Geometry and forces sit in the same line of the same extxyz, written by ASE
from one snapshot, so they ought to correspond.  But if ASE had written the
positions of step N and the results of step N-1, nothing in the file would show
it, and part of what we call "the model's force error" would be a displacement
error instead.

This script closes that gap the direct way: load the same checkpoint, put the
calculator on the geometry from the file, and compare the fresh forces with the
stored ones.  Agreement to numerical noise means the chain is tight.  A
systematic difference means the force analysis has to be redone from fresh
single points.

Nothing here re-optimises anything.  One energy and one force evaluation per
structure, at the coordinates as they stand.
"""
import argparse
import json
import os
import sys

import numpy as np
from ase.io import read

H = '/home/energy/s242862'
MODELS = {
    'UMA-S': ('uma_neb_results', f'{H}/checkpoints/uma-s-1p2.pt', 'omol'),
    'UMA-M': ('uma_m_neb_results', f'{H}/checkpoints/uma-m-1p1.pt', 'omol'),
    'eSEN': ('esen_neb_results', f'{H}/checkpoints/esen_sm_conserving_all.pt', None),
}

MR = ['rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885', 'rxn6196',
      'rxn0346', 'rxn4518', 'rxn3107', 'rxn8837', 'rxn7060', 'rxn5691',
      'rxn1283', 'rxn8827', 'rxn4522', 'rxn1147', 'rxn0894', 'rxn7957',
      'rxn5690']


def make_calc(ckpt, task):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    unit = pretrained_mlip.load_predict_unit(ckpt, device='cuda')
    # UMA needs the task; eSEN is single-task and rejects the argument.
    return (FAIRChemCalculator(unit, task_name=task) if task
            else FAIRChemCalculator(unit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    subdir, ckpt, task = MODELS[a.model]
    if not os.path.exists(ckpt):
        print(f'checkpoint fehlt: {ckpt}')
        sys.exit(1)

    print(f'{a.model}: {ckpt}  task={task}')
    calc = make_calc(ckpt, task)

    rows = []
    print(f'{"rxn":<9}{"dE [meV]":>11}{"MAE F":>10}{"max dF":>10}'
          f'{"|F| gespeichert":>17}{"|F| neu":>10}   Urteil')
    print('-' * 82)
    for rx in MR:
        p = f'{H}/{subdir}/{rx}/transition_state.xyz'
        if not os.path.exists(p):
            continue
        at = read(p)                      # stored results land in a SPCalculator
        try:
            e_old = float(at.get_potential_energy())
            f_old = np.array(at.get_forces())
        except Exception as exc:
            print(f'{rx:<9}  keine gespeicherten Werte ({exc})')
            continue

        pos_before = at.get_positions().copy()
        at.calc = calc
        e_new = float(at.get_potential_energy())
        f_new = np.array(at.get_forces())
        # the calculator must not have moved anything
        assert np.abs(at.get_positions() - pos_before).max() == 0.0

        dE = (e_new - e_old) * 1000.0
        dF = f_new - f_old
        mae = float(np.abs(dF).mean())
        mx = float(np.abs(dF).max())
        fo = float(np.abs(f_old).max())
        fn = float(np.abs(f_new).max())
        # noise level for a deterministic model on identical input is ~1e-6;
        # anything at the size of the reported error (0.03) is a real problem
        verdict = ('identisch' if mx < 1e-4 else
                   'kleine Abweichung' if mx < 5e-3 else
                   '*** WEICHT AB ***')
        print(f'{rx:<9}{dE:>11.4f}{mae:>10.5f}{mx:>10.5f}{fo:>17.4f}'
              f'{fn:>10.4f}   {verdict}')
        rows.append({'rxn': rx, 'model': a.model, 'dE_meV': dE,
                     'mae': mae, 'maxdiff': mx, 'f_stored': fo, 'f_new': fn})

    if rows:
        mx = [r['maxdiff'] for r in rows]
        print()
        print(f'  n = {len(rows)}   median max|dF| = {np.median(mx):.2e}   '
              f'groesste = {max(mx):.2e} eV/A')
        print(f'  zum Vergleich: der berichtete Kraftfehler gegen DFT '
              f'liegt bei 0.031 eV/A')
        bad = [r for r in rows if r['maxdiff'] >= 5e-3]
        print(f'  Strukturen mit relevanter Abweichung: {len(bad)}')
        for r in bad:
            print(f'     {r["rxn"]}  max|dF| {r["maxdiff"]:.4f}')

    out = a.out or f'{H}/model_sp_recheck_{a.model}.json'
    json.dump(rows, open(out, 'w'), indent=1)
    print(f'\ngeschrieben: {out}')


if __name__ == '__main__':
    main()
