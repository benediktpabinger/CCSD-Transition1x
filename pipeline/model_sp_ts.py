"""Modell-Einzelpunkte an den gespeicherten Transition1x-Uebergangszustaenden.

Ein OMol25-Modell (UMA-S, UMA-M oder MACE-OMol) rechnet Energie und Kraefte
an den Geometrien aus ~/t1x_ts_all.json, ohne Optimierung. Die DFT-Referenz
an denselben Geometrien ist das OMol25-Label (~/omol25_ts_all.jsonl); der
Vergleich passiert spaeter in pipeline/model_sp_table.py. Dieses Skript
schreibt nur, was das Modell sagt.

Ladung und Spin werden EXPLIZIT in atoms.info gesetzt (--charge, --spin);
'spin' ist die Multiplizitaet (1 fuer ein Singulett), wie in OMol25 und ORCA.
fairchem 2.20 setzt fuer task_name='omol' fehlende Werte selbst auf charge=0,
spin=1 (validate_atoms_data, mit Warnung), die Thesis-NEBs ohne atoms.info
liefen also richtig. spin=0 waere kein Zustand und ist katastrophal (Spin-Test
2026-10-29 an den 45 Ankern: UMA-S 3.2 eV daneben, Kraft-MAE 0.33 eV/A; mit
spin=1 2.4 meV und 0.013 eV/A). Fuer MACE-OMol prueft derselbe Test
(--spin 0/1/2), welcher Wert zum Singulett-Label passt.

Ausgabe: JSONL, eine Zeile je Reaktion:
    rxn, model, charge, spin, energy_ev, forces_ev_a (N x 3), fmax_ev_a, seconds

    python pipeline/model_sp_ts.py --model uma-s --spin 1 --out ~/model_sp_ts/uma-s_spin1.jsonl
    python pipeline/model_sp_ts.py --model mace-omol --spin 1 --rxns ~/anchors45.txt --out ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from ase import Atoms

H = os.path.expanduser('~')
CHECKPOINTS = {'uma-s': f'{H}/checkpoints/uma-s-1p2.pt',
               'uma-m': f'{H}/checkpoints/uma-m-1p1.pt',
               'mace-omol': f'{H}/.cache/mace/MACE-omol-0-extra-large-1024.model'}


def make_calc(model, device):
    if model.startswith('uma'):
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        pu = pretrained_mlip.load_predict_unit(CHECKPOINTS[model], device=device)
        return FAIRChemCalculator(pu, task_name='omol')
    from mace.calculators import mace_omol
    return mace_omol(model=CHECKPOINTS[model], device=device, default_dtype='float64')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(CHECKPOINTS))
    ap.add_argument('--charge', type=int, default=0)
    ap.add_argument('--spin', type=int, default=1)
    ap.add_argument('--rxns', help='Datei mit einer Reaktion je Zeile; sonst alle im H5')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    ts = json.load(open(f'{H}/t1x_ts_all.json'))
    rxns = sorted(ts)
    if a.rxns:
        want = {l.strip() for l in open(a.rxns) if l.strip()}
        rxns = [r for r in rxns if r in want]
        missing = want - set(rxns)
        if missing:
            sys.exit(f'ABBRUCH: {len(missing)} Reaktionen nicht im H5: {sorted(missing)[:5]}')
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['rxn'] for l in open(a.out)}
    todo = [r for r in rxns if r not in done]
    print(f'{a.model}  charge {a.charge}  spin {a.spin}  {len(todo)} Reaktionen '
          f'({len(done)} schon in {a.out})', flush=True)

    calc = make_calc(a.model, a.device)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t_all = time.time()
    with open(a.out, 'a') as fh:
        for i, rx in enumerate(todo):
            t = ts[rx]
            atoms = Atoms(numbers=t['numbers'], positions=t['positions'])
            atoms.info['charge'] = a.charge
            atoms.info['spin'] = a.spin
            atoms.calc = calc
            t0 = time.time()
            e = float(atoms.get_potential_energy())
            F = np.asarray(atoms.get_forces(), dtype=float)
            fh.write(json.dumps(dict(
                rxn=rx, model=a.model, charge=a.charge, spin=a.spin,
                energy_ev=e, forces_ev_a=F.tolist(), fmax_ev_a=float(np.abs(F).max()),
                seconds=round(time.time() - t0, 3))) + '\n')
            if i % 500 == 0 or i == len(todo) - 1:
                fh.flush()
                print(f'  {i + 1}/{len(todo)}  {rx}  E = {e:.4f} eV  fmax = {np.abs(F).max():.4f}  '
                      f'({(time.time() - t_all) / 60:.1f} min)', flush=True)
    print(f'ENDE {a.model}: {len(todo)} Reaktionen in {(time.time() - t_all) / 60:.1f} min', flush=True)
