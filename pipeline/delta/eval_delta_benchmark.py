"""
Evaluate MACE + delta head on the 10 MR benchmark reactions.

For these reactions we have all four quantities on the same geometries:
  1. E_wB97X_true  — ORCA wB97X-D3/6-31G(d) SPs (from mr_benchmark/results/{rxn}_sp.json)
  2. E_wB97M_true  — wB97M-V/def2-TZVP from neb.db (same rows)
  3. E_wB97X_MACE  — MACE prediction on those geometries
  4. E_wB97M_pred  — MACE + delta head prediction

Barriers extracted from the last N images (converged MEP).
All comparisons use relative energies (referenced to first image) to cancel offsets.

Output: ~/delta_head/eval_benchmark_results.json

Usage:
    python eval_delta_benchmark.py [--n-images 10]
"""
import argparse
import json
import os

import ase.db
import numpy as np
import torch
import torch.nn.functional as F
torch.serialization.add_safe_globals([slice])  # must be before e3nn import (PyTorch 2.6+)
from ase import Atoms
from e3nn import o3
from mace.calculators import MACECalculator
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.blocks import NonLinearReadoutBlock
from mace.tools import torch_geometric

HOME      = '/home/energy/s242862'
MODEL     = f'{HOME}/mace_t1x_p10_compiled.model'
HEAD_PATH = f'{HOME}/delta_head/delta_head.pt'
NEB_DIR   = f'{HOME}/orca_neb_results'
SP_DIR    = f'{HOME}/mr_benchmark/results'
OUT_PATH  = f'{HOME}/delta_head/eval_benchmark_results.json'

TOP10 = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
         'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150']

HIDDEN_IRREPS     = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS        = o3.Irreps("16x0e")
NODE_FEATS_OFFSET = 1024


def load_data(rxn, neb_dir, sp_dir, n_images):
    """
    Load last n_images from neb.db and match with ORCA wB97X-D3 SPs.
    Returns list of dicts with atoms, e_wb97m_true, e_wb97x_true (or None).
    """
    db_path = os.path.join(neb_dir, rxn, 'neb.db')
    sp_path = os.path.join(sp_dir, f'{rxn}_sp.json')

    with ase.db.connect(db_path) as db:
        rows = list(db.select())

    # Build geom_idx → e_wb97x_true lookup from SP json
    wb97x_by_idx = {}
    if os.path.exists(sp_path):
        with open(sp_path) as f:
            sp_data = json.load(f)
        for g in sp_data.get('geometries', []):
            wb97x_by_idx[g['geom_idx']] = g['e_wb97x_eV']

    rows = rows[-n_images:]
    result = []
    for i_global, row in enumerate(rows):
        geom_idx = len(list(ase.db.connect(db_path).select())) - n_images + i_global
        atoms    = row.toatoms()
        e_wb97m  = atoms.get_potential_energy()
        e_wb97x  = wb97x_by_idx.get(geom_idx, None)
        result.append({'atoms': atoms, 'e_wb97m_true': e_wb97m,
                       'e_wb97x_true': e_wb97x, 'geom_idx': geom_idx})
    return result


def run_batch(atoms_list, model, delta_head, z_table, r_max, device):
    configs   = [config_from_atoms(a) for a in atoms_list]
    data_list = [AtomicData.from_config(c, z_table=z_table, cutoff=r_max, heads=['Default'])
                 for c in configs]
    batch = torch_geometric.Batch.from_data_list(data_list).to(device)
    with torch.no_grad():
        batch_dict   = {key: batch[key] for key in batch.keys}
        out          = model(batch_dict, training=False, compute_force=False,
                             compute_virials=False, compute_stress=False)
        e_wb97x_mace = out['energy'].cpu().numpy()
        node_feats   = out['node_feats'][:, NODE_FEATS_OFFSET:]
        per_atom     = delta_head(node_feats).squeeze(-1)
        per_struct   = torch.zeros(batch.num_graphs, device=per_atom.device)
        per_struct.scatter_add_(0, batch.batch, per_atom)
        delta_pred   = per_struct.cpu().numpy()
    return e_wb97x_mace, delta_pred


def barrier_meV(energies):
    ts  = int(np.argmax(energies))
    fwd = (energies[ts] - energies[0]) * 1000
    rev = (energies[ts] - energies[-1]) * 1000
    return round(fwd, 1), round(rev, 1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    calc  = MACECalculator(model_paths=MODEL, device=str(device), default_dtype='float32')
    model = calc.models[0]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    z_table = calc.z_table
    r_max   = float(model.r_max)

    delta_head = NonLinearReadoutBlock(
        irreps_in=HIDDEN_IRREPS, MLP_irreps=MLP_IRREPS, gate=F.silu
    ).to(device)
    delta_head.load_state_dict(torch.load(HEAD_PATH, map_location=device, weights_only=True))
    delta_head.eval()
    print(f'Loaded MACE + delta head\n')

    hdr = f"{'Rxn':12s}  {'fwd_wb97x':>10}  {'fwd_mace':>9}  {'fwd_delta':>10}  {'fwd_true':>9}  {'has_sp':>6}"
    print(hdr)
    print('-' * len(hdr))

    rows = []
    for rxn in TOP10:
        data = load_data(rxn, NEB_DIR, SP_DIR, args.n_images)

        atoms_list     = [d['atoms']        for d in data]
        e_wb97m_true   = np.array([d['e_wb97m_true']  for d in data])
        e_wb97x_true   = np.array([d['e_wb97x_true'] if d['e_wb97x_true'] is not None
                                   else np.nan for d in data])
        has_sp = not np.all(np.isnan(e_wb97x_true))

        e_wb97x_mace, delta_pred = run_batch(
            atoms_list, model, delta_head, z_table, r_max, device
        )
        e_wb97m_pred = e_wb97x_mace + delta_pred

        # Relative energies
        e_wb97m_true_rel  = e_wb97m_true  - e_wb97m_true[0]
        e_wb97x_mace_rel  = e_wb97x_mace  - e_wb97x_mace[0]
        e_wb97m_pred_rel  = e_wb97m_pred  - e_wb97m_pred[0]
        e_wb97x_true_rel  = e_wb97x_true  - e_wb97x_true[0]  # may have NaN

        fwd_true,  rev_true  = barrier_meV(e_wb97m_true)
        fwd_mace,  rev_mace  = barrier_meV(e_wb97x_mace)
        fwd_delta, rev_delta = barrier_meV(e_wb97m_pred)

        if has_sp and not np.any(np.isnan(e_wb97x_true)):
            fwd_wb97x, rev_wb97x = barrier_meV(e_wb97x_true)
        else:
            fwd_wb97x = rev_wb97x = None

        mae_mace  = np.abs(e_wb97x_mace_rel - e_wb97m_true_rel).mean() * 1000
        mae_delta = np.abs(e_wb97m_pred_rel  - e_wb97m_true_rel).mean() * 1000

        fwd_wb97x_str = f'{fwd_wb97x:>9.0f}m' if fwd_wb97x is not None else '       N/A'
        print(f'{rxn:12s}  {fwd_wb97x_str}  {fwd_mace:>8.0f}m  {fwd_delta:>9.0f}m  '
              f'{fwd_true:>8.0f}m  {str(has_sp):>6}')

        rows.append({
            'rxn':             rxn,
            'has_wb97x_sp':    has_sp,
            'fwd_wb97m_true':  fwd_true,
            'rev_wb97m_true':  rev_true,
            'fwd_wb97x_true':  fwd_wb97x,
            'rev_wb97x_true':  rev_wb97x,
            'fwd_wb97x_mace':  fwd_mace,
            'rev_wb97x_mace':  rev_mace,
            'fwd_wb97m_pred':  fwd_delta,
            'rev_wb97m_pred':  rev_delta,
            'mae_mace_meV':    round(mae_mace, 1),
            'mae_delta_meV':   round(mae_delta, 1),
        })

    print()
    has_sp_rows = [r for r in rows if r['has_wb97x_sp']]
    if has_sp_rows:
        mae_fwd_wb97x  = np.mean([abs(r['fwd_wb97x_true'] - r['fwd_wb97m_true']) for r in has_sp_rows])
        mae_fwd_mace   = np.mean([abs(r['fwd_wb97x_mace'] - r['fwd_wb97m_true']) for r in rows])
        mae_fwd_delta  = np.mean([abs(r['fwd_wb97m_pred'] - r['fwd_wb97m_true']) for r in rows])
        print(f'Forward barrier MAE vs wB97M-V true ({len(rows)} reactions):')
        print(f'  wB97X-D3 true (ORCA SP):  {mae_fwd_wb97x:.0f} meV  ({len(has_sp_rows)} rxns)')
        print(f'  MACE alone:               {mae_fwd_mace:.0f} meV')
        print(f'  MACE + delta head:        {mae_fwd_delta:.0f} meV')

    with open(OUT_PATH, 'w') as f:
        json.dump({'n_images': args.n_images, 'reactions': rows}, f, indent=2)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-images', type=int, default=10)
    args = parser.parse_args()
    main(args)
