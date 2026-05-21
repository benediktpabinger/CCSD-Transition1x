"""
Evaluate MACE + delta head on the test set using wB97M-V NEB geometries.

For each test reaction, loads the last N images from neb.db (the converged MEP).
The wB97M-V energy is read from the same neb.db row — so predictions and ground
truth are evaluated at exactly the same geometries.

Three methods compared on every geometry:
  1. wB97X-D3 (MACE alone):   E_wB97X_pred
  2. wB97M-V (true):          E_wB97M_true  (from neb.db)
  3. MACE + delta head:       E_wB97X_pred + delta_pred

Metrics reported per reaction and overall:
  - MAE on relative energies (meV), referenced to first image per reaction
  - MAE on forward barrier (meV)
  - MAE on reverse barrier (meV)

Output: ~/delta_head/eval_neb_results.json

Usage:
    python eval_delta_neb.py [--n-images 10] [--n-reactions 279]
"""
import argparse
import json
import os
from collections import defaultdict

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
OUT_PATH  = f'{HOME}/delta_head/eval_neb_results.json'

HIDDEN_IRREPS     = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS        = o3.Irreps("16x0e")
NODE_FEATS_OFFSET = 1024

HA_TO_EV = 27.2114


def get_test_reactions(neb_dir):
    return sorted([
        d for d in os.listdir(neb_dir)
        if os.path.isdir(os.path.join(neb_dir, d)) and d.startswith('rxn')
        and os.path.exists(os.path.join(neb_dir, d, 'neb.db'))
    ])


def load_neb_images(rxn, neb_dir, n_images):
    """Load last n_images from neb.db. Returns list of (atoms, e_wb97m_eV)."""
    db_path = os.path.join(neb_dir, rxn, 'neb.db')
    with ase.db.connect(db_path) as db:
        rows = list(db.select())
    rows = rows[-n_images:]
    result = []
    for row in rows:
        atoms = row.toatoms()
        e_wb97m = atoms.get_potential_energy()  # eV, stored from ORCA NEB
        result.append((atoms, e_wb97m))
    return result


def run_batch(atoms_list, model, delta_head, z_table, r_max, device):
    """Returns (e_wb97x_pred, delta_pred, forces_list).
    forces_list[i] is np.array [n_atoms_i, 3] in eV/Å (wB97X-D3 MACE forces)."""
    configs   = [config_from_atoms(a) for a in atoms_list]
    data_list = [AtomicData.from_config(c, z_table=z_table, cutoff=r_max, heads=['Default'])
                 for c in configs]
    batch = torch_geometric.Batch.from_data_list(data_list).to(device)

    batch_dict   = {key: batch[key] for key in batch.keys}
    out          = model(batch_dict, training=False, compute_force=True,
                         compute_virials=False, compute_stress=False)
    e_wb97x_pred = out['energy'].detach().cpu().numpy()
    forces_all   = out['forces'].detach().cpu().numpy()  # [N_atoms_total, 3]

    with torch.no_grad():
        node_feats = out['node_feats'][:, NODE_FEATS_OFFSET:]
        per_atom   = delta_head(node_feats).squeeze(-1)
        per_struct = torch.zeros(batch.num_graphs, device=per_atom.device)
        per_struct.scatter_add_(0, batch.batch, per_atom)
        delta_pred = per_struct.cpu().numpy()

    atom_counts = [(batch.batch == i).sum().item() for i in range(batch.num_graphs)]
    forces_list = np.split(forces_all, np.cumsum(atom_counts[:-1]))

    return e_wb97x_pred, delta_pred, forces_list


def barrier(energies):
    """Forward and reverse barrier from an energy profile (eV). Returns meV."""
    ts = int(np.argmax(energies))
    fwd = (energies[ts] - energies[0]) * 1000
    rev = (energies[ts] - energies[-1]) * 1000
    return fwd, rev


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

    rxns = get_test_reactions(NEB_DIR)
    if args.n_reactions < len(rxns):
        rxns = rxns[:args.n_reactions]
    print(f'Evaluating {len(rxns)} test reactions, last {args.n_images} NEB images each\n')

    all_e_true_rel, all_e_mace_rel, all_e_delta_rel = [], [], []
    barrier_rows = []

    print(f"{'Rxn':12s}  {'N':>3}  {'MAE_MACE':>10}  {'MAE_delta':>10}  "
          f"{'fwd_true':>9}  {'fwd_mace':>9}  {'fwd_delta':>9}")
    print('-' * 80)

    for rxn in rxns:
        try:
            images = load_neb_images(rxn, NEB_DIR, args.n_images)
        except Exception as e:
            print(f'{rxn}: failed to load — {e}')
            continue

        atoms_list   = [a for a, _ in images]
        e_wb97m_true = np.array([e for _, e in images])

        try:
            e_wb97x_pred, delta_pred, forces_list = run_batch(
                atoms_list, model, delta_head, z_table, r_max, device
            )
        except Exception as e:
            print(f'{rxn}: MACE failed — {e}')
            continue

        e_mace_delta_pred = e_wb97x_pred + delta_pred

        # Relative energies (referenced to first image) in eV
        e_true_rel  = e_wb97m_true  - e_wb97m_true[0]
        e_mace_rel  = e_wb97x_pred  - e_wb97x_pred[0]
        e_delta_rel = e_mace_delta_pred - e_mace_delta_pred[0]

        mae_mace  = np.abs(e_mace_rel  - e_true_rel).mean() * 1000
        mae_delta = np.abs(e_delta_rel - e_true_rel).mean() * 1000

        fwd_true,  rev_true  = barrier(e_wb97m_true)
        fwd_mace,  rev_mace  = barrier(e_wb97x_pred)
        fwd_delta, rev_delta = barrier(e_mace_delta_pred)

        all_e_true_rel.extend(e_true_rel.tolist())
        all_e_mace_rel.extend(e_mace_rel.tolist())
        all_e_delta_rel.extend(e_delta_rel.tolist())

        barrier_rows.append({
            'rxn':            rxn,
            'n_images':       len(images),
            'mae_mace_meV':   round(mae_mace, 1),
            'mae_delta_meV':  round(mae_delta, 1),
            'fwd_true_meV':   round(fwd_true, 1),
            'fwd_mace_meV':   round(fwd_mace, 1),
            'fwd_delta_meV':  round(fwd_delta, 1),
            'rev_true_meV':   round(rev_true, 1),
            'rev_mace_meV':   round(rev_mace, 1),
            'rev_delta_meV':  round(rev_delta, 1),
            'forces_wb97x_eV_per_ang': [f.tolist() for f in forces_list],
        })

        print(f'{rxn:12s}  {len(images):>3}  {mae_mace:>9.1f}m  {mae_delta:>9.1f}m  '
              f'{fwd_true:>8.0f}m  {fwd_mace:>8.0f}m  {fwd_delta:>8.0f}m')

    all_e_true_rel  = np.array(all_e_true_rel)
    all_e_mace_rel  = np.array(all_e_mace_rel)
    all_e_delta_rel = np.array(all_e_delta_rel)

    mae_mace_overall  = np.abs(all_e_mace_rel  - all_e_true_rel).mean() * 1000
    mae_delta_overall = np.abs(all_e_delta_rel - all_e_true_rel).mean() * 1000

    fwd_true  = np.array([r['fwd_true_meV']  for r in barrier_rows])
    fwd_mace  = np.array([r['fwd_mace_meV']  for r in barrier_rows])
    fwd_delta = np.array([r['fwd_delta_meV'] for r in barrier_rows])
    rev_true  = np.array([r['rev_true_meV']  for r in barrier_rows])
    rev_mace  = np.array([r['rev_mace_meV']  for r in barrier_rows])
    rev_delta = np.array([r['rev_delta_meV'] for r in barrier_rows])

    mae_fwd_mace  = np.abs(fwd_mace  - fwd_true).mean()
    mae_fwd_delta = np.abs(fwd_delta - fwd_true).mean()
    mae_rev_mace  = np.abs(rev_mace  - rev_true).mean()
    mae_rev_delta = np.abs(rev_delta - rev_true).mean()

    print('\n' + '=' * 80)
    print(f'Overall ({len(barrier_rows)} reactions, {len(all_e_true_rel)} geoms):')
    print(f'  Relative energy MAE:')
    print(f'    wB97X-D3 (MACE alone):  {mae_mace_overall:.1f} meV')
    print(f'    MACE + delta head:      {mae_delta_overall:.1f} meV')
    print(f'  Forward barrier MAE:')
    print(f'    wB97X-D3 (MACE alone):  {mae_fwd_mace:.1f} meV')
    print(f'    MACE + delta head:      {mae_fwd_delta:.1f} meV')
    print(f'  Reverse barrier MAE:')
    print(f'    wB97X-D3 (MACE alone):  {mae_rev_mace:.1f} meV')
    print(f'    MACE + delta head:      {mae_rev_delta:.1f} meV')

    summary = {
        'n_reactions':            len(barrier_rows),
        'n_geoms':                len(all_e_true_rel),
        'n_images_per_rxn':       args.n_images,
        'mae_energy_mace_meV':    round(mae_mace_overall, 1),
        'mae_energy_delta_meV':   round(mae_delta_overall, 1),
        'mae_fwd_mace_meV':       round(mae_fwd_mace, 1),
        'mae_fwd_delta_meV':      round(mae_fwd_delta, 1),
        'mae_rev_mace_meV':       round(mae_rev_mace, 1),
        'mae_rev_delta_meV':      round(mae_rev_delta, 1),
        'reactions':              barrier_rows,
    }

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.ndarray):  return o.tolist()
            return super().default(o)

    with open(OUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2, cls=_Enc)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-images',    type=int, default=10,
                        help='Number of final NEB images to use per reaction (default 10)')
    parser.add_argument('--n-reactions', type=int, default=999,
                        help='Max reactions to evaluate (default: all)')
    args = parser.parse_args()
    main(args)
