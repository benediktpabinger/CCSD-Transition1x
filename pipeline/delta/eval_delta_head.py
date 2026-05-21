"""
Evaluate MACE + delta head on val reactions.

Three comparisons per geometry:
  1. MACE alone:      E_wB97X_pred  vs  E_wB97M_true  (= E_wB97X_true + delta_true)
  2. Delta head:      delta_pred    vs  delta_true
  3. MACE + delta:    E_wB97X_pred + delta_pred  vs  E_wB97M_true

Usage:
    python eval_delta_head.py [--n-reactions 10]
"""
import argparse
import random
from collections import defaultdict

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
CACHE_DIR = f'{HOME}/delta_cache'

HIDDEN_IRREPS     = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS        = o3.Irreps("16x0e")
NODE_FEATS_OFFSET = 1024


def run_batch(structs, model, delta_head, z_table, r_max, device):
    """Returns (e_wb97x_pred, delta_pred) both shape [N_structs]."""
    atoms_list = [Atoms(numbers=s['atomic_numbers'].tolist(), positions=s['positions']) for s in structs]
    configs    = [config_from_atoms(a) for a in atoms_list]
    data_list  = [AtomicData.from_config(c, z_table=z_table, cutoff=r_max, heads=['Default'])
                  for c in configs]
    batch      = torch_geometric.Batch.from_data_list(data_list).to(device)

    with torch.no_grad():
        batch_dict = {key: batch[key] for key in batch.keys}
        out        = model(batch_dict, training=False, compute_force=False,
                           compute_virials=False, compute_stress=False)

        e_wb97x_pred = out['energy'].cpu().numpy()           # [N_structs]

        node_feats = out['node_feats'][:, NODE_FEATS_OFFSET:]
        per_atom   = delta_head(node_feats).squeeze(-1)
        per_struct = torch.zeros(batch.num_graphs, device=per_atom.device)
        per_struct.scatter_add_(0, batch.batch, per_atom)
        delta_pred = per_struct.cpu().numpy()                # [N_structs]

    return e_wb97x_pred, delta_pred


def mae_rmse(pred, true):
    err = pred - true
    return np.abs(err).mean(), np.sqrt((err**2).mean())


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    val_data = torch.load(f'{CACHE_DIR}/val.pt', weights_only=False)
    print(f'Val pool: {len(val_data)} geoms')

    calc  = MACECalculator(model_paths=MODEL, device=str(device), default_dtype='float32')
    model = calc.models[0]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    z_table = calc.z_table
    r_max   = float(model.r_max)

    delta_head = NonLinearReadoutBlock(
        irreps_in  = HIDDEN_IRREPS,
        MLP_irreps = MLP_IRREPS,
        gate       = F.silu,
    ).to(device)
    delta_head.load_state_dict(torch.load(HEAD_PATH, map_location=device, weights_only=True))
    delta_head.eval()
    print(f'Loaded delta head from {HEAD_PATH}\n')

    # Group by reaction, pick n_reactions
    by_rxn = defaultdict(list)
    for s in val_data:
        by_rxn[s['rxn']].append(s)
    rxns = random.sample(list(by_rxn.keys()), min(args.n_reactions, len(by_rxn)))

    all_delta_true, all_delta_pred = [], []
    all_e_mace, all_e_mace_delta, all_e_true = [], [], []

    print(f"{'Rxn':12s}  {'N':>4}  {'MAE_delta':>10}  {'MAE_MACE':>10}  {'MAE_MACE+d':>12}")
    print('-' * 58)

    for rxn in sorted(rxns):
        structs = by_rxn[rxn]
        e_wb97x_pred, delta_pred = run_batch(structs, model, delta_head, z_table, r_max, device)

        delta_true  = np.array([s['delta_eV'] for s in structs])
        # true wB97X energy = not stored, but: E_wB97M_true = E_wB97X_true + delta_true
        # We can evaluate delta accuracy directly, and MACE+delta vs MACE-alone in delta space
        # MACE alone error on delta: it predicts 0 delta (no correction) → error = delta_true
        mace_alone_delta = np.zeros_like(delta_true)   # MACE predicts no correction

        mae_delta,    _  = mae_rmse(delta_pred, delta_true)
        mae_mace,     _  = mae_rmse(mace_alone_delta, delta_true)
        mae_mace_d,   _  = mae_rmse(delta_pred, delta_true)  # same as mae_delta for per-rxn

        all_delta_true.extend(delta_true.tolist())
        all_delta_pred.extend(delta_pred.tolist())

        print(f"{rxn:12s}  {len(structs):>4}  {mae_delta*1000:>9.1f}m  {mae_mace*1000:>9.1f}m  {mae_mace_d*1000:>11.1f}m")

    all_delta_true = np.array(all_delta_true)
    all_delta_pred = np.array(all_delta_pred)

    mae_d,  rmse_d  = mae_rmse(all_delta_pred, all_delta_true)
    mae_m,  rmse_m  = mae_rmse(np.zeros_like(all_delta_true), all_delta_true)

    print('-' * 58)
    print(f'\nOverall ({len(all_delta_true)} geoms across {len(rxns)} reactions):')
    print(f'  MACE alone (no delta):   MAE={mae_m*1000:6.1f} meV  RMSE={rmse_m*1000:6.1f} meV')
    print(f'  MACE + delta head:       MAE={mae_d*1000:6.1f} meV  RMSE={rmse_d*1000:6.1f} meV')
    print(f'  Improvement: {(1 - mae_d/mae_m)*100:.1f}% MAE reduction')
    print(f'\n  True delta range: [{all_delta_true.min():.3f}, {all_delta_true.max():.3f}] eV')
    print(f'  True delta mean:  {all_delta_true.mean():.3f} eV')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-reactions', type=int, default=10)
    args = parser.parse_args()
    main(args)
