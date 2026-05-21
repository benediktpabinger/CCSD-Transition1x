"""
Generate evaluation plots for MACE + delta head.
Saves PNGs to ~/delta_head/plots/.

Usage:
    python plot_delta_eval.py [--n-reactions 20]
"""
import argparse
import random
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
PLOT_DIR  = f'{HOME}/delta_head/plots'

HIDDEN_IRREPS     = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS        = o3.Irreps("16x0e")
NODE_FEATS_OFFSET = 1024


def run_batch(structs, model, delta_head, z_table, r_max, device):
    atoms_list = [Atoms(numbers=s['atomic_numbers'].tolist(), positions=s['positions']) for s in structs]
    configs    = [config_from_atoms(a) for a in atoms_list]
    data_list  = [AtomicData.from_config(c, z_table=z_table, cutoff=r_max, heads=['Default'])
                  for c in configs]
    batch = torch_geometric.Batch.from_data_list(data_list).to(device)
    with torch.no_grad():
        batch_dict = {key: batch[key] for key in batch.keys}
        out        = model(batch_dict, training=False, compute_force=False,
                           compute_virials=False, compute_stress=False)
        node_feats = out['node_feats'][:, NODE_FEATS_OFFSET:]
        per_atom   = delta_head(node_feats).squeeze(-1)
        per_struct = torch.zeros(batch.num_graphs, device=per_atom.device)
        per_struct.scatter_add_(0, batch.batch, per_atom)
    return per_struct.cpu().numpy()


def main(args):
    import os
    os.makedirs(PLOT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_data = torch.load(f'{CACHE_DIR}/val.pt', weights_only=False)

    calc  = MACECalculator(model_paths=MODEL, device=str(device), default_dtype='float32')
    model = calc.models[0]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    z_table = calc.z_table
    r_max   = float(model.r_max)

    delta_head = NonLinearReadoutBlock(irreps_in=HIDDEN_IRREPS, MLP_irreps=MLP_IRREPS, gate=F.silu).to(device)
    delta_head.load_state_dict(torch.load(HEAD_PATH, map_location=device, weights_only=True))
    delta_head.eval()

    by_rxn = defaultdict(list)
    for s in val_data:
        by_rxn[s['rxn']].append(s)
    rxns = random.sample(list(by_rxn.keys()), min(args.n_reactions, len(by_rxn)))

    all_true, all_pred, all_rxn_labels = [], [], []
    rxn_maes = {}

    for rxn in sorted(rxns):
        structs   = by_rxn[rxn]
        pred      = run_batch(structs, model, delta_head, z_table, r_max, device)
        true      = np.array([s['delta_eV'] for s in structs])
        mae       = np.abs(pred - true).mean()
        rxn_maes[rxn] = mae
        all_true.extend(true)
        all_pred.extend(pred)
        all_rxn_labels.extend([rxn] * len(true))

    all_true  = np.array(all_true)
    all_pred  = np.array(all_pred)
    errors_meV = (all_pred - all_true) * 1000

    # --- Plot 1: Scatter predicted vs true ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_true, all_pred, alpha=0.3, s=8, color='steelblue')
    lims = [min(all_true.min(), all_pred.min()) - 0.05,
            max(all_true.max(), all_pred.max()) + 0.05]
    ax.plot(lims, lims, 'k--', lw=1, label='Perfect')
    ax.set_xlabel('True delta (eV)')
    ax.set_ylabel('Predicted delta (eV)')
    ax.set_title(f'MACE + delta head — {len(rxns)} val reactions\nMAE={np.abs(errors_meV).mean():.0f} meV')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/scatter.png', dpi=150)
    plt.close()

    # --- Plot 2: Error distribution ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors_meV, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='k', lw=1)
    ax.axvline(np.mean(errors_meV),  color='red',    lw=1.5, linestyle='--', label=f'mean={np.mean(errors_meV):.0f} meV')
    ax.axvline(np.median(errors_meV), color='orange', lw=1.5, linestyle='--', label=f'median={np.median(errors_meV):.0f} meV')
    ax.set_xlabel('Error (meV)')
    ax.set_ylabel('Count')
    ax.set_title('Delta prediction error distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/error_hist.png', dpi=150)
    plt.close()

    # --- Plot 3: Per-reaction MAE bar chart ---
    sorted_rxns = sorted(rxn_maes, key=rxn_maes.get)
    maes_meV    = [rxn_maes[r] * 1000 for r in sorted_rxns]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(sorted_rxns)), maes_meV, color='steelblue')
    ax.axhline(np.mean(maes_meV), color='red', lw=1.5, linestyle='--', label=f'mean={np.mean(maes_meV):.0f} meV')
    ax.set_xticks(range(len(sorted_rxns)))
    ax.set_xticklabels(sorted_rxns, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MAE (meV)')
    ax.set_title('Per-reaction MAE (sorted)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/per_rxn_mae.png', dpi=150)
    plt.close()

    # --- Plot 4: Delta along reaction path for 4 example reactions ---
    example_rxns = sorted_rxns[:2] + sorted_rxns[-2:]  # best 2 + worst 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, rxn in zip(axes.flat, example_rxns):
        structs = by_rxn[rxn]
        pred    = run_batch(structs, model, delta_head, z_table, r_max, device)
        true    = np.array([s['delta_eV'] for s in structs])
        idxs    = [s['geom_idx'] for s in structs]
        order   = np.argsort(idxs)
        mae     = np.abs(pred - true).mean() * 1000
        ax.plot(np.array(idxs)[order], true[order],  'k-o',  ms=3, label='True')
        ax.plot(np.array(idxs)[order], pred[order],  'r--o', ms=3, label='Pred')
        ax.set_title(f'{rxn}  MAE={mae:.0f} meV')
        ax.set_xlabel('Geometry index')
        ax.set_ylabel('Delta (eV)')
        ax.legend(fontsize=8)
    fig.suptitle('Delta along reaction path (best 2 + worst 2 reactions)')
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/reaction_paths.png', dpi=150)
    plt.close()

    print(f'Plots saved to {PLOT_DIR}/')
    print(f'Overall MAE: {np.abs(errors_meV).mean():.1f} meV over {len(all_true)} geoms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-reactions', type=int, default=20)
    args = parser.parse_args()
    main(args)
