"""
Check whether MACEDeltaCalculator's analytic (autograd) forces are
conservative — i.e. whether F = -dE/dx matches a finite-difference estimate
of the same energy function.

If the analytic force disagrees with finite differences, the NEB optimizer
is being fed a non-conservative force field, which would explain why
MACE+delta NEB produces worse geometries than bare MACE despite having
*lower* single-point force MAE at fixed geometries (line search / CG-style
optimizers assume forces are the true energy gradient).

Tests both bare MACE and MACE+delta (fw=2.0) calculators on a handful of
real geometries (NEB TS images already on disk), for direct comparison.
"""
import numpy as np
import torch
import torch.nn.functional as F
torch.serialization.add_safe_globals([slice])
from e3nn import o3
from ase.io import read

from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.blocks import NonLinearReadoutBlock
from mace.tools import torch_geometric, AtomicNumberTable

HOME       = '/home/energy/s242862'
MODEL_PATH = f'{HOME}/mace_t1x_p10_compiled.model'
HEAD_PATH  = f'{HOME}/delta_head/delta_head_fw2.00.pt'
HIDDEN_IRREPS    = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS       = o3.Irreps("64x0e")
NODE_FEATS_OFFSET = 1024

TEST_GEOMS = [
    f'{HOME}/orca_neb_results/rxn7949/transition_state.xyz',
    f'{HOME}/orca_neb_results/rxn8885/transition_state.xyz',
    f'{HOME}/orca_neb_results/rxn0346/transition_state.xyz',
]

EPS_LIST = [0.005, 0.01, 0.02]  # Angstrom, finite-difference steps to test convergence


def load_models(device):
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    delta_head = NonLinearReadoutBlock(HIDDEN_IRREPS, MLP_IRREPS, gate=F.silu).to(device)
    delta_head.load_state_dict(torch.load(HEAD_PATH, map_location=device))
    delta_head.eval()
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = float(model.r_max) if hasattr(model, 'r_max') else 6.0
    return model, delta_head, z_table, r_max


def energy_and_force(atoms, model, delta_head, z_table, r_max, device, use_delta):
    """Returns (total_energy: float, analytic_forces: np.ndarray [N,3])."""
    config = config_from_atoms(atoms)
    data   = AtomicData.from_config(config, z_table=z_table, cutoff=r_max, heads=['Default'])
    batch  = torch_geometric.Batch.from_data_list([data]).to(device)
    batch.positions.requires_grad_(True)
    bd = {key: batch[key] for key in batch.keys}

    with torch.enable_grad():
        out_mace = model(bd, training=False, compute_force=True,
                          compute_virials=False, compute_stress=False)
        e_mace = out_mace['energy']
        f_mace = out_mace['forces']

        if not use_delta:
            return e_mace.item(), f_mace.detach().cpu().numpy()

        batch2 = torch_geometric.Batch.from_data_list([data]).to(device)
        batch2.positions.requires_grad_(True)
        bd2 = {key: batch2[key] for key in batch2.keys}
        out2 = model(bd2, training=False, compute_force=False,
                      compute_virials=False, compute_stress=False)
        node_feats  = out2['node_feats'][:, NODE_FEATS_OFFSET:]
        per_atom    = delta_head(node_feats).squeeze(-1)
        per_struct  = per_atom.sum()
        delta_forces = -torch.autograd.grad(per_struct, bd2['positions'])[0]

        energy = (e_mace + per_struct).item()
        forces = (f_mace + delta_forces).detach().cpu().numpy()
        return energy, forces


def energy_only(atoms, model, delta_head, z_table, r_max, device, use_delta):
    """Returns total_energy: float (no grad needed, used for finite differences)."""
    config = config_from_atoms(atoms)
    data   = AtomicData.from_config(config, z_table=z_table, cutoff=r_max, heads=['Default'])
    batch  = torch_geometric.Batch.from_data_list([data]).to(device)
    bd = {key: batch[key] for key in batch.keys}
    with torch.no_grad():
        out = model(bd, training=False, compute_force=False,
                     compute_virials=False, compute_stress=False)
        e_mace = out['energy'].item()
        if not use_delta:
            return e_mace
        node_feats = out['node_feats'][:, NODE_FEATS_OFFSET:]
        per_atom   = delta_head(node_feats).squeeze(-1)
        return e_mace + per_atom.sum().item()


def finite_diff_forces(atoms, model, delta_head, z_table, r_max, device, use_delta, eps):
    n = len(atoms)
    fd_forces = np.zeros((n, 3))
    base_pos = atoms.get_positions().copy()
    for i in range(n):
        for d in range(3):
            pos_plus = base_pos.copy(); pos_plus[i, d] += eps
            pos_minus = base_pos.copy(); pos_minus[i, d] -= eps
            atoms.set_positions(pos_plus)
            e_plus = energy_only(atoms, model, delta_head, z_table, r_max, device, use_delta)
            atoms.set_positions(pos_minus)
            e_minus = energy_only(atoms, model, delta_head, z_table, r_max, device, use_delta)
            fd_forces[i, d] = -(e_plus - e_minus) / (2 * eps)
    atoms.set_positions(base_pos)
    return fd_forces


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model, delta_head, z_table, r_max = load_models(device)

    for geom_path in TEST_GEOMS:
        atoms = read(geom_path)
        rxn = geom_path.split('/')[-2]
        print(f'\n{"="*70}\n{rxn} ({len(atoms)} atoms)\n{"="*70}')

        for use_delta, label in [(False, 'bare MACE'), (True, 'MACE+delta fw2.0')]:
            e, f_analytic = energy_and_force(atoms, model, delta_head, z_table, r_max, device, use_delta)
            print(f'  [{label}] E={e:.4f} eV  |F_analytic| mean={np.mean(np.abs(f_analytic)):.4f} eV/A')

            for eps in EPS_LIST:
                f_fd = finite_diff_forces(atoms, model, delta_head, z_table, r_max, device, use_delta, eps)
                diff = f_analytic - f_fd
                max_abs_diff  = np.max(np.abs(diff))
                mean_abs_diff = np.mean(np.abs(diff))
                rel_norm      = np.linalg.norm(diff) / (np.linalg.norm(f_fd) + 1e-12)
                print(f'    eps={eps:.3f} A: |F_fd| mean={np.mean(np.abs(f_fd)):.4f}  '
                      f'max|diff|={max_abs_diff:.5f}  mean|diff|={mean_abs_diff:.5f}  rel_norm={rel_norm:.4f}')


if __name__ == '__main__':
    main()
