"""
Evaluate energies + forces on the 30 benchmark reactions, last 10 NEB images.

Four methods compared on every geometry:
  1. wB97X-D3/6-31G(d) true  — ORCA EnGrad outputs (~/mr_benchmark/orca_engrad/)
  2. wB97M-V/def2-TZVP true  — neb.db (same geometries)
  3. MACE                     — wB97X-D3 predicted energy + forces
  4. MACE + delta             — corrected energy + forces (autograd through head)

Metrics vs wB97M-V ground truth:
  - Relative energy MAE (meV), referenced to first image per reaction
  - Force MAE (meV/Ang), absolute

Per-image raw data saved to JSON for future analysis.

Output: ~/delta_head/eval_benchmark_sp.json

Usage:
    python eval_benchmark_sp.py
"""
import json
import os
import re

import ase.db
import numpy as np
import torch
import torch.nn.functional as F
torch.serialization.add_safe_globals([slice])  # must precede e3nn import
from e3nn import o3
from mace.calculators import MACECalculator
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.blocks import NonLinearReadoutBlock
from mace.tools import torch_geometric

HOME       = '/home/energy/s242862'
MODEL      = f'{HOME}/mace_t1x_p10_compiled.model'
HEAD_PATH  = f'{HOME}/delta_head/delta_head.pt'
NEB_DIR    = f'{HOME}/orca_neb_results'
ENGRAD_DIR = f'{HOME}/mr_benchmark/orca_engrad'
OUT_PATH   = f'{HOME}/delta_head/eval_benchmark_sp.json'

N_IMAGES = 10
EH_BOHR_TO_EV_ANG = 51.42208619

HIDDEN_IRREPS     = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS        = o3.Irreps("16x0e")
NODE_FEATS_OFFSET = 1024

ALL30 = [
    'rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885',
    'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150',
    'rxn9246', 'rxn4498', 'rxn1061', 'rxn4003', 'rxn4004',
    'rxn4063', 'rxn4114', 'rxn4060', 'rxn1961', 'rxn1962',
    'rxn0896', 'rxn1154', 'rxn5690', 'rxn4513', 'rxn7955',
    'rxn4519', 'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155',
]

ENERGY_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')


def parse_engrad(sp_dir):
    """Parse energy (eV) and forces (eV/Ang) from ORCA EnGrad output."""
    out_path    = os.path.join(sp_dir, 'sp.out')
    engrad_path = os.path.join(sp_dir, 'sp.engrad')

    with open(out_path) as f:
        content = f.read()
    m = ENERGY_RE.findall(content)
    if not m:
        return None, None
    energy_eV = float(m[-1]) * 27.2114

    if not os.path.exists(engrad_path):
        return energy_eV, None

    with open(engrad_path) as f:
        lines = f.readlines()

    grad_values, in_grad = [], False
    for line in lines:
        stripped = line.strip()
        if 'current gradient' in stripped.lower():
            in_grad = True
            continue
        if in_grad:
            # bare '#' lines are section separators — skip them
            if stripped == '#':
                continue
            # '#' with content = next section header — stop
            if stripped.startswith('#'):
                break
            try:
                grad_values.append(float(stripped))
            except ValueError:
                continue

    if not grad_values:
        return energy_eV, None

    forces = -np.array(grad_values).reshape(-1, 3) * EH_BOHR_TO_EV_ANG
    return energy_eV, forces


def load_neb_images(rxn):
    """Load last N_IMAGES from neb.db. Returns (atoms_list, e_wb97m, f_wb97m)."""
    db_path = os.path.join(NEB_DIR, rxn, 'neb.db')
    with ase.db.connect(db_path) as db:
        rows = list(db.select())
    rows = rows[-N_IMAGES:]
    atoms_list, energies, forces = [], [], []
    for row in rows:
        atoms = row.toatoms()
        atoms_list.append(atoms)
        energies.append(atoms.get_potential_energy())
        try:
            forces.append(atoms.get_forces())
        except Exception:
            forces.append(None)
    return atoms_list, np.array(energies), forces


def run_mace_delta(atoms_list, model, delta_head, z_table, r_max, device):
    """Two-pass: pass 1 = MACE energy+forces, pass 2 = delta energy+forces.
    Returns (e_mace, f_mace_list, e_delta, f_delta_list) all numpy."""
    configs   = [config_from_atoms(a) for a in atoms_list]
    data_list = [AtomicData.from_config(c, z_table=z_table, cutoff=r_max, heads=['Default'])
                 for c in configs]

    # Pass 1: MACE energy + forces (positions need grad for internal force autograd)
    batch1 = torch_geometric.Batch.from_data_list(data_list).to(device)
    batch1.positions.requires_grad_(True)
    with torch.enable_grad():
        bd1 = {key: batch1[key] for key in batch1.keys}
        out1 = model(bd1, training=False, compute_force=True,
                     compute_virials=False, compute_stress=False)
        e_mace = out1['energy'].detach().cpu().numpy()
        f_mace = out1['forces'].detach().cpu().numpy()

    # Pass 2: delta energy + forces via autograd through head
    batch2 = torch_geometric.Batch.from_data_list(data_list).to(device)
    batch2.positions.requires_grad_(True)
    bd2 = {key: batch2[key] for key in batch2.keys}
    out2 = model(bd2, training=False, compute_force=False,
                 compute_virials=False, compute_stress=False)

    node_feats = out2['node_feats'][:, NODE_FEATS_OFFSET:]
    per_atom   = delta_head(node_feats).squeeze(-1)
    per_struct = torch.zeros(batch2.num_graphs, device=per_atom.device)
    per_struct.scatter_add_(0, batch2.batch, per_atom)

    delta_f_all = -torch.autograd.grad(per_struct.sum(), bd2['positions'])[0].cpu().numpy()
    e_delta = e_mace + per_struct.detach().cpu().numpy()
    f_delta = f_mace + delta_f_all

    counts = [(batch2.batch == i).sum().item() for i in range(batch2.num_graphs)]
    splits = np.cumsum(counts[:-1])
    return e_mace, np.split(f_mace, splits), e_delta, np.split(f_delta, splits)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


def fmae(pred_list, true_list):
    errs = [np.abs(np.array(p) - np.array(t)).mean()
            for p, t in zip(pred_list, true_list) if t is not None and p is not None]
    return float(np.mean(errs)) * 1000 if errs else None


def main():
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
    print('Loaded MACE + delta head\n')

    hdr = (f"{'Rxn':12s}  {'eMAE_mace':>10}  {'eMAE_delta':>10}  {'eMAE_wb97x':>10}  "
           f"{'fMAE_mace':>10}  {'fMAE_delta':>10}  {'fMAE_wb97x':>10}")
    print(hdr); print('-' * len(hdr))

    rows = []
    all_e_wb97m, all_e_wb97x, all_e_mace, all_e_delta = [], [], [], []
    all_fe_mace, all_fe_delta, all_fe_wb97x = [], [], []

    for rxn in ALL30:
        try:
            atoms_list, e_wb97m, f_wb97m = load_neb_images(rxn)
        except Exception as e:
            print(f'{rxn}: neb.db error - {e}'); continue

        e_wb97x_list, f_wb97x_list = [], []
        for i in range(N_IMAGES):
            e, f = parse_engrad(os.path.join(ENGRAD_DIR, rxn, f'geom_{i:04d}'))
            e_wb97x_list.append(e); f_wb97x_list.append(f)

        if any(e is None for e in e_wb97x_list):
            print(f'{rxn}: missing EnGrad outputs, skipping'); continue
        e_wb97x = np.array(e_wb97x_list)

        e_mace, f_mace_list, e_delta, f_delta_list = run_mace_delta(
            atoms_list, model, delta_head, z_table, r_max, device
        )

        # Relative energies in meV
        def rel(e): return (np.array(e) - np.array(e)[0]) * 1000
        e_wb97m_r = rel(e_wb97m); e_wb97x_r = rel(e_wb97x)
        e_mace_r  = rel(e_mace);  e_delta_r = rel(e_delta)

        emae_mace  = float(np.abs(e_mace_r  - e_wb97m_r).mean())
        emae_delta = float(np.abs(e_delta_r - e_wb97m_r).mean())
        emae_wb97x = float(np.abs(e_wb97x_r - e_wb97m_r).mean())

        fm_mace  = fmae(f_mace_list,  f_wb97m)
        fm_delta = fmae(f_delta_list, f_wb97m)
        fm_wb97x = fmae(f_wb97x_list, f_wb97m)

        fmt = lambda v: f'{v:.1f}' if v is not None else 'N/A'
        print(f'{rxn:12s}  {emae_mace:>9.1f}m  {emae_delta:>9.1f}m  {emae_wb97x:>9.1f}m  '
              f'{fmt(fm_mace):>10}m  {fmt(fm_delta):>10}m  {fmt(fm_wb97x):>10}m')

        # Accumulate overall
        all_e_wb97m.extend(e_wb97m_r); all_e_wb97x.extend(e_wb97x_r)
        all_e_mace.extend(e_mace_r);   all_e_delta.extend(e_delta_r)
        for p, t in zip(f_mace_list,  f_wb97m):
            if t is not None: all_fe_mace.extend(np.abs(np.array(p)-t).flatten())
        for p, t in zip(f_delta_list, f_wb97m):
            if t is not None: all_fe_delta.extend(np.abs(np.array(p)-t).flatten())
        for p, t in zip(f_wb97x_list, f_wb97m):
            if t is not None and p is not None:
                all_fe_wb97x.extend(np.abs(np.array(p)-t).flatten())

        rows.append({
            'rxn': rxn,
            # Summary metrics
            'emae_mace_meV':   round(emae_mace,  1),
            'emae_delta_meV':  round(emae_delta, 1),
            'emae_wb97x_meV':  round(emae_wb97x, 1),
            'fmae_mace_meVA':  round(fm_mace,  1) if fm_mace  is not None else None,
            'fmae_delta_meVA': round(fm_delta, 1) if fm_delta is not None else None,
            'fmae_wb97x_meVA': round(fm_wb97x, 1) if fm_wb97x is not None else None,
            # Per-image raw data (eV for energies, eV/Ang for forces)
            'e_wb97m_eV':   e_wb97m.tolist(),
            'e_wb97x_eV':   e_wb97x.tolist(),
            'e_mace_eV':    e_mace.tolist(),
            'e_delta_eV':   e_delta.tolist(),
            'f_wb97m_eV_per_ang':  [f.tolist() if f is not None else None for f in f_wb97m],
            'f_wb97x_eV_per_ang':  [f.tolist() if f is not None else None for f in f_wb97x_list],
            'f_mace_eV_per_ang':   [f.tolist() for f in f_mace_list],
            'f_delta_eV_per_ang':  [f.tolist() for f in f_delta_list],
        })

    ref = np.array(all_e_wb97m)
    print(f'\n{"=" * 80}')
    print(f'Overall ({len(rows)} reactions, {len(ref)} geoms):')
    print(f'  Relative energy MAE vs wB97M-V (meV):')
    print(f'    wB97X-D3 true:   {np.abs(np.array(all_e_wb97x)-ref).mean():.1f}')
    print(f'    MACE:            {np.abs(np.array(all_e_mace) -ref).mean():.1f}')
    print(f'    MACE + delta:    {np.abs(np.array(all_e_delta)-ref).mean():.1f}')
    print(f'  Force MAE vs wB97M-V (meV/Ang):')
    print(f'    wB97X-D3 true:   {np.mean(all_fe_wb97x)*1000:.1f}')
    print(f'    MACE:            {np.mean(all_fe_mace) *1000:.1f}')
    print(f'    MACE + delta:    {np.mean(all_fe_delta)*1000:.1f}')

    with open(OUT_PATH, 'w') as f:
        json.dump({'n_reactions': len(rows), 'reactions': rows}, f,
                  indent=2, cls=NumpyEncoder)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
