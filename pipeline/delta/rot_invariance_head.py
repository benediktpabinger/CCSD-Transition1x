"""
Rotation-invariance test for the delta correction head: old (v2) vs. fixed.

For each benchmark reaction, take the ORCA transition-state geometry, apply
K random proper rotations, and evaluate the predicted correction Delta with
both heads. A rotation-invariant head must return the same Delta for every
orientation (up to float32 noise). MACE's own energy is evaluated as a
control — it is invariant by construction.

Reports, per head: the spread of Delta over the K orientations
(max - min, and standard deviation) for each reaction, plus aggregates.

Output: ~/delta_head/rotation_invariance.json
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
torch.serialization.add_safe_globals([slice])
from ase.io import read
from e3nn import o3
from mace.calculators import MACECalculator
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.blocks import NonLinearReadoutBlock
from mace.tools import torch_geometric

HOME     = '/home/energy/s242862'
MODEL    = f'{HOME}/mace_t1x_p10_compiled.model'
NEB_DIR  = f'{HOME}/orca_neb_results'
OUT_PATH = f'{HOME}/delta_head/rotation_invariance.json'
K_ROT    = 24
SEED     = 0

HEADS = {
    'v2_old': dict(
        path=f'{HOME}/delta_head/delta_head_fw2.00.pt',
        irreps=o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o"),
        offset=1024,
    ),
    'fixed': dict(
        path=f'{HOME}/delta_head/delta_head_fixed_fw2.00.pt',
        irreps=o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o + 1024x0e"),
        offset=0,
    ),
}
MLP_IRREPS = o3.Irreps("64x0e")

ALL30 = [
    'rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885',
    'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150',
    'rxn9246', 'rxn4498', 'rxn1061', 'rxn4003', 'rxn4004',
    'rxn4063', 'rxn4114', 'rxn4060', 'rxn1961', 'rxn1962',
    'rxn0896', 'rxn1154', 'rxn5690', 'rxn4513', 'rxn7955',
    'rxn4519', 'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155',
]


def random_rotations(k, seed):
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(k):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        mats.append(q)
    return mats


def load_heads(device):
    heads = {}
    for name, spec in HEADS.items():
        h = NonLinearReadoutBlock(irreps_in=spec['irreps'], MLP_irreps=MLP_IRREPS, gate=F.silu).to(device)
        h.load_state_dict(torch.load(spec['path'], map_location=device, weights_only=True))
        h.eval()
        heads[name] = (h, spec['offset'])
    return heads


@torch.no_grad()
def evaluate(atoms, model, heads, z_table, r_max, device):
    cfg   = config_from_atoms(atoms)
    data  = AtomicData.from_config(cfg, z_table=z_table, cutoff=r_max, heads=['Default'])
    batch = torch_geometric.Batch.from_data_list([data]).to(device)
    bd    = {key: batch[key] for key in batch.keys}
    out   = model(bd, training=False, compute_force=False,
                  compute_virials=False, compute_stress=False)
    res = {'mace_energy': float(out['energy'].item())}
    feats = out['node_feats']
    for name, (h, off) in heads.items():
        res[name] = float(h(feats[:, off:]).squeeze(-1).sum().item())
    return res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    calc  = MACECalculator(model_paths=MODEL, device=str(device), default_dtype='float32')
    model = calc.models[0]
    model.eval()
    z_table = calc.z_table
    r_max   = float(model.r_max)
    heads   = load_heads(device)
    rots    = random_rotations(K_ROT, SEED)

    keys = ['mace_energy'] + list(HEADS.keys())
    rows = []
    hdr = f"{'rxn':10s}" + ''.join(f"  {k+'_spread_meV':>22}" for k in keys)
    print(hdr); print('-' * len(hdr))

    for rxn in ALL30:
        ts_path = os.path.join(NEB_DIR, rxn, 'transition_state.xyz')
        if not os.path.exists(ts_path):
            print(f'{rxn}: no transition_state.xyz, skipping'); continue
        atoms0 = read(ts_path)
        pos0   = atoms0.get_positions()
        com    = pos0.mean(axis=0)

        vals = {k: [] for k in keys}
        for R in rots:
            a = atoms0.copy()
            a.set_positions((pos0 - com) @ R.T + com)
            res = evaluate(a, model, heads, z_table, r_max, device)
            for k in keys:
                vals[k].append(res[k])

        row = {'rxn': rxn, 'n_atoms': len(atoms0)}
        for k in keys:
            v = np.array(vals[k])
            row[f'{k}_mean_eV']    = float(v.mean())
            row[f'{k}_spread_meV'] = float((v.max() - v.min()) * 1000)
            row[f'{k}_std_meV']    = float(v.std() * 1000)
        rows.append(row)
        print(f"{rxn:10s}" + ''.join(f"  {row[k+'_spread_meV']:22.3f}" for k in keys), flush=True)

    summary = {}
    for k in keys:
        sp = np.array([r[f'{k}_spread_meV'] for r in rows])
        sd = np.array([r[f'{k}_std_meV'] for r in rows])
        summary[k] = {
            'mean_spread_meV':   float(sp.mean()),
            'median_spread_meV': float(np.median(sp)),
            'max_spread_meV':    float(sp.max()),
            'mean_std_meV':      float(sd.mean()),
        }
    print('\nSummary (over reactions):')
    for k, s in summary.items():
        print(f"  {k:12s} mean spread {s['mean_spread_meV']:9.3f} meV | median {s['median_spread_meV']:9.3f} | "
              f"max {s['max_spread_meV']:9.3f} | mean std {s['mean_std_meV']:9.3f}")

    payload = {
        'description': ('Spread (max-min) and std of the predicted correction Delta over K random '
                        'proper rotations of each ORCA transition-state geometry. mace_energy is '
                        'the invariant control.'),
        'k_rotations': K_ROT, 'seed': SEED,
        'heads': {n: {'path': s['path'], 'irreps': str(s['irreps']), 'node_feats_offset': s['offset']}
                  for n, s in HEADS.items()},
        'summary': summary,
        'reactions': rows,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
