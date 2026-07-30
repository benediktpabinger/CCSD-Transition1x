"""
Evaluate MACE+delta (fw=2.0) single-point energies on the bare-MACE NEB
geometries and compute forward barriers.

Both MACE and MACE+delta see exactly the same NEB path (found by bare MACE),
so any barrier difference is purely from the delta energy correction — no
geometry error mixed in.

Comparison:
  - bare MACE barrier on MACE-NEB path       (from neb.db energies)
  - MACE+delta fw=2.0 barrier on same path   (NEW — delta as post-hoc SP)
  - NEVPT2 reference (CASSCF-optimised TS)

Output: ~/delta_head/eval_delta_on_mace_neb.json

Usage (on cluster):
    module load Python/3.11.3-GCCcore-12.3.0 Boost/1.82.0-GCC-12.3.0 \
                GSL/2.7-GCC-12.3.0 FlexiBLAS/3.3.1-GCC-12.3.0 HDF5/1.14.0-gompi-2023a
    python3 ~/pipeline/delta/eval_delta_on_mace_neb.py
"""
import json
import os

import sqlite3

import numpy as np
import torch
import torch.nn.functional as F
torch.serialization.add_safe_globals([slice])
from e3nn import o3
from mace.calculators import MACECalculator
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.blocks import NonLinearReadoutBlock
from mace.tools import torch_geometric

HOME         = '/home/energy/s242862'
MODEL        = f'{HOME}/mace_t1x_p10_compiled.model'
HEAD_PATH    = f'{HOME}/delta_head/delta_head_fw2.00.pt'
MACE_NEB_DIR = f'{HOME}/mace_bare_neb_results'
BM_PATH      = f'{HOME}/delta_head/full_benchmark_results.json'
OUT_PATH     = f'{HOME}/delta_head/eval_delta_on_mace_neb.json'

N_IMAGES          = 10
HIDDEN_IRREPS     = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS        = o3.Irreps("64x0e")
NODE_FEATS_OFFSET = 1024

ALL30 = [
    'rxn7949', 'rxn8832', 'rxn1320', 'rxn4113', 'rxn8885',
    'rxn7945', 'rxn7937', 'rxn6196', 'rxn0346', 'rxn1150',
    'rxn9246', 'rxn4498', 'rxn1061', 'rxn4003', 'rxn4004',
    'rxn4063', 'rxn4114', 'rxn4060', 'rxn1961', 'rxn1962',
    'rxn0896', 'rxn1154', 'rxn5690', 'rxn4513', 'rxn7955',
    'rxn4519', 'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155',
]
HIGH10 = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
          'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
MID10  = {'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
          'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155'}


def load_mace_neb(rxn):
    # Use sqlite3 directly — ase.db has buffer-stride errors with some neb.db files
    db_path = os.path.join(MACE_NEB_DIR, rxn, 'neb.db')
    con = sqlite3.connect(db_path)
    all_e = [r[0] for r in con.execute('SELECT energy FROM systems ORDER BY id').fetchall()]
    all_pos = [r[0] for r in con.execute('SELECT positions FROM systems ORDER BY id').fetchall()]
    all_num = [r[0] for r in con.execute('SELECT numbers FROM systems ORDER BY id').fetchall()]
    con.close()
    all_e   = all_e[-N_IMAGES:]
    all_pos = all_pos[-N_IMAGES:]
    all_num = all_num[-N_IMAGES:]
    import ase, pickle
    atoms_list = []
    for pos_blob, num_blob in zip(all_pos, all_num):
        pos = np.frombuffer(pos_blob, dtype=np.float64).reshape(-1, 3)
        num = np.frombuffer(num_blob, dtype=np.int32)
        atoms_list.append(ase.Atoms(numbers=num, positions=pos))
    return atoms_list, np.array(all_e)


def load_nevpt2_from_bm(bm_by_rxn, rxn):
    r = bm_by_rxn.get(rxn, {})
    return r.get('nevpt2_fwd_meV'), r.get('nevpt2_rev_meV'), r.get('nevpt2_reliable', False)


def run_delta_sp(atoms_list, model, delta_head, z_table, r_max, device):
    configs   = [config_from_atoms(a) for a in atoms_list]
    data_list = [AtomicData.from_config(c, z_table=z_table, cutoff=r_max, heads=['Default'])
                 for c in configs]
    batch = torch_geometric.Batch.from_data_list(data_list).to(device)
    with torch.no_grad():
        bd  = {key: batch[key] for key in batch.keys}
        out = model(bd, training=False, compute_force=False,
                    compute_virials=False, compute_stress=False)
        e_mace_raw = out['energy'].cpu().numpy()
        node_feats = out['node_feats'][:, NODE_FEATS_OFFSET:]
    per_atom   = delta_head(node_feats).squeeze(-1)
    per_struct = torch.zeros(batch.num_graphs, device=per_atom.device)
    per_struct.scatter_add_(0, batch.batch, per_atom)
    e_delta_raw = e_mace_raw + per_struct.detach().cpu().numpy()
    return e_mace_raw, e_delta_raw


def barrier(e_arr):
    rel = (e_arr - e_arr[0]) * 1000
    fwd = float(rel.max())
    rev = float(fwd - rel[-1])
    return fwd, rev


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)):    return int(o)
        return super().default(o)


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
    print(f'Loaded delta head fw=2.0 (64x0e) from {HEAD_PATH}\n')

    with open(BM_PATH) as f:
        bm = json.load(f)
    bm_by_rxn = {r['rxn']: r for r in bm['reactions']}

    hdr = f"{'rxn':12s}  {'NEVPT2':>8}  {'MACE':>8}  {'MACE+d':>8}  {'err_MACE':>9}  {'err_delta':>9}  {'improv':>7}  {'grp':>4}"
    print(hdr)
    print('-' * len(hdr))

    results = []
    for rxn in ALL30:
        try:
            atoms_list, e_mace_neb = load_mace_neb(rxn)
        except Exception as e:
            print(f'{rxn}: no MACE NEB — {e}')
            continue

        nevpt2_fwd, nevpt2_rev, reliable = load_nevpt2_from_bm(bm_by_rxn, rxn)
        grp = 'High' if rxn in HIGH10 else ('Mid' if rxn in MID10 else 'Low')

        e_mace_sp, e_delta_sp = run_delta_sp(atoms_list, model, delta_head, z_table, r_max, device)

        mace_fwd,  mace_rev  = barrier(e_mace_sp)
        delta_fwd, delta_rev = barrier(e_delta_sp)

        err_m = mace_fwd  - nevpt2_fwd if nevpt2_fwd is not None else None
        err_d = delta_fwd - nevpt2_fwd if nevpt2_fwd is not None else None
        # improvement: negative means delta is closer to NEVPT2
        improvement = (abs(err_d) - abs(err_m)) if (err_m is not None and err_d is not None) else None

        nev_s = f'{nevpt2_fwd:8.0f}' if nevpt2_fwd is not None else f'{"N/A":>8}'
        em_s  = f'{err_m:+9.0f}'     if err_m       is not None else f'{"N/A":>9}'
        ed_s  = f'{err_d:+9.0f}'     if err_d       is not None else f'{"N/A":>9}'
        imp_s = f'{improvement:+8.0f}' if improvement is not None else f'{"N/A":>8}'
        rel_s = ('*' if reliable else '') + grp
        print(f'{rxn:12s}  {nev_s}  {mace_fwd:8.0f}  {delta_fwd:8.0f}  {em_s}  {ed_s}  {imp_s}  {rel_s}')

        results.append({
            'rxn':              rxn,
            'group':            grp,
            'nevpt2_fwd_meV':   nevpt2_fwd,
            'nevpt2_reliable':  reliable,
            'mace_sp_fwd_meV':  mace_fwd,
            'mace_sp_rev_meV':  mace_rev,
            'delta_sp_fwd_meV': delta_fwd,
            'delta_sp_rev_meV': delta_rev,
            'err_mace_meV':     err_m,
            'err_delta_meV':    err_d,
        })

    print()
    for label, filt in [
        ('All 30',         lambda r: r['nevpt2_fwd_meV'] is not None),
        ('High MR',        lambda r: r['group'] == 'High' and r['nevpt2_fwd_meV'] is not None),
        ('Mid  MR',        lambda r: r['group'] == 'Mid'  and r['nevpt2_fwd_meV'] is not None),
        ('Low  MR',        lambda r: r['group'] == 'Low'  and r['nevpt2_fwd_meV'] is not None),
        ('Reliable (n=8)', lambda r: r['nevpt2_reliable']),
    ]:
        sub = [r for r in results if filt(r)]
        if not sub:
            continue
        mae_m = np.mean([abs(r['err_mace_meV'])  for r in sub])
        mae_d = np.mean([abs(r['err_delta_meV']) for r in sub])
        me_m  = np.mean([r['err_mace_meV']       for r in sub])
        me_d  = np.mean([r['err_delta_meV']       for r in sub])
        print(f'{label:<18}  n={len(sub)}  '
              f'MAE  MACE={mae_m:6.0f}  delta={mae_d:6.0f} meV  |  '
              f'ME   MACE={me_m:+6.0f}  delta={me_d:+6.0f} meV')

    with open(OUT_PATH, 'w') as f:
        json.dump({'reactions': results}, f, indent=2, cls=NumpyEncoder)
    print(f'\nSaved → {OUT_PATH}')


if __name__ == '__main__':
    main()
