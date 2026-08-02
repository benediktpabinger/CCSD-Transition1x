"""
Fixed-geometry single-point comparison on the RKS-stable reaction subset.

Reaction set (22 reactions)
---------------------------
Chosen because the restricted Kohn-Sham (RKS) solution is the stable SCF
solution for these reactions -- no broken-symmetry (UKS/BS) instability.  The
reference wB97M-V/def2-TZVP description is therefore free of spin-symmetry
breaking pathology, and OMol25-trained models (UMA-S, UMA-M, eSEN), which are
trained on closed-shell RKS data, are expected to be able to reproduce them.
This makes the comparison a fair test of the models rather than a probe of
multireference character.

All 22 are a subset of the original 30-reaction benchmark, so wB97M-V NEB
geometries and wB97X-D3 EnGrad single points already exist for every one.

Methods evaluated on identical geometries
-----------------------------------------
  1. wB97M-V/def2-TZVP   -- reference, read from neb.db (energies + forces)
  2. wB97X-D3/6-31G(d)   -- true DFT, ORCA EnGrad outputs
  3. MACE                -- frozen MACE (wB97X-D3 level)
  4. MACE + delta        -- MACE + fw=2.0 delta head (64x0e)
  5. UMA-S               -- fairchem, task_name='omol'
  6. UMA-M               -- fairchem, task_name='omol'
  7. eSEN                -- fairchem, default task

Geometries: last 10 images of the converged ORCA wB97M-V CI-NEB band, i.e.
exactly the geometries used by eval_benchmark_sp_fw2.py.

Charge/spin: not set explicitly -- fairchem OMol25 defaults (neutral,
closed-shell singlet) are used, identical to uma_neb.py / esen_neb.py, so
these numbers stay comparable with the existing NEB results.

Output: ~/delta_head/eval_sp_rks_stable.json
"""
import gc
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
HEAD_PATH  = f'{HOME}/delta_head/delta_head_fw2.00.pt'
NEB_DIR    = f'{HOME}/orca_neb_results'
ENGRAD_DIR = f'{HOME}/mr_benchmark/orca_engrad'
CKPT_DIR   = f'{HOME}/checkpoints'
OUT_PATH   = f'{HOME}/delta_head/eval_sp_rks_stable.json'

N_IMAGES = 10
EH_BOHR_TO_EV_ANG = 51.42208619

HIDDEN_IRREPS     = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS        = o3.Irreps("64x0e")
NODE_FEATS_OFFSET = 1024

# Tier assignment as specified for the RKS-stable set.
# NOTE: rxn0896 is 'mid' in the original 30-reaction FOD grouping but is
# assigned to 'high' here; this file follows the RKS-stable assignment.
GROUPS = {
    'high': ['rxn7945', 'rxn7937', 'rxn1150', 'rxn0896'],
    'mid':  ['rxn1154', 'rxn4513', 'rxn7955', 'rxn4519',
             'rxn4500', 'rxn2553', 'rxn8829', 'rxn1155'],
    'low':  ['rxn9246', 'rxn4498', 'rxn1061', 'rxn4003', 'rxn4004',
             'rxn4063', 'rxn4114', 'rxn4060', 'rxn1961', 'rxn1962'],
}
RXN_GROUP = {r: g for g, rs in GROUPS.items() for r in rs}
ALL_RXNS  = [r for g in ('high', 'mid', 'low') for r in GROUPS[g]]

FAIRCHEM_MODELS = [
    ('uma_s', f'{CKPT_DIR}/uma-s-1p2.pt',             'omol'),
    ('uma_m', f'{CKPT_DIR}/uma-m-1p1.pt',             'omol'),
    ('esen',  f'{CKPT_DIR}/esen_sm_conserving_all.pt', None),
]

ENERGY_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')


# ---------------------------------------------------------------- reference IO

def parse_engrad(sp_dir):
    """Energy (eV) and forces (eV/Ang) from an ORCA EnGrad run."""
    out_path    = os.path.join(sp_dir, 'sp.out')
    engrad_path = os.path.join(sp_dir, 'sp.engrad')
    if not os.path.exists(out_path):
        return None, None
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
        s = line.strip()
        if 'current gradient' in s.lower():
            in_grad = True
            continue
        if in_grad:
            if s == '#':
                continue
            if s.startswith('#'):
                break
            try:
                grad_values.append(float(s))
            except ValueError:
                continue
    if not grad_values:
        return energy_eV, None
    forces = -np.array(grad_values).reshape(-1, 3) * EH_BOHR_TO_EV_ANG
    return energy_eV, forces


def load_neb_images(rxn):
    """Last N_IMAGES of the converged ORCA wB97M-V CI-NEB band."""
    with ase.db.connect(os.path.join(NEB_DIR, rxn, 'neb.db')) as db:
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


# ------------------------------------------------------------------ MACE/delta

def run_mace_delta(atoms_list, model, delta_head, z_table, r_max, device):
    configs   = [config_from_atoms(a) for a in atoms_list]
    data_list = [AtomicData.from_config(c, z_table=z_table, cutoff=r_max, heads=['Default'])
                 for c in configs]

    batch1 = torch_geometric.Batch.from_data_list(data_list).to(device)
    batch1.positions.requires_grad_(True)
    with torch.enable_grad():
        bd1 = {key: batch1[key] for key in batch1.keys}
        out1 = model(bd1, training=False, compute_force=True,
                     compute_virials=False, compute_stress=False)
        e_mace = out1['energy'].detach().cpu().numpy()
        f_mace = out1['forces'].detach().cpu().numpy()

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


# -------------------------------------------------------------------- metrics

def rel(e):
    e = np.asarray(e, float)
    return (e - e[0]) * 1000.0


def emae(pred_e, ref_e):
    """MAE on image-0-anchored relative profiles (meV). Each method self-anchored."""
    return float(np.abs(rel(pred_e) - rel(ref_e)).mean())


def fmae(pred_list, ref_list):
    """Per-component force MAE (meV/Ang), averaged over images."""
    errs = [np.abs(np.asarray(p, float) - np.asarray(t, float)).mean()
            for p, t in zip(pred_list, ref_list) if p is not None and t is not None]
    return float(np.mean(errs)) * 1000 if errs else None


def cos_sim(pred_list, ref_list):
    """Per-atom force cosine similarity; atoms with |F| < 1e-6 skipped."""
    sims = []
    for fp, fr in zip(pred_list, ref_list):
        if fp is None or fr is None:
            continue
        fp = np.asarray(fp, float); fr = np.asarray(fr, float)
        n_p = np.linalg.norm(fp, axis=1); n_r = np.linalg.norm(fr, axis=1)
        valid = (n_p > 1e-6) & (n_r > 1e-6)
        if not valid.any():
            continue
        d = np.sum(fp[valid] * fr[valid], axis=1)
        sims.extend((d / (n_p[valid] * n_r[valid])).tolist())
    return sims


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


# ------------------------------------------------------------------------ main

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Reactions: {len(ALL_RXNS)} (RKS-stable subset of the original 30)\n')

    # ---- Pass 1: geometries + DFT references -------------------------------
    data = {}
    for rxn in ALL_RXNS:
        try:
            atoms_list, e_wb97m, f_wb97m = load_neb_images(rxn)
        except Exception as e:
            print(f'{rxn}: neb.db error - {e}')
            continue
        e_x, f_x = [], []
        ok = True
        for i in range(N_IMAGES):
            e, f = parse_engrad(os.path.join(ENGRAD_DIR, rxn, f'geom_{i:04d}'))
            if e is None:
                ok = False
                break
            e_x.append(e); f_x.append(f)
        if not ok:
            print(f'{rxn}: missing EnGrad, skipping')
            continue
        data[rxn] = {
            'atoms': atoms_list,
            'e': {'wb97m': np.array(e_wb97m), 'wb97x': np.array(e_x)},
            'f': {'wb97m': f_wb97m,           'wb97x': f_x},
        }
    print(f'Loaded reference data for {len(data)} reactions\n')

    # ---- Pass 2: MACE and MACE+delta ---------------------------------------
    print('=== MACE / MACE+delta ===')
    calc  = MACECalculator(model_paths=MODEL, device=str(device), default_dtype='float32')
    model = calc.models[0]; model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    delta_head = NonLinearReadoutBlock(irreps_in=HIDDEN_IRREPS,
                                       MLP_irreps=MLP_IRREPS, gate=F.silu).to(device)
    delta_head.load_state_dict(torch.load(HEAD_PATH, map_location=device, weights_only=True))
    delta_head.eval()

    for rxn, d in data.items():
        e_m, f_m, e_d, f_d = run_mace_delta(d['atoms'], model, delta_head,
                                            calc.z_table, float(model.r_max), device)
        d['e']['mace'] = e_m;  d['f']['mace'] = list(f_m)
        d['e']['delta'] = e_d; d['f']['delta'] = list(f_d)
        print(f'  {rxn} done')

    del model, calc, delta_head
    gc.collect(); torch.cuda.empty_cache()

    # ---- Pass 3: fairchem models, one at a time (uma-m is 11 GB) -----------
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    for tag, ckpt, task in FAIRCHEM_MODELS:
        if not os.path.exists(ckpt):
            print(f'\n=== {tag}: checkpoint missing ({ckpt}), skipping ===')
            continue
        print(f'\n=== {tag} ({os.path.basename(ckpt)}) ===')
        predict_unit = pretrained_mlip.load_predict_unit(ckpt, device=str(device))
        for rxn, d in data.items():
            es, fs = [], []
            for atoms in d['atoms']:
                a = atoms.copy()
                a.calc = (FAIRChemCalculator(predict_unit, task_name=task) if task
                          else FAIRChemCalculator(predict_unit))
                es.append(a.get_potential_energy())
                fs.append(a.get_forces())
            d['e'][tag] = np.array(es); d['f'][tag] = fs
            print(f'  {rxn} done')
        del predict_unit
        gc.collect(); torch.cuda.empty_cache()

    # ---- Metrics -----------------------------------------------------------
    METHODS = ['mace', 'delta', 'uma_s', 'uma_m', 'esen', 'wb97x']
    REFS    = ['wb97m', 'wb97x']

    rows = []
    for rxn, d in data.items():
        row = {'rxn': rxn, 'group': RXN_GROUP[rxn], 'natoms': len(d['atoms'][0])}
        for ref in REFS:
            for m in METHODS:
                if m == ref or m not in d['e']:
                    continue
                row[f'emae_{m}_vs_{ref}_meV']  = round(emae(d['e'][m], d['e'][ref]), 1)
                fv = fmae(d['f'][m], d['f'][ref])
                row[f'fmae_{m}_vs_{ref}_meVA'] = round(fv, 1) if fv is not None else None
                cs = cos_sim(d['f'][m], d['f'][ref])
                row[f'cos_{m}_vs_{ref}']       = round(float(np.mean(cs)), 4) if cs else None
        for m in METHODS + ['wb97m']:
            if m in d['e']:
                row[f'e_{m}_eV'] = np.asarray(d['e'][m]).tolist()
                row[f'f_{m}_eV_per_ang'] = [
                    (np.asarray(x).tolist() if x is not None else None) for x in d['f'][m]
                ]
        rows.append(row)

    # ---- Summary tables ----------------------------------------------------
    def agg(metric_fmt, subset=None):
        out = {}
        for m in METHODS:
            vals = [r[metric_fmt.format(m=m)] for r in rows
                    if (subset is None or r['group'] == subset)
                    and r.get(metric_fmt.format(m=m)) is not None]
            out[m] = round(float(np.mean(vals)), 1) if vals else None
        return out

    summary = {}
    for ref in REFS:
        summary[f'vs_{ref}'] = {
            'eMAE_meV':       agg('emae_{m}_vs_' + ref + '_meV'),
            'fMAE_meVA':      agg('fmae_{m}_vs_' + ref + '_meVA'),
            'by_tier_eMAE':   {t: agg('emae_{m}_vs_' + ref + '_meV', t)
                               for t in ('high', 'mid', 'low')},
            'by_tier_fMAE':   {t: agg('fmae_{m}_vs_' + ref + '_meVA', t)
                               for t in ('high', 'mid', 'low')},
        }
        cos = {}
        for m in METHODS:
            vals = [r[f'cos_{m}_vs_{ref}'] for r in rows
                    if r.get(f'cos_{m}_vs_{ref}') is not None]
            cos[m] = round(float(np.mean(vals)), 4) if vals else None
        summary[f'vs_{ref}']['cosine'] = cos

    for ref in REFS:
        print(f'\n{"="*72}')
        print(f'SUMMARY vs {ref}  ({len(rows)} reactions, {len(rows)*N_IMAGES} geometries)')
        print(f'{"method":<10}{"eMAE(meV)":>12}{"fMAE(meV/A)":>14}{"cosine":>10}')
        s = summary[f'vs_{ref}']
        for m in METHODS:
            if m == ref:
                continue
            e = s['eMAE_meV'].get(m); f_ = s['fMAE_meVA'].get(m); c = s['cosine'].get(m)
            print(f'{m:<10}{("%.1f"%e) if e is not None else "n/a":>12}'
                  f'{("%.1f"%f_) if f_ is not None else "n/a":>14}'
                  f'{("%.4f"%c) if c is not None else "n/a":>10}')
        for t in ('high', 'mid', 'low'):
            n = sum(1 for r in rows if r['group'] == t)
            parts = []
            for m in METHODS:
                if m == ref:
                    continue
                v = s['by_tier_eMAE'][t].get(m)
                parts.append(f'{m}={v:.0f}' if v is not None else f'{m}=n/a')
            print(f'  eMAE {t:<5} n={n}  ' + '  '.join(parts))

    payload = {
        'description': 'Fixed-geometry SP comparison on the RKS-stable subset',
        'rationale': ('Reactions chosen because the RKS solution is the stable SCF '
                      'solution -- no broken-symmetry instability. The wB97M-V reference '
                      'is therefore free of spin-symmetry-breaking pathology, so '
                      'OMol25-trained models (UMA-S, UMA-M, eSEN) should be able to '
                      'reproduce these reactions. Fair model test, not an MR probe.'),
        'geometries': ('last 10 images of the converged ORCA wB97M-V/def2-TZVP CI-NEB '
                       'band per reaction; identical to eval_benchmark_sp_fw2.py'),
        'charge_spin': ('not set explicitly; fairchem OMol25 defaults (neutral, '
                        'closed-shell singlet), identical to uma_neb.py / esen_neb.py'),
        'metric_definitions': {
            'eMAE': 'MAE on image-0-anchored relative energy profiles, each method '
                    'self-anchored: rel(e)=(e-e[0])*1000; mean over 10 images, then '
                    'macro-averaged over reactions',
            'fMAE': 'mean |F_pred - F_ref| over all force COMPONENTS (natoms x 3) per '
                    'image, averaged over images, then macro-averaged over reactions',
            'cosine': 'per-atom cosine between predicted and reference force 3-vectors; '
                      'atoms with |F| < 1e-6 eV/A skipped',
        },
        'tier_note': ('rxn0896 is "mid" in the original 30-reaction FOD grouping but is '
                      'assigned to "high" in this RKS-stable set'),
        'n_reactions': len(rows),
        'n_images_per_reaction': N_IMAGES,
        'summary': summary,
        'reactions': rows,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
