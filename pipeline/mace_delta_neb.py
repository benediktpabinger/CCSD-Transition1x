"""
NEB using MACE + delta head as ASE calculator.

The calculator wraps the frozen MACE backbone + delta correction head,
computing E = E_MACE(wB97X-D3) + delta(node_feats) ≈ E_wB97M-V.
Forces are obtained via autograd through both MACE and the delta head.

Requires sm_90a GPU (H200) — compiled MACE model targets this arch.

Results stored in ~/mace_delta_neb_results/{reaction}/

Usage:
    python mace_delta_neb.py \
        --h5file     ~/data/Transition1x.h5 \
        --reaction   rxn7949 \
        --output     ~/mace_delta_neb_results/rxn7949
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.serialization.add_safe_globals([slice])
from e3nn import o3
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.mep import NEB, NEBTools
from ase.mep.neb import NEBOptimizer
from ase.optimize.bfgs import BFGS
import ase.db
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.blocks import NonLinearReadoutBlock
from mace.tools import torch_geometric

HOME             = '/home/energy/s242862'
MODEL_PATH       = f'{HOME}/mace_t1x_p10_compiled.model'
HEAD_PATH        = f'{HOME}/delta_head/delta_head_fw1.00.pt'
HIDDEN_IRREPS    = o3.Irreps("1024x0e + 1024x1o + 1024x2e + 1024x3o")
MLP_IRREPS       = o3.Irreps("64x0e")
NODE_FEATS_OFFSET = 1024


class MACEDeltaCalculator(Calculator):
    """ASE Calculator wrapping frozen MACE + delta correction head."""

    implemented_properties = ['energy', 'forces']

    def __init__(self, model, delta_head, z_table, r_max, device='cuda', **kwargs):
        super().__init__(**kwargs)
        self.model      = model
        self.delta_head = delta_head
        self.z_table    = z_table
        self.r_max      = r_max
        self.device     = device

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        config    = config_from_atoms(self.atoms)
        data      = AtomicData.from_config(config, z_table=self.z_table,
                                           cutoff=self.r_max, heads=['Default'])
        batch     = torch_geometric.Batch.from_data_list([data]).to(self.device)
        batch.positions.requires_grad_(True)

        bd = {key: batch[key] for key in batch.keys}

        with torch.enable_grad():
            # MACE energy + forces
            out_mace = self.model(bd, training=False, compute_force=True,
                                  compute_virials=False, compute_stress=False)
            e_mace = out_mace['energy']
            f_mace = out_mace['forces']

            if self.delta_head is None:
                energy = e_mace.item()
                forces = f_mace.detach().cpu().numpy()
            else:
                # Delta correction via autograd
                batch2 = torch_geometric.Batch.from_data_list([data]).to(self.device)
                batch2.positions.requires_grad_(True)
                bd2 = {key: batch2[key] for key in batch2.keys}
                out2 = self.model(bd2, training=False, compute_force=False,
                                  compute_virials=False, compute_stress=False)
                node_feats  = out2['node_feats'][:, NODE_FEATS_OFFSET:]
                per_atom    = self.delta_head(node_feats).squeeze(-1)
                per_struct  = per_atom.sum()
                delta_forces = -torch.autograd.grad(per_struct, bd2['positions'])[0]

                energy = (e_mace + per_struct).item()
                forces = (f_mace + delta_forces).detach().cpu().numpy()

        self.results = {
            'energy': energy,
            'forces': forces,
        }


def load_models(device, head_path=None, no_delta=False):
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()

    if no_delta:
        delta_head = None
    else:
        delta_head = NonLinearReadoutBlock(HIDDEN_IRREPS, MLP_IRREPS, gate=F.silu).to(device)
        delta_head.load_state_dict(torch.load(head_path or HEAD_PATH, map_location=device))
        delta_head.eval()

    from mace.tools import AtomicNumberTable
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])

    r_max = float(model.r_max) if hasattr(model, 'r_max') else 6.0

    return model, delta_head, z_table, r_max


def load_wB97x_images(h5file, reaction, split):
    with h5py.File(h5file, 'r') as f:
        split_group = f[split]
        for formula in split_group:
            if reaction in split_group[formula]:
                rxn_group = split_group[formula][reaction]
                positions     = rxn_group['positions'][:]
                atomic_numbers = rxn_group['atomic_numbers'][:]
                total = positions.shape[0]
                final_positions = [positions[0]] + list(positions[-8:]) + [positions[9]]
                images = [Atoms(numbers=atomic_numbers, positions=pos)
                          for pos in final_positions]
                print(f'Loaded 10 wB97x images from H5 ({total} total configs)')
                return images
    raise ValueError(f"Reaction '{reaction}' not found in split '{split}' of {h5file}")


def plot_mep(images, output):
    neb_tools = NEBTools(images)
    fit = neb_tools.get_fit()
    fig, ax = plt.subplots()
    ax.plot(fit.fit_path, fit.fit_energies, 'b-o', markersize=4,
            label=f'MACE+delta barrier: {max(fit.fit_energies):.3f} eV')
    ax.set_ylabel('Energy [eV]')
    ax.set_xlabel('Reaction Coordinate [Å]')
    ax.legend()
    fig.savefig(os.path.join(output, 'mep.png'))
    plt.close(fig)


class DBWriter:
    def __init__(self, db_path, images):
        self.images  = images
        self.db_path = db_path

    def write(self):
        with ase.db.connect(self.db_path) as db:
            for atoms in self.images:
                if atoms.calc and atoms.calc.results:
                    db.write(atoms, data=atoms.calc.results)


def main(args):
    if not os.path.exists(args.h5file):
        print(f'ERROR: HDF5 file not found: {args.h5file}')
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    # Remove stale neb.db so ASE doesn't append to a previous run's database
    stale_db = os.path.join(args.output, 'neb.db')
    if os.path.exists(stale_db):
        os.remove(stale_db)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading bare MACE ...' if args.no_delta else 'Loading MACE + delta head ...')
    model, delta_head, z_table, r_max = load_models(device, head_path=args.head, no_delta=args.no_delta)

    print(f'Loading wB97x NEB images for {args.reaction} ...')
    images = load_wB97x_images(args.h5file, args.reaction, args.split)

    print('Attaching MACE+delta calculator to all images ...')
    for atoms in images:
        atoms.calc = MACEDeltaCalculator(model, delta_head, z_table, r_max, device)

    r_xyz = os.path.join(args.output, 'reactant.xyz')
    p_xyz = os.path.join(args.output, 'product.xyz')

    if args.skip_relax and os.path.exists(r_xyz) and os.path.exists(p_xyz):
        print('Skipping endpoint relaxation ...')
        images[0].set_positions(read(r_xyz).get_positions())
        images[-1].set_positions(read(p_xyz).get_positions())
    else:
        print('Relaxing reactant ...')
        BFGS(images[0], logfile=os.path.join(args.output, 'relax_r.log')).run(fmax=0.05)
        write(r_xyz, images[0])
        print('Relaxing product ...')
        BFGS(images[-1], logfile=os.path.join(args.output, 'relax_p.log')).run(fmax=0.05)
        write(p_xyz, images[-1])

    print('Running NEB (bare MACE) ...' if args.no_delta else 'Running NEB (MACE+delta) ...')
    neb = NEB(images, climb=False, parallel=False, method='improvedtangent')
    neb_tools = NEBTools(images)
    if args.optimizer == 'bfgs':
        relax_neb = BFGS(neb, logfile=os.path.join(args.output, 'neb.log'))
    else:
        relax_neb = NEBOptimizer(neb, logfile=os.path.join(args.output, 'neb.log'))

    db_path   = os.path.join(args.output, 'neb.db')
    db_writer = DBWriter(db_path, images)
    fmaxs     = []

    relax_neb.attach(db_writer.write)
    relax_neb.attach(lambda: fmaxs.append(neb_tools.get_fmax()))
    relax_neb.run(fmax=args.neb_fmax, steps=args.steps)

    print('Running CI-NEB ...')
    neb.climb = True
    converged = relax_neb.run(fmax=args.cineb_fmax, steps=args.steps)

    if converged:
        open(os.path.join(args.output, 'converged'), 'w').close()
        print('CI-NEB converged!')
    else:
        print('WARNING: CI-NEB did not converge within step limit')

    json.dump(fmaxs, open(os.path.join(args.output, 'fmaxs.json'), 'w'))

    ts_out = max(images, key=lambda x: x.get_potential_energy())
    write(os.path.join(args.output, 'transition_state.xyz'), ts_out)
    write(r_xyz, images[0])
    write(p_xyz, images[-1])

    plot_mep(images, args.output)

    print(f'\nDone. Results in {args.output}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5file',      default=f'{HOME}/data/Transition1x.h5')
    parser.add_argument('--reaction',    required=True)
    parser.add_argument('--split',       default='test')
    parser.add_argument('--output',      required=True)
    parser.add_argument('--neb-fmax',    type=float, default=0.15)
    parser.add_argument('--cineb-fmax',  type=float, default=0.05)
    parser.add_argument('--steps',       type=int,   default=500)
    parser.add_argument('--skip-relax',  action='store_true')
    parser.add_argument('--optimizer',   default='ode', choices=['ode', 'bfgs'])
    parser.add_argument('--head',        default=None, help='path to delta head checkpoint (default: delta_head_fw1.00.pt)')
    parser.add_argument('--no-delta',    action='store_true', help='use bare MACE only, skip delta correction head entirely')
    parser.add_argument('--device',      default=None, help='cuda or cpu (default: auto-detect)')
    args = parser.parse_args()
    main(args)
