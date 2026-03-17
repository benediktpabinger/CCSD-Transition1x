"""
CCSD calculations for a range of configurations — designed for SLURM array jobs.
Each job processes a slice of configs (--start-config to --end-config)
and writes its own partial JSON result file.
"""
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from transition1x import Dataloader
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_reaction_data(h5file, reaction_name, split='test'):
    """Extract all configurations for a specific reaction"""
    dataloader = Dataloader(h5file, split)
    configurations = []
    for config in dataloader:
        if config['rxn'] == reaction_name:
            configurations.append(config)
    print(f"Found {len(configurations)} total configurations for {reaction_name}")
    return configurations


def run_ccsd_calculation(atoms, label, basis='cc-pVDZ', directory='.'):
    """Run CCSD energy calculation using ORCA"""
    orca_path = shutil.which('orca')
    if orca_path is None:
        raise RuntimeError("ORCA not found in PATH. Load with: module load ORCA/5.0.4-gompi-2023a")

    profile = OrcaProfile(command=orca_path)
    calc = ORCA(
        profile=profile,
        label=label,
        directory=directory,
        orcasimpleinput=f'CCSD {basis} TightSCF',
        orcablocks='%maxcore 4000\n%pal nprocs 1 end'
    )
    atoms.calc = calc

    try:
        energy = atoms.get_potential_energy()
        return {'energy': energy, 'success': True}
    except Exception as e:
        print(f"  ERROR: {e}")
        # Try to print the ORCA output for diagnostics
        out_file = os.path.join(directory, f"{label}.out")
        if os.path.exists(out_file):
            with open(out_file) as f:
                tail = f.readlines()[-30:]
            print("  Last lines of ORCA output:")
            print(''.join(tail))
        return {'energy': None, 'success': False, 'error': str(e)}


def main(args):
    if not os.path.exists(args.h5file):
        print(f"ERROR: HDF5 file not found: {args.h5file}")
        sys.exit(1)

    # Load all configs for this reaction
    all_configs = get_reaction_data(args.h5file, args.reaction, args.split)

    # Determine slice for this job
    start = args.start_config
    end = args.end_config if args.end_config is not None else len(all_configs)
    end = min(end, len(all_configs))
    configs = all_configs[start:end]

    print(f"This job processes configs {start} to {end-1} ({len(configs)} total)")

    # Output directory (shared across all jobs)
    output_dir = f"ccsd_{args.reaction.replace('+', '_').replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"\nStarting CCSD calculations with {args.basis} basis set...")
    print("=" * 60)

    for local_i, config in enumerate(configs):
        global_i = start + local_i  # index in full 268-config list
        atoms = Atoms(config['atomic_numbers'])
        atoms.set_positions(config['positions'])

        # Each calculation gets its own subdirectory so ORCA files don't collide
        calc_dir = os.path.join(output_dir, f"config_{global_i:04d}")
        os.makedirs(calc_dir, exist_ok=True)

        print(f"\nConfig {global_i} ({local_i+1}/{len(configs)}): {config['formula']}")
        result = run_ccsd_calculation(atoms, 'orca', basis=args.basis, directory=calc_dir)

        entry = {
            'config_index': global_i,
            'rxn': config['rxn'],
            'formula': config['formula'],
            'positions': config['positions'].tolist() if hasattr(config['positions'], 'tolist') else list(config['positions']),
            'atomic_numbers': config['atomic_numbers'].tolist() if hasattr(config['atomic_numbers'], 'tolist') else list(config['atomic_numbers']),
            'ccsd.energy': result['energy'],
            'wB97x.energy': float(config['wB97x_6-31G(d).energy']),
            'wB97x.atomization_energy': float(config['wB97x_6-31G(d).atomization_energy']),
            'success': result['success']
        }

        if not result['success']:
            entry['error'] = result['error']
        else:
            energy_diff = abs(result['energy'] - config['wB97x_6-31G(d).energy'])
            entry['energy_diff'] = energy_diff
            print(f"  CCSD:  {result['energy']:.6f} eV")
            print(f"  wB97x: {config['wB97x_6-31G(d).energy']:.6f} eV")
            print(f"  Diff:  {energy_diff:.6f} eV")

        results.append(entry)

    # Save partial results — filename includes range so jobs don't overwrite each other
    output_file = os.path.join(output_dir, f'results_{start:04d}_{end-1:04d}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*60}")
    print(f"Saved: {output_file}")
    print(f"Successful: {successful}/{len(results)}")


if __name__ == "__main__":
    parser = ArgumentParser(description='CCSD calculations for a config range (SLURM array jobs)')
    parser.add_argument('--h5file', required=True, help='Path to HDF5 file')
    parser.add_argument('--reaction', required=True, help='Reaction name (e.g. rxn1961)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--start-config', type=int, required=True, help='First config index (0-based)')
    parser.add_argument('--end-config', type=int, default=None, help='Last config index (exclusive)')
    parser.add_argument('--basis', default='cc-pVDZ', help='Basis set')

    args = parser.parse_args()
    main(args)
