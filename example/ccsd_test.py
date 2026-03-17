"""
Test script: Recalculate all points of ONE reaction with CCSD
"""
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile
from transition1x import Dataloader
from tqdm import tqdm
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


def get_reaction_names(h5file, split='test', max_reactions=10):
    """Get list of available reaction names"""
    dataloader = Dataloader(h5file, split)
    reactions = set()
    
    for config in dataloader:
        reactions.add(config['rxn'])
        if len(reactions) >= max_reactions:
            break
    
    return sorted(list(reactions))


def get_reaction_data(h5file, reaction_name, split='test'):
    """Extract all configurations for a specific reaction"""
    dataloader = Dataloader(h5file, split)
    configurations = []
    
    print(f"Loading configurations for reaction: {reaction_name}")
    for config in dataloader:
        if config['rxn'] == reaction_name:
            configurations.append(config)
    
    print(f"Found {len(configurations)} configurations for {reaction_name}")
    return configurations


def run_ccsd_calculation(atoms, label, basis='cc-pVDZ'):
    """
    Run CCSD calculation using ORCA
    
    Args:
        atoms: ASE Atoms object
        label: Calculation label
        basis: Basis set
    
    Returns:
        dict with energy and forces
    """
    orca_path = shutil.which('orca')
    if orca_path is None:
        raise RuntimeError("ORCA executable not found in PATH. Run: module load ORCA/5.0.4-gompi-2023a")

    profile = OrcaProfile(command=orca_path)
    calc = ORCA(
        profile=profile,
        label=label,
        orcasimpleinput=f'CCSD {basis} TightSCF',
        orcablocks='%maxcore 2000\n%pal nprocs 1 end'
    )
    
    atoms.calc = calc
    
    try:
        energy = atoms.get_potential_energy()
        # CCSD forces are expensive, skip for this test
        # forces = atoms.get_forces()
        
        return {
            'energy': energy,
            'forces': None,
            'success': True
        }
    except Exception as e:
        print(f"  ERROR in calculation: {e}")
        return {
            'energy': None,
            'forces': None,
            'success': False,
            'error': str(e)
        }


def main(args):
    # Check if h5 file exists
    if not os.path.exists(args.h5file):
        print(f"ERROR: HDF5 file not found: {args.h5file}")
        sys.exit(1)
    
    # Get available reactions if needed
    if args.list_reactions:
        print("Available reactions:")
        reactions = get_reaction_names(args.h5file, args.split)
        for i, rxn in enumerate(reactions, 1):
            print(f"  {i}. {rxn}")
        sys.exit(0)
    
    # Get reaction configurations
    configurations = get_reaction_data(args.h5file, args.reaction, args.split)
    
    if len(configurations) == 0:
        print(f"ERROR: No configurations found for reaction '{args.reaction}'")
        print("Use --list-reactions to see available reactions")
        sys.exit(1)
    
    # Limit number of calculations for testing
    if args.max_configs:
        configurations = configurations[:args.max_configs]
        print(f"Limited to {args.max_configs} configurations for testing")
    
    # Create output directory
    output_dir = f"ccsd_{args.reaction.replace('+', '_').replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}/")
    
    # Results storage
    results = []
    
    # Run CCSD calculations
    print(f"\nStarting CCSD calculations with {args.basis} basis set...")
    print("=" * 60)
    
    for i, config in enumerate(tqdm(configurations, desc="CCSD")):
        # Create atoms object
        atoms = Atoms(config['atomic_numbers'])
        atoms.set_positions(config['positions'])
        
        # Label for this calculation
        label = os.path.join(output_dir, f"config_{i:04d}")
        
        # Run CCSD
        print(f"\nConfiguration {i+1}/{len(configurations)}: {config['formula']}")
        result = run_ccsd_calculation(atoms, label, basis=args.basis)
        
        # Store results
        result_entry = {
            'config_index': i,
            'rxn': config['rxn'],
            'formula': config['formula'],
            'positions': config['positions'].tolist() if hasattr(config['positions'], 'tolist') else list(config['positions']),
            'atomic_numbers': config['atomic_numbers'].tolist() if hasattr(config['atomic_numbers'], 'tolist') else list(config['atomic_numbers']),
            'ccsd.energy': result['energy'],
            'ccsd.forces': result['forces'],
            'wB97x.energy': float(config['wB97x_6-31G(d).energy']),
            'wB97x.atomization_energy': float(config['wB97x_6-31G(d).atomization_energy']),
            'success': result['success']
        }
        
        if not result['success']:
            result_entry['error'] = result['error']
        else:
            energy_diff = abs(result['energy'] - config['wB97x_6-31G(d).energy'])
            result_entry['energy_diff'] = energy_diff
            print(f"  CCSD energy: {result['energy']:.6f} eV")
            print(f"  wB97x energy: {config['wB97x_6-31G(d).energy']:.6f} eV")
            print(f"  Difference: {energy_diff:.6f} eV")
        
        results.append(result_entry)
    
    # Save results
    output_file = os.path.join(output_dir, f'ccsd_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nSummary:")
    print(f"  Reaction: {args.reaction}")
    print(f"  Total calculations: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")
    
    # Energy statistics
    if successful > 0:
        energy_diffs = [r['energy_diff'] for r in results if r['success']]
        
        if energy_diffs:
            print(f"\nEnergy comparison (CCSD vs wB97x):")
            print(f"  Mean absolute difference: {np.mean(energy_diffs):.4f} eV")
            print(f"  Max absolute difference: {np.max(energy_diffs):.4f} eV")
            print(f"  Min absolute difference: {np.min(energy_diffs):.4f} eV")
            print(f"  Std deviation: {np.std(energy_diffs):.4f} eV")


if __name__ == "__main__":
    parser = ArgumentParser(description='Test CCSD calculation for one reaction')
    
    parser.add_argument('--h5file', default='data/Transition1x.h5',
                        help='Path to HDF5 file')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='Data split to use')
    parser.add_argument('--reaction', type=str, default=None,
                        help='Reaction name (e.g., "C2H4+H")')
    parser.add_argument('--list-reactions', action='store_true',
                        help='List available reactions and exit')
    parser.add_argument('--max-configs', type=int, default=None,
                        help='Maximum number of configurations to calculate (for testing)')
    parser.add_argument('--basis', default='cc-pVDZ',
                        help='Basis set (e.g., cc-pVDZ, cc-pVTZ)')
    
    args = parser.parse_args()
    
    # Check if reaction is specified
    if not args.list_reactions and args.reaction is None:
        parser.error("Please specify --reaction or use --list-reactions")
    
    main(args)
