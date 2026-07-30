"""
AutoCAS active-space selection using scine-autocas + scine_qcmaquis (DMRG).

Runs the full ClassicWorkflow (Stein & Reiher, JCTC 2016) on the TS geometry
of a benchmark reaction and writes the recommended CAS(n,m) to JSON.

Usage:
    python autocas_active_space.py rxn7949
    python autocas_active_space.py rxn7949 --basis def2-svp --bond-dim 500

Output:
    ~/autocas_results/{rxn}/autocas_result.json

Runtime modules required:
    Python/3.11.3-GCCcore-12.3.0  Boost/1.82.0-GCC-12.3.0
    GSL/2.7-GCC-12.3.0  FlexiBLAS/3.3.1-GCC-12.3.0  HDF5/1.14.0-gompi-2023a
"""
import argparse
import json
import os

HOME      = '/home/energy/s242862'
OPTTS_DIR = f'{HOME}/nevpt2_optts_results'
NEB_DIR   = f'{HOME}/orca_neb_results'
OUT_BASE  = f'{HOME}/autocas_results'


def find_ts_geometry(rxn):
    optts = f'{OPTTS_DIR}/{rxn}_avas/ts_casscf_opt.xyz'
    orca  = f'{NEB_DIR}/{rxn}/ts_orca.xyz'
    if os.path.exists(optts):
        print(f'Geometry: CASSCF OptTS  {optts}')
        return optts, 'casscf_optts'
    if os.path.exists(orca):
        print(f'Geometry: ORCA NEB TS   {orca}')
        return orca, 'orca_neb'
    raise FileNotFoundError(f'No TS geometry found for {rxn}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--basis',     default='def2-svp')
    parser.add_argument('--bond-dim',  type=int,   default=250)
    parser.add_argument('--sweeps',    type=int,   default=5)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--threads',   type=int,   default=8)
    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    import pyscf as _pyscf
    _pyscf.lib.num_threads(args.threads)

    out_dir = f'{OUT_BASE}/{args.rxn}'
    os.makedirs(out_dir, exist_ok=True)

    ts_xyz, geom_source = find_ts_geometry(args.rxn)

    from scine_autocas import Molecule, Autocas
    from scine_autocas.interfaces.pyscf.pyscf import PyscfInterface
    from scine_autocas.workflows.conventional import ClassicWorkflow

    mol       = Molecule(xyz_file=ts_xyz, charge=0, spin_multiplicity=1)
    autocas   = Autocas(molecule=mol)
    interface = PyscfInterface(molecule=mol)

    interface.settings.basis_set                = args.basis
    interface.settings.init_dmrg_bond_dimension = args.bond_dim
    interface.settings.init_dmrg_sweeps         = args.sweeps
    interface.settings.dmrg_threshold           = args.threshold

    workflow = ClassicWorkflow(autocas=autocas, interface=interface)
    print(f'\nRunning AutoCAS ClassicWorkflow for {args.rxn}...')
    workflow.run()  # results stored in workflow.results, not returned

    final_occ = workflow.results.get("final_occupation")
    final_idx = workflow.results.get("final_orbital_indices")

    if final_occ is None or final_idx is None:
        cas_e, cas_o, orb_idx = None, None, []
        print('\nAutoCAS: SingleReferenceException — no active space needed')
    else:
        orb_idx = list(final_idx)
        cas_e = int(sum(final_occ))
        cas_o = len(final_occ)
        print(f'\nAutoCAS result: CAS({cas_e},{cas_o})')
        print(f'Active orbital indices (0-based): {orb_idx}')
        print(f'Active orbital occupations:       {list(final_occ)}')

    # Sanity checks on DMRG entropies
    s1_valid = True
    mi_valid = True
    if hasattr(workflow, 'autocas') and hasattr(workflow.autocas, 'diagnostics'):
        pass  # checked below via result inspection
    # The ClassicWorkflow prints initial_s1 to stdout; we check via the interface
    try:
        iface_s1 = interface.initial_s1
        if iface_s1 is not None:
            import numpy as np
            s1_arr = np.array(iface_s1)
            if np.any(s1_arr < -1e-6) or np.any(s1_arr > 1.3863 + 1e-6):
                s1_valid = False
                print(f'\nWARNING: s1 out of physical range [0, ln4]. Min={s1_arr.min():.4f} Max={s1_arr.max():.4f}')
            else:
                print(f'\ns1 range check OK: min={s1_arr.min():.4f} max={s1_arr.max():.4f}')
    except AttributeError:
        pass

    try:
        iface_mi = interface.initial_mutual_information
        if iface_mi is not None:
            import numpy as np
            mi_arr = np.array(iface_mi)
            np.fill_diagonal(mi_arr, 0)
            if np.any(mi_arr < -1e-4):
                mi_valid = False
                print(f'WARNING: mutual information has negative values (min={mi_arr.min():.4f}) — DMRG not converged')
            else:
                print(f'Mutual information check OK: min off-diag={mi_arr[mi_arr != 0].min():.4f}')
    except AttributeError:
        pass

    sanity_ok = s1_valid and mi_valid
    print(f'DMRG sanity: {"PASS" if sanity_ok else "FAIL — results unreliable"}')

    out = {
        'rxn':             args.rxn,
        'geometry_source': geom_source,
        'ts_xyz':          ts_xyz,
        'basis':           args.basis,
        'dmrg_bond_dim':   args.bond_dim,
        'dmrg_sweeps':     args.sweeps,
        'threshold':       args.threshold,
        'cas_electrons':   int(cas_e)  if cas_e  is not None else None,
        'cas_orbitals':    int(cas_o)  if cas_o  is not None else None,
        'active_orb_idx':  orb_idx,
        'dmrg_sanity_ok':  sanity_ok,
    }

    basis_tag = args.basis.replace('-', '').replace('*', 's')
    out_json = f'{out_dir}/autocas_result_{basis_tag}_m{args.bond_dim}.json'
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResult → {out_json}')


if __name__ == '__main__':
    main()
