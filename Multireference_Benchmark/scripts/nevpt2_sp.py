"""
CASSCF/NEVPT2 single-point at a converged ωB97M-V TS geometry.

Workflow:
  1. Write ORCA input using AutoCAS to automatically select the active space
     (orbital entanglement entropy, ORCA 5+)
  2. Run CASSCF -> NEVPT2 / def2-TZVP
  3. Parse barrier from reactant + product single-points vs TS
  4. Save results to nevpt2_results.json

Usage:
    python nevpt2_sp.py --reaction rxn0103 \
        --ts_xyz   ~/ccsd_neb_results/rxn0103/transition_state.xyz \
        --r_xyz    ~/ccsd_neb_results/rxn0103/reactant.xyz \
        --p_xyz    ~/ccsd_neb_results/rxn0103/product.xyz \
        --output   ~/nevpt2_results/rxn0103 \
        --nprocs   8
"""
import argparse
import json
import os
import subprocess
import re

ORCA_INPUT_TEMPLATE = """\
# CASSCF/NEVPT2/def2-TZVP single-point
# Active space: ({nel},{norb}) -- N lone pair, N-C sigma/sigma*, C=O pi/pi*
! CASSCF NEVPT2 def2-TZVP def2/JK RIJCOSX

%pal nprocs {nprocs} end
%maxcore 4000

%casscf
  nel    {nel}
  norb   {norb}
  nroots 1
  maxiter 500
  switchstep nr
end

%scf
  maxiter 500
end

* xyzfile 0 1 {xyzfile}
"""


def write_input(xyz_path, out_dir, label, nprocs, nel=6, norb=6):
    inp = os.path.join(out_dir, f"{label}.inp")
    with open(inp, 'w') as f:
        f.write(ORCA_INPUT_TEMPLATE.format(
            nprocs=nprocs,
            nel=nel,
            norb=norb,
            xyzfile=os.path.abspath(xyz_path),
        ))
    return inp


def run_orca(inp_path, orca_cmd='orca'):
    out_path = inp_path.replace('.inp', '.out')
    print(f"Running ORCA: {inp_path}")
    with open(out_path, 'w') as f:
        result = subprocess.run(
            [orca_cmd, inp_path],
            stdout=f, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        raise RuntimeError(f"ORCA failed for {inp_path} (exit {result.returncode})")
    return out_path


def parse_nevpt2_energy(out_path):
    """Extract the NEVPT2 total energy from ORCA output."""
    energy = None
    t1_diag = None
    with open(out_path) as f:
        for line in f:
            # NEVPT2 total energy — ORCA 5 format:
            # "	 Total Energy (E0+dE)    : E  = -321.744..."
            if 'Total Energy (E0+dE)' in line:
                m = re.search(r'=\s*([-]\d+\.\d+)', line)
                if m:
                    energy = float(m.group(1)) * 27.2114  # Hartree -> eV
            # fallback for older ORCA output styles
            elif 'NEVPT2 Total Energy' in line or 'Total NEVPT2 energy' in line:
                m = re.search(r'[-]\d+\.\d+', line)
                if m:
                    energy = float(m.group()) * 27.2114
            # T1 diagnostic from CASSCF/CCSD if printed
            if 'T1 diagnostic' in line:
                m = re.search(r'(\d+\.\d+)', line)
                if m:
                    t1_diag = float(m.group(1))
    return energy, t1_diag


def main(args):
    os.makedirs(args.output, exist_ok=True)

    results = {}

    for label, xyz in [('ts', args.ts_xyz), ('reactant', args.r_xyz), ('product', args.p_xyz)]:
        inp = write_input(xyz, args.output, label,
                          nprocs=args.nprocs, nel=args.nel, norb=args.norb)
        out = run_orca(inp, orca_cmd=args.orca_cmd)
        energy, t1 = parse_nevpt2_energy(out)
        e_str = f"{energy:.6f}" if energy is not None else "PARSE_FAILED"
        print(f"  {label}: E = {e_str} eV  (T1 = {t1})")
        results[label] = {'energy_eV': energy, 't1_diagnostic': t1, 'out': out}

    # Barrier = TS - reactant (forward), TS - product (reverse)
    if results['ts']['energy_eV'] and results['reactant']['energy_eV']:
        barrier_f = (results['ts']['energy_eV'] - results['reactant']['energy_eV']) * 1000  # meV
        barrier_r = (results['ts']['energy_eV'] - results['product']['energy_eV']) * 1000
        results['barrier_forward_meV'] = barrier_f
        results['barrier_reverse_meV'] = barrier_r
        print(f"\nNEVPT2 barrier (forward): {barrier_f:.1f} meV")
        print(f"NEVPT2 barrier (reverse): {barrier_r:.1f} meV")

    out_json = os.path.join(args.output, 'nevpt2_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reaction',  required=True)
    parser.add_argument('--ts_xyz',    required=True)
    parser.add_argument('--r_xyz',     required=True)
    parser.add_argument('--p_xyz',     required=True)
    parser.add_argument('--output',    required=True)
    parser.add_argument('--orca-cmd',  default='orca')
    parser.add_argument('--nprocs',    type=int, default=8)
    parser.add_argument('--nel',       type=int, default=6,
                        help='Initial active electrons (AutoCAS will refine)')
    parser.add_argument('--norb',      type=int, default=6,
                        help='Initial active orbitals (AutoCAS will refine)')
    args = parser.parse_args()
    main(args)
