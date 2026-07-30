"""
Generate and run ORCA BS-UKS NEB-TS for High-MR + next-HIGH reactions.
Uses wB97M-V/def2-TZVP with BrokenSym 1,1 to find diradical-character TSs.

Reuses reactant.xyz and product.xyz from the existing ORCA RKS NEB results.
Output TS geometry can then be compared to:
  - ORCA RKS NEB TS  (= current DFT reference)
  - CASSCF OptTS     (= MR reference)

Usage (called by SLURM job):
    python bs_uks_neb.py rxn7949 --nprocs 8 --maxcore 8000
"""
import argparse, os, subprocess, shutil

HOME     = '/home/energy/s242862'
NEB_DIR  = f'{HOME}/orca_neb_results'
OUT_BASE = f'{HOME}/bs_uks_neb_results'


def write_orca_input(rxn, out_dir, nprocs, maxcore):
    r_xyz = f'{NEB_DIR}/{rxn}/reactant.xyz'
    p_xyz = f'{out_dir}/product.xyz'      # copied locally so ORCA can find it

    inp = f"""\
! UKS wB97M-V def2-TZVP NEB-TS TightSCF SlowConv

%pal
  nprocs {nprocs}
end

%maxcore {maxcore}

%scf
  BrokenSym 1,1
  MaxIter 500
end

%neb
  Product "{p_xyz}"
  NImages 8
  MaxIter 500
  Preopt true
  PrintLevel 3
end

* xyzfile 0 1 {r_xyz}
"""
    inp_path = os.path.join(out_dir, 'bs_uks_neb.inp')
    with open(inp_path, 'w') as f:
        f.write(inp)
    return inp_path


def main(args):
    rxn     = args.rxn
    out_dir = f'{OUT_BASE}/{rxn}'
    os.makedirs(out_dir, exist_ok=True)

    # Copy product.xyz locally (ORCA needs it accessible from the run directory)
    p_src = f'{NEB_DIR}/{rxn}/product.xyz'
    p_dst = f'{out_dir}/product.xyz'
    shutil.copy2(p_src, p_dst)
    print(f'{rxn}: copied product.xyz → {p_dst}', flush=True)

    # Write ORCA input
    inp_path = write_orca_input(rxn, out_dir, args.nprocs, args.maxcore)
    print(f'{rxn}: wrote {inp_path}', flush=True)

    # Run ORCA
    out_log = os.path.join(out_dir, 'bs_uks_neb.out')
    print(f'{rxn}: running ORCA NEB-TS (BS-UKS) → {out_log}', flush=True)
    with open(out_log, 'w') as log:
        ret = subprocess.run(
            [args.orca_path, inp_path],
            cwd=out_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if ret.returncode != 0:
        print(f'{rxn}: ORCA exited with code {ret.returncode} — check {out_log}', flush=True)
    else:
        # ORCA NEB-TS writes the optimised TS to bs_uks_neb_NEB-TS_converged.xyz
        ts_candidates = [
            f'{out_dir}/bs_uks_neb_NEB-TS_converged.xyz',
            f'{out_dir}/bs_uks_neb_NEB-HEI_image.xyz',   # fallback: highest image
        ]
        found = None
        for p in ts_candidates:
            if os.path.exists(p):
                found = p
                break
        if found:
            dst = f'{out_dir}/transition_state_bs_uks.xyz'
            shutil.copy2(found, dst)
            print(f'{rxn}: TS geometry → {dst}', flush=True)
        else:
            print(f'{rxn}: WARNING — no TS xyz found in {out_dir}', flush=True)

    print(f'{rxn}: done.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rxn')
    parser.add_argument('--nprocs',    type=int, default=8)
    parser.add_argument('--maxcore',   type=int, default=8000)
    parser.add_argument('--orca-path', default='orca',
                        help='Full path to ORCA executable if not in PATH')
    args = parser.parse_args()
    main(args)
