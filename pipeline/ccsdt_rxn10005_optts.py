"""
CCSD(T)/def2-TZVP single points for rxn10005 at the CASSCF OptTS geometry.

Geometries:
  TS  = ~/nevpt2_optts_results/rxn10005_avas/ts_casscf_opt.xyz  (CASSCF-optimised)
  R/P = ~/orca_neb_results/rxn10005/reactant.xyz / product.xyz  (ORCA NEB endpoints)

NEVPT2(opt) = 3452 meV. Largest molecule (20e,13o AS) — 96GB/24h requested.

Run on cluster: python3 pipeline/ccsdt_rxn10005_optts.py
"""
import json, os, numpy as np
from pyscf import gto, scf, cc

RXN        = 'rxn10005'
NEVPT2_OPT = 3452

TS_XYZ  = os.path.expanduser(f'~/nevpt2_optts_results/{RXN}_avas/ts_casscf_opt.xyz')
R_XYZ   = os.path.expanduser(f'~/orca_neb_results/{RXN}/reactant.xyz')
P_XYZ   = os.path.expanduser(f'~/orca_neb_results/{RXN}/product.xyz')
OUTFILE = os.path.expanduser(f'~/nevpt2_optts_results/{RXN}_avas/ccsdt_{RXN}_optts_result.json')

BASIS  = 'def2-tzvp'
CHARGE = 0
SPIN   = 0


def read_xyz(path):
    """Parse standard or ASE-extended XYZ. Returns (atoms, coords_angstrom)."""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms, coords = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(coords)


def run_ccsdt(label, atoms, coords):
    print(f'\n{"="*60}')
    print(f'CCSD(T) for {label}')
    print(f'{"="*60}')

    mol = gto.Mole()
    mol.atom = [[a, tuple(c)] for a, c in zip(atoms, coords)]
    mol.basis = BASIS
    mol.charge = CHARGE
    mol.spin   = SPIN
    mol.verbose = 4
    mol.max_memory = 56000   # MB — larger for rxn10005
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.conv_tol  = 1e-10
    e_hf = mf.kernel()
    if not mf.converged:
        print(f'WARNING: RHF did not converge for {label}')

    mycc = cc.RCCSD(mf)
    mycc.max_cycle = 200
    mycc.conv_tol  = 1e-8
    e_corr, _, _ = mycc.kernel()
    e_t = mycc.ccsd_t()

    e_total = e_hf + e_corr + e_t
    print(f'{label}: E(HF)={e_hf:.8f}  E(corr)={e_corr:.8f}  E(T)={e_t:.8f}  E(tot)={e_total:.8f} Ha')
    return {'label': label, 'e_hf': e_hf, 'e_ccsd': e_hf + e_corr,
            'e_t': e_t, 'e_total': e_total, 'converged': mf.converged}


r_atoms,  r_coords  = read_xyz(R_XYZ)
ts_atoms, ts_coords = read_xyz(TS_XYZ)
p_atoms,  p_coords  = read_xyz(P_XYZ)
print(f'{RXN}: {len(ts_atoms)} atoms, basis {BASIS}')

results = {}
for label, atoms, coords in [('R', r_atoms, r_coords),
                              ('TS', ts_atoms, ts_coords),
                              ('P', p_atoms, p_coords)]:
    results[label] = run_ccsdt(label, atoms, coords)

Ha_to_meV = 27211.4
e_r  = results['R']['e_total']
e_ts = results['TS']['e_total']
e_p  = results['P']['e_total']
fwd_meV = (e_ts - e_r) * Ha_to_meV
rev_meV = (e_ts - e_p) * Ha_to_meV

print(f'\n{"="*60}')
print(f'{RXN} CCSD(T)/def2-TZVP at CASSCF OptTS geometry')
print(f'  Forward barrier:  {fwd_meV:.1f} meV')
print(f'  Reverse barrier:  {rev_meV:.1f} meV')
print(f'  NEVPT2(opt):      {NEVPT2_OPT} meV')
print(f'  Delta:            {fwd_meV - NEVPT2_OPT:.1f} meV  (CCSD(T) - NEVPT2)')
print(f'{"="*60}')

output = {
    'rxn': RXN,
    'geometry': 'CASSCF OptTS (TS) + ORCA NEB (R, P)',
    'basis': BASIS,
    'method': 'RCCSD(T)',
    'points': list(results.values()),
    'fwd_meV': round(fwd_meV, 1),
    'rev_meV': round(rev_meV, 1),
    'nevpt2_opt_fwd_meV': NEVPT2_OPT,
    'delta_ccsdt_minus_nevpt2_meV': round(fwd_meV - NEVPT2_OPT, 1),
}
with open(OUTFILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved to {OUTFILE}')
