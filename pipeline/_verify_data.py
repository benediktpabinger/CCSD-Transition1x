"""
End-to-end data verification.
Checks every assumption the analysis makes against the raw files on the cluster.
"""
import os, getpass
import paramiko, json, io
import numpy as np

HOST = os.environ.get('SSH_HOST', 'slid.fysik.dtu.dk')
USER = os.environ.get('SSH_USER')
PASS = os.environ.get('SSH_PASS') or getpass.getpass(f'SSH password for {USER}@{HOST}: ')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS)

def run(cmd):
    _, out, err = ssh.exec_command(cmd)
    return out.read().decode('utf-8', errors='replace').strip()

sftp = ssh.open_sftp()

def load_json(path):
    try:
        with sftp.open(path) as f:
            return json.load(f)
    except Exception as e:
        return f'ERROR: {e}'

HOME = '/home/energy/s242862'
TOP10 = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
         'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150']
SAMPLE = ['rxn7949', 'rxn9246', 'rxn0896']  # one per group

SEP = '=' * 80

# ────────────────────────────────────────────────────────────────────────────
# CHECK 1: T1x HDF5 — shape of transition_state/positions
# ────────────────────────────────────────────────────────────────────────────
print(SEP)
print('CHECK 1: T1x HDF5 — transition_state/positions shape')
print('Question: is [0] the full (N,3) geometry or just first atom?')
print(SEP)

inspect = run("""python3 -c "
import h5py, numpy as np
f = h5py.File('/home/energy/s242862/data/Transition1x.h5', 'r')
for split in ('test','val','train'):
    for formula in f[split]:
        for rxn in f[split][formula]:
            if rxn == 'rxn7949':
                grp = f[split][formula][rxn]
                ts = grp['transition_state']
                print('Keys in transition_state:', list(ts.keys()))
                print('positions shape:', ts['positions'].shape)
                print('atomic_numbers shape:', ts['atomic_numbers'].shape)
                print('energy shape:', ts['wB97x_6-31G(d).energy'].shape)
                pos = ts['positions'][:]
                print('positions[0] shape:', pos[0].shape if hasattr(pos[0],'shape') else type(pos[0]))
                print('First 3 rows of positions[0]:')
                print(np.round(pos[0][:3], 4))
                print('Number of atoms:', len(pos[0]))
                r = grp['reactant']
                print('reactant positions shape:', r['positions'].shape)
                print('reactant energy:', r['wB97x_6-31G(d).energy'][:])
                print('ts energy:', ts['wB97x_6-31G(d).energy'][:])
                print('barrier (ts-r)*1000 meV:', (ts['wB97x_6-31G(d).energy'][0] - r['wB97x_6-31G(d).energy'][0])*1000)
" 2>&1""")
print(inspect)

# ────────────────────────────────────────────────────────────────────────────
# CHECK 2: neb.db structure — how many rows, what are they?
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print('CHECK 2: neb.db — total rows, energy profile, what are rows[-10:]?')
print('Question: does rows[-10:][0] correspond to the reactant (lowest energy)?')
print(SEP)

for rxn in SAMPLE:
    inspect2 = run(f"""python3 -c "
import ase.db, numpy as np
db_path = '{HOME}/orca_neb_results/{rxn}/neb.db'
with ase.db.connect(db_path) as db:
    rows = list(db.select())
print('{rxn}: total rows =', len(rows))
energies_all = [r.toatoms().get_potential_energy() for r in rows]
energies_last10 = [r.toatoms().get_potential_energy() for r in rows[-10:]]
print('  First 5 energies (eV):', [round(e,4) for e in energies_all[:5]])
print('  Last 10 energies (eV):', [round(e,4) for e in energies_last10])
e = np.array(energies_last10)
rel = (e - e[0]) * 1000
print('  Relative energies last10 (meV):', np.round(rel, 1).tolist())
print('  Max-rel idx:', int(rel.argmax()), '  max-rel val:', round(rel.max(),1))
print('  Is E[0] lowest?', energies_last10[0] == min(energies_last10))
print('  Is E[-1] (product) second lowest?', round(energies_last10[-1],4))
" 2>&1""")
    print(inspect2)
    print()

# ────────────────────────────────────────────────────────────────────────────
# CHECK 3: CCSD(T) uses transition_state.xyz — verify it exists and matches neb TS
# ────────────────────────────────────────────────────────────────────────────
print(SEP)
print('CHECK 3: CCSD(T) geometry source — transition_state.xyz from orca_neb_results/')
print('Question: does this file exist and how does its energy compare to max-NEB-image?')
print(SEP)

for rxn in SAMPLE[:2]:
    inspect3 = run(f"""python3 -c "
import json, os
from ase.io import read
import ase.db, numpy as np

# CCSD(T) barrier
ccsdt_path = '{HOME}/mr_benchmark/results/{rxn}_ccsdt.json'
if os.path.exists(ccsdt_path):
    with open(ccsdt_path) as f:
        d = json.load(f)
    fwd_val = d.get('barrier_fwd_meV')
    print(f'{rxn} CCSD(T): fwd={fwd_val} meV  (from transition_state.xyz)')
else:
    print('{rxn} CCSD(T): FILE MISSING')

# transition_state.xyz - does it exist?
ts_xyz = '{HOME}/orca_neb_results/{rxn}/transition_state.xyz'
if os.path.exists(ts_xyz):
    ts = read(ts_xyz)
    print('  transition_state.xyz: N_atoms =', len(ts), '  positions[0]:', np.round(ts.get_positions()[0], 4))
else:
    print('  transition_state.xyz: MISSING')

# wB97M-V energy at TS from neb.db (max of last 10)
with ase.db.connect('{HOME}/orca_neb_results/{rxn}/neb.db') as db:
    rows = list(db.select())
e_last10 = np.array([r.toatoms().get_potential_energy() for r in rows[-10:]])
rel = (e_last10 - e_last10[0]) * 1000
print('  wB97M-V barrier from neb.db last10:', round(rel.max(), 1), 'meV  (max-E image idx:', int(rel.argmax()), ')')
" 2>&1""")
    print(inspect3)
    print()

# ────────────────────────────────────────────────────────────────────────────
# CHECK 4: orca_engrad — verify energy for a specific image matches parsing
# ────────────────────────────────────────────────────────────────────────────
print(SEP)
print('CHECK 4: orca_engrad — verify parse_engrad gives correct wB97X-D3 energy')
print('Question: does max(orca_engrad[0:9]) - orca_engrad[0] match T1x barrier?')
print(SEP)

for rxn in ['rxn7949']:
    inspect4 = run(f"""python3 -c "
import re, os, numpy as np
import ase.db

ENERGY_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
EH_TO_EV = 27.2114

def parse_energy(sp_dir):
    out = sp_dir + '/sp.out'
    if not os.path.exists(out): return None
    with open(out) as f: content = f.read()
    m = ENERGY_RE.findall(content)
    return float(m[-1]) * EH_TO_EV if m else None

e_list = []
for i in range(10):
    sp_dir = '{HOME}/mr_benchmark/orca_engrad/{rxn}/geom_{i:04d}'.format(i=i)
    e = parse_energy(sp_dir)
    e_list.append(e)

print('{rxn} orca_engrad energies (eV):', [round(e,4) if e else None for e in e_list])
e = np.array([x for x in e_list if x is not None])
rel = (e - e[0]) * 1000
print('  Relative energies (meV):', np.round(rel,1).tolist())
print('  wB97X-D3 barrier from engrad:', round(rel.max(),1), 'meV')

# Compare with neb.db wB97M-V barrier for sanity
with ase.db.connect('{HOME}/orca_neb_results/{rxn}/neb.db') as db:
    rows = list(db.select())
e_neb = np.array([r.toatoms().get_potential_energy() for r in rows[-10:]])
rel_neb = (e_neb - e_neb[0]) * 1000
print('  wB97M-V barrier from neb.db:', round(rel_neb.max(),1), 'meV')
" 2>&1""")
    print(inspect4)

# ────────────────────────────────────────────────────────────────────────────
# CHECK 5: eval_benchmark_sp.json — verify MACE energy for rxn7949
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print('CHECK 5: eval_benchmark_sp.json — spot check rxn7949 energies')
print(SEP)

d = load_json(f'{HOME}/delta_head/eval_benchmark_sp.json')
if isinstance(d, str):
    print(d)
else:
    r7949 = next(r for r in d['reactions'] if r['rxn'] == 'rxn7949')
    e_m = np.array(r7949['e_wb97m_eV'])
    e_x = np.array(r7949['e_wb97x_eV'])
    e_mace = np.array(r7949['e_mace_eV'])
    e_delta = np.array(r7949['e_delta_eV'])
    def rel(e): return (e - e[0]) * 1000

    print('rxn7949 relative energies (meV), 10 images:')
    print(f'  wB97M-V:    {np.round(rel(e_m),1).tolist()}')
    print(f'  wB97X-D3:   {np.round(rel(e_x),1).tolist()}')
    print(f'  MACE:       {np.round(rel(e_mace),1).tolist()}')
    print(f'  MACE+delta: {np.round(rel(e_delta),1).tolist()}')
    print(f'  wB97M-V barrier (max):    {rel(e_m).max():.1f} meV')
    print(f'  wB97X-D3 barrier (max):   {rel(e_x).max():.1f} meV')
    print(f'  MACE barrier (max):       {rel(e_mace).max():.1f} meV')
    print(f'  MACE+delta barrier (max): {rel(e_delta).max():.1f} meV')
    print()
    print('  Cross-check with full_benchmark_results.json:')
    bm = load_json('/home/energy/s242862/delta_head/full_benchmark_results.json')
    if not isinstance(bm, str):
        r = next(r for r in bm['reactions'] if r['rxn'] == 'rxn7949')
        print(f'  neb_wb97m_fwd_meV = {r["neb_wb97m_fwd_meV"]}  (should match wB97M-V barrier above)')
        print(f'  mace_fwd_meV      = {r["mace_fwd_meV"]}  (should match MACE barrier above)')
        print(f'  delta_fwd_meV     = {r["delta_fwd_meV"]}  (should match delta barrier above)')

# ────────────────────────────────────────────────────────────────────────────
# CHECK 6: NEVPT2 — verify which reactions are reliable and confirm barriers
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print('CHECK 6: NEVPT2 — which top-10 reactions are reliable, spot-check barriers')
print(SEP)

FRAC_LO, FRAC_HI = 0.05, 1.95
def count_frac(occ): return sum(1 for n in occ if FRAC_LO < n < FRAC_HI)

for rxn in TOP10:
    path = f'{HOME}/nevpt2_results/{rxn}_pyscf_avas/nevpt2_results.json'
    nev = load_json(path)
    if isinstance(nev, str):
        print(f'{rxn}: {nev}')
        continue
    r_frac  = count_frac(nev.get('reactant', {}).get('nat_occ', []))
    ts_frac = count_frac(nev.get('ts', {}).get('nat_occ', []))
    p_frac  = count_frac(nev.get('product', {}).get('nat_occ', []))
    fwd = nev.get('barrier_forward_meV')
    rev = nev.get('barrier_reverse_meV')
    neg_rev = rev is not None and rev < 0
    flags = []
    if r_frac == 0: flags.append('0frac@R')
    if ts_frac == 0: flags.append('0frac@TS')
    if p_frac == 0: flags.append('0frac@P')
    if neg_rev: flags.append(f'neg_rev={rev:.0f}')
    reliable = len(flags) == 0
    print(f'{rxn}: frac(R/TS/P)={r_frac}/{ts_frac}/{p_frac}  fwd={fwd:.0f if fwd else "N/A"}  rev={rev:.0f if rev else "N/A"}  '
          f'reliable={reliable}  {"FLAGS: " + str(flags) if flags else ""}')

# ────────────────────────────────────────────────────────────────────────────
# CHECK 7: CCSD(T) — confirm all 30 reactions have results
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print('CHECK 7: CCSD(T) — do all 30 reactions have results?')
print(SEP)

ALL30 = TOP10 + ['rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
                 'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
                 'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
                 'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155']

for rxn in ALL30:
    cct = load_json(f'{HOME}/mr_benchmark/results/{rxn}_ccsdt.json')
    if isinstance(cct, str):
        print(f'{rxn}: MISSING')
    else:
        fwd = cct.get('barrier_fwd_meV')
        rev = cct.get('barrier_rev_meV')
        print(f'{rxn}: fwd={fwd:.0f if fwd else "ERR"}  rev={rev:.0f if rev else "ERR"}')

sftp.close()
ssh.close()
