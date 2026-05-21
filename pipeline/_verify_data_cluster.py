"""
End-to-end data verification — runs on cluster where h5py/ase are available.
"""
import json, os, re
import numpy as np
import h5py
import ase.db
from ase.io import read

HOME = '/home/energy/s242862'
SEP  = '=' * 80

TOP10    = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
            'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150']
BOTTOM10 = ['rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
            'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962']
MIDDLE10 = ['rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
            'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155']
ALL30 = TOP10 + BOTTOM10 + MIDDLE10

FRAC_LO, FRAC_HI = 0.05, 1.95
ENERGY_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')


# ─── CHECK 1: T1x HDF5 — shape of transition_state/positions ────────────────
print(SEP)
print('CHECK 1: T1x HDF5 — transition_state/positions shape for rxn7949')
print('Does positions[0] give (N_atoms, 3) or just one atom position?')
print(SEP)

with h5py.File(f'{HOME}/data/Transition1x.h5', 'r') as h5:
    for split in ('test','val','train'):
        for formula in h5.get(split, {}):
            if 'rxn7949' in h5[split][formula]:
                grp = h5[split][formula]['rxn7949']
                ts  = grp['transition_state']
                r   = grp['reactant']
                print(f'  Found in split={split}, formula={formula}')
                print(f'  transition_state keys: {list(ts.keys())}')
                print(f'  positions dataset shape: {ts["positions"].shape}')
                print(f'  positions[0] shape: {np.array(ts["positions"][0]).shape}')
                print(f'  N atoms: {len(ts["positions"][0])}')
                print(f'  First 3 atom positions: {np.round(ts["positions"][0][:3], 4).tolist()}')
                ts_e = float(ts['wB97x_6-31G(d).energy'][0])
                r_e  = float(r['wB97x_6-31G(d).energy'][0])
                print(f'  T1x TS energy: {ts_e:.6f} Ha')
                print(f'  T1x R  energy: {r_e:.6f} Ha')
                print(f'  T1x barrier (ts-r)*1000: {(ts_e - r_e)*1000:.1f} meV')
                break

# ─── CHECK 2: neb.db — structure of rows ────────────────────────────────────
print()
print(SEP)
print('CHECK 2: neb.db — total rows, is rows[-10:][0] the reactant?')
print(SEP)

for rxn in ['rxn7949', 'rxn9246', 'rxn0896']:
    db_path = f'{HOME}/orca_neb_results/{rxn}/neb.db'
    with ase.db.connect(db_path) as db:
        rows = list(db.select())
    n_total = len(rows)
    e_all  = [r.toatoms().get_potential_energy() for r in rows]
    e_last = [r.toatoms().get_potential_energy() for r in rows[-10:]]
    e_arr  = np.array(e_last)
    rel    = (e_arr - e_arr[0]) * 1000

    print(f'{rxn}:')
    print(f'  Total rows in neb.db: {n_total}')
    print(f'  First 5 energies (eV): {[round(e,4) for e in e_all[:5]]}')
    print(f'  Last 10 energies (eV): {[round(e,4) for e in e_last]}')
    print(f'  Relative last-10 (meV): {np.round(rel,1).tolist()}')
    print(f'  TS image index (argmax): {int(rel.argmax())}')
    print(f'  Forward barrier (max-first): {rel.max():.1f} meV')
    print(f'  Is E[0] the minimum of last10? {bool(e_last[0] == min(e_last))}')
    print(f'  Is E[-1] also low (product)? E[-1]-E[0]={rel[-1]:.1f} meV')
    print()

# ─── CHECK 3: CCSD(T) geometry — transition_state.xyz vs max-NEB-image ──────
print(SEP)
print('CHECK 3: What geometry does CCSD(T) use?')
print('Is transition_state.xyz = the NEB TS image or a separately optimised structure?')
print(SEP)

for rxn in ['rxn7949', 'rxn8832']:
    ts_xyz = f'{HOME}/orca_neb_results/{rxn}/transition_state.xyz'
    cct_path = f'{HOME}/mr_benchmark/results/{rxn}_ccsdt.json'

    if os.path.exists(ts_xyz):
        ts_atoms = read(ts_xyz)
        ts_pos = ts_atoms.get_positions()
        print(f'{rxn} transition_state.xyz: N_atoms={len(ts_atoms)}')
        print(f'  First 3 positions: {np.round(ts_pos[:3], 4).tolist()}')
    else:
        print(f'{rxn} transition_state.xyz: MISSING')

    # Compare with max-energy NEB image position
    with ase.db.connect(f'{HOME}/orca_neb_results/{rxn}/neb.db') as db:
        rows = list(db.select())
    e_last = [r.toatoms().get_potential_energy() for r in rows[-10:]]
    ts_idx = int(np.argmax(e_last))
    neb_ts_atoms = rows[-10:][ts_idx].toatoms()
    neb_ts_pos   = neb_ts_atoms.get_positions()
    print(f'  Max-energy NEB image index: {ts_idx} of last 10')
    print(f'  NEB TS first 3 positions:  {np.round(neb_ts_pos[:3], 4).tolist()}')

    if os.path.exists(ts_xyz):
        if ts_pos.shape == neb_ts_pos.shape:
            rmsd = float(np.sqrt(np.mean((ts_pos - neb_ts_pos)**2)))
            print(f'  RMSD(transition_state.xyz vs max-NEB-image): {rmsd:.4f} A')
        else:
            print(f'  Shape mismatch: {ts_pos.shape} vs {neb_ts_pos.shape}')

    if os.path.exists(cct_path):
        with open(cct_path) as f:
            cct = json.load(f)
        print(f'  CCSD(T) barrier: {cct.get("barrier_fwd_meV"):.1f} meV')
    print()

# ─── CHECK 4: orca_engrad — wB97X-D3 barrier vs neb.db barrier ──────────────
print(SEP)
print('CHECK 4: orca_engrad parsing — do energies use same 10 images as neb.db?')
print(SEP)

for rxn in ['rxn7949', 'rxn9246']:
    e_list = []
    for i in range(10):
        sp_dir = f'{HOME}/mr_benchmark/orca_engrad/{rxn}/geom_{i:04d}'
        out_path = f'{sp_dir}/sp.out'
        if not os.path.exists(out_path):
            e_list.append(None)
            continue
        with open(out_path) as f:
            content = f.read()
        m = ENERGY_RE.findall(content)
        e_list.append(float(m[-1]) * 27.2114 if m else None)

    if any(e is None for e in e_list):
        print(f'{rxn}: missing engrad outputs: {[i for i,e in enumerate(e_list) if e is None]}')
        continue

    e_arr = np.array(e_list)
    rel_x = (e_arr - e_arr[0]) * 1000

    with ase.db.connect(f'{HOME}/orca_neb_results/{rxn}/neb.db') as db:
        rows = list(db.select())
    e_last = np.array([r.toatoms().get_potential_energy() for r in rows[-10:]])
    rel_m = (e_last - e_last[0]) * 1000

    print(f'{rxn}:')
    print(f'  orca_engrad wB97X-D3 relative (meV): {np.round(rel_x,1).tolist()}')
    print(f'  neb.db     wB97M-V   relative (meV): {np.round(rel_m,1).tolist()}')
    print(f'  wB97X-D3 barrier: {rel_x.max():.1f}  wB97M-V barrier: {rel_m.max():.1f} meV')
    print(f'  TS image agrees? wB97X-D3={int(rel_x.argmax())} wB97M-V={int(rel_m.argmax())}')
    print()

# ─── CHECK 5: NEVPT2 plausibility for all top-10 ────────────────────────────
print(SEP)
print('CHECK 5: NEVPT2 — which top-10 are reliable?')
print(SEP)

def count_frac(occ): return sum(1 for n in occ if FRAC_LO < n < FRAC_HI)

n_reliable = 0
for rxn in TOP10:
    path = f'{HOME}/nevpt2_results/{rxn}_pyscf_avas/nevpt2_results.json'
    if not os.path.exists(path):
        print(f'{rxn}: FILE MISSING'); continue
    with open(path) as f:
        nev = json.load(f)
    r_f  = count_frac(nev.get('reactant', {}).get('nat_occ', []))
    ts_f = count_frac(nev.get('ts', {}).get('nat_occ', []))
    p_f  = count_frac(nev.get('product', {}).get('nat_occ', []))
    fwd  = nev.get('barrier_forward_meV')
    rev  = nev.get('barrier_reverse_meV')
    flags = []
    if r_f == 0:  flags.append('0frac@R')
    if ts_f == 0: flags.append('0frac@TS')
    if p_f == 0:  flags.append('0frac@P')
    if rev is not None and rev < 0: flags.append(f'neg_rev={rev:.0f}')
    reliable = not flags
    if reliable: n_reliable += 1
    status = 'RELIABLE' if reliable else 'FLAG: ' + ', '.join(flags)
    fwd_s = f'{fwd:.0f}' if fwd is not None else 'N/A'
    print(f'{rxn}: frac(R/TS/P)={r_f}/{ts_f}/{p_f}  fwd={fwd_s} meV  {status}')

print(f'\nTotal reliable: {n_reliable} / {len(TOP10)}')

# ─── CHECK 6: CCSD(T) — all 30 reactions present and valid ──────────────────
print()
print(SEP)
print('CHECK 6: CCSD(T) barriers — all 30 reactions')
print(SEP)

missing, errors = [], []
for rxn in ALL30:
    path = f'{HOME}/mr_benchmark/results/{rxn}_ccsdt.json'
    if not os.path.exists(path):
        missing.append(rxn); print(f'{rxn}: MISSING'); continue
    with open(path) as f:
        d = json.load(f)
    fwd = d.get('barrier_fwd_meV')
    rev = d.get('barrier_rev_meV')
    if fwd is None:
        errors.append(rxn); print(f'{rxn}: ERROR — no barrier_fwd_meV'); continue
    rev_s = f'{rev:.1f}' if rev is not None else 'N/A'
    print(f'{rxn}: fwd={fwd:.1f}  rev={rev_s} meV')

print(f'\nMissing: {missing}')
print(f'Errors:  {errors}')

# ─── CHECK 7: eval_benchmark_sp.json — spot-check rxn7949 ───────────────────
print()
print(SEP)
print('CHECK 7: eval_benchmark_sp.json — cross-check rxn7949 barriers')
print('Barriers = max(relative energy) from the 10-image energy profile')
print(SEP)

with open(f'{HOME}/delta_head/eval_benchmark_sp.json') as f:
    sp_data = {r['rxn']: r for r in json.load(f)['reactions']}

for rxn in ['rxn7949', 'rxn9246']:
    r = sp_data.get(rxn)
    if r is None:
        print(f'{rxn}: not in eval_benchmark_sp.json'); continue
    def rel(key): return (np.array(r[key]) - np.array(r[key])[0]) * 1000
    em = rel('e_wb97m_eV'); ex = rel('e_wb97x_eV')
    emace = rel('e_mace_eV'); ed = rel('e_delta_eV')
    print(f'{rxn}:')
    print(f'  wB97M-V  rel (meV): {np.round(em,1).tolist()}  -> barrier {em.max():.1f}')
    print(f'  wB97X-D3 rel (meV): {np.round(ex,1).tolist()}  -> barrier {ex.max():.1f}')
    print(f'  MACE     rel (meV): {np.round(emace,1).tolist()}  -> barrier {emace.max():.1f}')
    print(f'  delta    rel (meV): {np.round(ed,1).tolist()}  -> barrier {ed.max():.1f}')
    print()

print('Verification complete.')
