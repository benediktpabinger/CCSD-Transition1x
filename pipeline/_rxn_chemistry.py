"""
Extract reaction chemistry (bond type, formula, atoms) for the 23 benchmark reactions
from Transition1x.h5.

Method:
  1. Find each rxn in f['test'] → get molecular formula from HDF5 path
  2. Get atomic_numbers (element types) and R/P positions from the NEB path
     (index 0 = reactant, index -1 = product)
  3. Build connectivity graph at R and P using covalent radii threshold
  4. Diff adjacency → broken bonds at R, formed bonds at P
  5. Classify reaction type from the set of changed bonds
"""
import sys
import h5py
import numpy as np

H5 = 'data/Transition1x.h5'

RXNS = [
    'rxn7949', 'rxn8832', 'rxn8885', 'rxn7945', 'rxn6196',
    'rxn3107', 'rxn7936', 'rxn7957', 'rxn7937', 'rxn0346',
    'rxn7060', 'rxn1320', 'rxn1147', 'rxn1150',
    'rxn0896', 'rxn8827', 'rxn10005', 'rxn8837',
    'rxn4518', 'rxn0101', 'rxn4522', 'rxn10054', 'rxn4113',
]

# Covalent radii in Angstrom (Alvarez 2008, standard values)
COV_RAD = {
    1:  0.31,   # H
    5:  0.84,   # B
    6:  0.76,   # C
    7:  0.71,   # N
    8:  0.66,   # O
    9:  0.57,   # F
    14: 1.11,   # Si
    15: 1.07,   # P
    16: 1.05,   # S
    17: 1.02,   # Cl
    35: 1.20,   # Br
}
SYMBOLS = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F',
           14:'Si', 15:'P', 16:'S', 17:'Cl', 35:'Br'}

SCALE = 1.25   # bond if dist < SCALE * (r_i + r_j)


def connectivity(atomic_numbers, positions):
    n = len(atomic_numbers)
    adj = set()
    for i in range(n):
        ri = COV_RAD.get(atomic_numbers[i], 0.8)
        for j in range(i+1, n):
            rj = COV_RAD.get(atomic_numbers[j], 0.8)
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < SCALE * (ri + rj):
                adj.add((i, j))
    return adj


def bond_label(atomic_numbers, i, j):
    si = SYMBOLS.get(atomic_numbers[i], '?')
    sj = SYMBOLS.get(atomic_numbers[j], '?')
    # canonical order
    pair = tuple(sorted([si, sj]))
    return f'{pair[0]}-{pair[1]}'


def classify(broken, formed, atomic_numbers):
    bl = sorted(set(broken))
    fl = sorted(set(formed))
    all_changed = sorted(set(bl + fl))
    return bl, fl, all_changed


with h5py.File(H5, 'r') as f:
    split = f['test']

    # build rxn → formula map
    rxn_to_formula = {}
    for formula in split:
        for rxn in split[formula]:
            if rxn in RXNS:
                rxn_to_formula[rxn] = formula

    print(f"{'rxn':10s} {'formula':16s} {'n_atoms':7s} {'bonds broken':30s} {'bonds formed':30s}")
    print('-' * 100)

    for rxn in RXNS:
        formula = rxn_to_formula.get(rxn)
        if formula is None:
            print(f'{rxn:10s} NOT FOUND')
            continue

        grp = split[formula][rxn]
        atomic_numbers = grp['atomic_numbers'][:]

        # prefer dedicated R/P subgroups; fall back to NEB path endpoints
        def get_pos(key):
            if key in grp and hasattr(grp[key], 'keys') and 'positions' in grp[key]:
                p = grp[key]['positions'][:]
                return p[0] if p.ndim == 3 else p   # (1, n, 3) → (n, 3)
            return None

        r_sub = get_pos('reactant')
        p_sub = get_pos('product')
        positions = grp['positions'][:]         # shape (n_configs, n_atoms, 3)

        pos_r = r_sub if r_sub is not None else positions[0]
        pos_p = p_sub if p_sub is not None else positions[-1]
        n = len(atomic_numbers)

        adj_r = connectivity(atomic_numbers, pos_r)
        adj_p = connectivity(atomic_numbers, pos_p)

        broken_idx = adj_r - adj_p   # in R but not P
        formed_idx = adj_p - adj_r   # in P but not R

        broken = [bond_label(atomic_numbers, i, j) for i,j in broken_idx]
        formed = [bond_label(atomic_numbers, i, j) for i,j in formed_idx]

        bl = ', '.join(sorted(set(broken))) if broken else '—'
        fl = ', '.join(sorted(set(formed))) if formed else '—'

        # detail: count each type
        from collections import Counter
        bc = Counter(broken)
        fc = Counter(formed)
        bc_str = ' '.join(f'{k}x{v}' if v>1 else k for k,v in sorted(bc.items()))
        fc_str = ' '.join(f'{k}x{v}' if v>1 else k for k,v in sorted(fc.items()))

        print(f'{rxn:10s} {formula:16s} {n:7d}   broken: {bc_str or "-":28s} formed: {fc_str or "-"}')
