
import os, h5py, numpy as np
from collections import defaultdict

H5 = '/home/energy/s242862/data/Transition1x.h5'

def get_rxn_info(h5file, split, rxn_id):
    with h5py.File(h5file, 'r') as f:
        grp = f[split]
        for formula in grp:
            if rxn_id in grp[formula]:
                rxn = grp[formula][rxn_id]
                try:
                    e_ts = float(rxn['transition_state']['wB97x_6-31G(d).energy'][0])
                    e_r  = float(rxn['reactant']['wB97x_6-31G(d).energy'][0])
                    zs   = rxn['transition_state']['atomic_numbers'][:]
                    return formula, e_ts - e_r, len(zs), int(sum(1 for z in zs if z > 1))
                except KeyError:
                    return formula, None, None, None
    return None, None, None, None

def fmax_trajectory(neb_log):
    vals = []
    if not os.path.exists(neb_log):
        return vals
    with open(neb_log) as f:
        for line in f:
            if 'NEBOptimizer' in line:
                try:
                    vals.append(float(line.split()[-1]))
                except:
                    pass
    return vals

def classify_trajectory(vals):
    if not vals:
        return 'crash'
    if len(vals) < 5:
        return 'too_few_steps'
    last10 = vals[-10:] if len(vals) >= 10 else vals
    slope = (last10[-1] - last10[0]) / len(last10)
    osc = max(last10) - min(last10)
    if slope < -0.002 and osc < 0.05:
        return 'still_decreasing'
    elif osc < 0.01:
        return 'hard_plateau'
    elif osc < 0.04:
        return 'soft_plateau'
    else:
        return 'oscillating'

splits = [
    ('test', '/home/energy/s242862/ccsd_dataset/test_reactions.txt', '/home/energy/s242862/orca_neb_results'),
    ('val',  '/home/energy/s242862/ccsd_dataset/val_reactions.txt',  '/home/energy/s242862/orca_neb_val_results'),
]

results = []
for split_name, rxn_file, neb_dir in splits:
    with open(rxn_file) as f:
        rxn_list = [l.strip() for l in f if l.strip()]
    for rxn in rxn_list:
        d = os.path.join(neb_dir, rxn)
        if os.path.exists(os.path.join(d, 'converged')):
            continue
        vals = fmax_trajectory(os.path.join(d, 'neb.log'))
        formula, barrier, n_atoms, n_heavy = get_rxn_info(H5, split_name, rxn)
        results.append({
            'rxn': rxn, 'split': split_name,
            'formula': formula,
            'barrier_eV': barrier,
            'n_atoms': n_atoms,
            'n_heavy': n_heavy,
            'total_steps': len(vals),
            'last_fmax': vals[-1] if vals else None,
            'first_fmax': vals[0] if vals else None,
            'traj_class': classify_trajectory(vals),
            'drop': (vals[0] - vals[-1]) if len(vals) > 1 else None,
        })

by_class = defaultdict(list)
for r in results:
    by_class[r['traj_class']].append(r)

print("=" * 70)
print("TOTAL FAILURES: %d" % len(results))
print("=" * 70)

for cls in ['still_decreasing', 'soft_plateau', 'hard_plateau', 'oscillating', 'crash', 'too_few_steps']:
    rxns = by_class.get(cls, [])
    if not rxns:
        continue
    barriers = [r['barrier_eV']*1000 for r in rxns if r['barrier_eV']]
    sizes = [r['n_heavy'] for r in rxns if r['n_heavy']]
    drops = [r['drop'] for r in rxns if r['drop']]
    print("
--- %s: %d reactions ---" % (cls.upper(), len(rxns)))
    if barriers:
        print("  T1x barrier: %.0f meV mean  (%.0f-%.0f meV range)" % (np.mean(barriers), min(barriers), max(barriers)))
    if sizes:
        print("  Heavy atoms: %.1f mean  (%d-%d range)" % (np.mean(sizes), min(sizes), max(sizes)))
    if drops:
        print("  fmax drop over all steps: %.3f mean" % np.mean(drops))
    for r in sorted(rxns, key=lambda x: x['last_fmax'] if x['last_fmax'] else 99):
        b = "%.0f meV" % (r['barrier_eV']*1000) if r['barrier_eV'] else "N/A"
        lf = "%.3f" % r['last_fmax'] if r['last_fmax'] else "N/A"
        ff = "%.3f" % r['first_fmax'] if r['first_fmax'] else "N/A"
        print("  %s (%s): %d steps  fmax: %s->%s  barrier=%s  formula=%s  heavy=%s" % (
            r['rxn'], r['split'], r['total_steps'], ff, lf, b, r['formula'], r['n_heavy']))
