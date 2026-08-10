"""Model transition states scored against the corrected (broken-symmetry)
reference instead of the RKS one.

Primary metric: the reactive coordinate -- the largest deviation in the two
bonds being broken and formed, which are what define a transition state. They
sit at 2-3 A at the saddle, so a covalent-radius bond list misses them; they are
read per reaction from the reactive_bonds field written by the TS optimisation.
Threshold 0.1 A: below that lies the scatter between functionals, above it the
difference is chemically real.

Secondary: all-atom Kabsch RMSD, the metric the benchmark already uses, so the
numbers connect to everything reported so far and the switch cannot be mistaken
for a metric chosen after the fact.

Carried along as diagnostics, not as criteria: heavy-atom RMSD and the largest
covalent bond-length difference. Where these disagree with the primary metric,
the disagreement is itself informative.

Only the 13 reactions with a frequency-confirmed BS transition state are used.
"""
import glob
import json
import os

import numpy as np
from ase.data import atomic_numbers, covalent_radii

H = '/home/energy/s242862'
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results', 'MACE': 'mace_bare_neb_results',
          'MACE+delta': 'mace_delta_neb_results_fw2'}
CONFIRMED = ['rxn0346', 'rxn0894', 'rxn1147', 'rxn1320', 'rxn4518', 'rxn5691',
             'rxn7949', 'rxn8827', 'rxn8837', 'rxn3107', 'rxn7957', 'rxn8832',
             'rxn8885']
THR_RC = 0.10       # reactive coordinate, primary
THR_RMSD = 0.30     # all-atom, the benchmark's existing threshold


def read_xyz(p):
    L = open(p).read().split('\n')
    n = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + n]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def kabsch(A, B, sel=None):
    if sel is None:
        sel = np.arange(len(A))
    Ac, Bc = A - A[sel].mean(0), B - B[sel].mean(0)
    V, S, W = np.linalg.svd(Ac[sel].T @ Bc[sel])
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ W))])
    d = (Ac @ (V @ D @ W))[sel] - Bc[sel]
    return float(np.sqrt((d ** 2).sum() / len(sel)))


def cov_bond_max(sym, x1, x2, scale=1.3):
    worst = 0.0
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            rc = scale * (covalent_radii[atomic_numbers[sym[i]]]
                          + covalent_radii[atomic_numbers[sym[j]]])
            d1 = float(np.linalg.norm(x1[i] - x1[j]))
            d2 = float(np.linalg.norm(x2[i] - x2[j]))
            if min(d1, d2) < rc:
                worst = max(worst, abs(d1 - d2))
    return worst


def reactive_pairs(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def bs_ts(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    return f
    return None


def rc_err(pairs, xa, xb):
    """Largest deviation over the reactive bonds."""
    if not pairs:
        return None
    return max(abs(float(np.linalg.norm(xa[i] - xa[j]))
                   - float(np.linalg.norm(xb[i] - xb[j])))
               for i, j, _ in pairs)


rows = []
skipped = []
for rx in CONFIRMED:
    bs = bs_ts(rx)
    ref = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
    pairs = reactive_pairs(rx)
    if not (bs and os.path.exists(ref) and pairs):
        skipped.append((rx, f'bs={bool(bs)} ref={os.path.exists(ref)} '
                            f'pairs={len(pairs)}'))
        continue
    s_bs, x_bs = read_xyz(bs)
    s_rf, x_rf = read_xyz(ref)
    heavy = np.array([i for i, s in enumerate(s_bs) if s != 'H'])
    for m, dn in MODELS.items():
        p = f'{H}/{dn}/{rx}/transition_state.xyz'
        if not os.path.exists(p):
            continue
        s_m, x_m = read_xyz(p)
        if s_m != s_bs:
            continue
        rows.append({
            'rxn': rx, 'model': m,
            'rc_ref': rc_err(pairs, x_m, x_rf),
            'rc_bs': rc_err(pairs, x_m, x_bs),
            'rmsd_ref': kabsch(x_m, x_rf), 'rmsd_bs': kabsch(x_m, x_bs),
            'heavy_ref': kabsch(x_m, x_rf, heavy),
            'heavy_bs': kabsch(x_m, x_bs, heavy),
            'cov_ref': cov_bond_max(s_bs, x_m, x_rf),
            'cov_bs': cov_bond_max(s_bs, x_m, x_bs),
        })

print(f'{len(rows)} Zeilen aus {len(set(r["rxn"] for r in rows))} Reaktionen')
if skipped:
    print('uebersprungen:', skipped)

print('\n' + '=' * 96)
print('PRIMAER — reaktive Koordinate [A], Schwelle 0.10')
print('=' * 96)
print(f"{'rxn':<10}" + ''.join(f'{m:>17}' for m in MODELS))
print(f"{'':<10}" + ''.join(f'{"vs RKS / vs BS":>17}' for _ in MODELS))
for rx in CONFIRMED:
    sub = {r['model']: r for r in rows if r['rxn'] == rx}
    if not sub:
        continue
    line = f'{rx:<10}'
    for m in MODELS:
        r = sub.get(m)
        if r is None:
            cell = '--'
        else:
            cell = '{:.3f}/{:.3f}'.format(r['rc_ref'], r['rc_bs'])
        line += '{:>17}'.format(cell)
    print(line)

print('\n' + '=' * 70)
print('ZUSAMMENFASSUNG je Modell')
print('=' * 70)
print(f"{'Modell':<12}{'RC vs RKS':>22}{'RC vs BS':>22}")
print(f"{'':<12}{'median   >0.1':>22}{'median   >0.1':>22}")
for m in MODELS:
    s = [r for r in rows if r['model'] == m]
    if not s:
        continue
    a = np.array([r['rc_ref'] for r in s])
    b = np.array([r['rc_bs'] for r in s])
    print(f'{m:<12}'
          f"{f'{np.median(a):.4f}   {int((a>THR_RC).sum())}/{len(s)}':>22}"
          f"{f'{np.median(b):.4f}   {int((b>THR_RC).sum())}/{len(s)}':>22}")

print(f"\n{'Modell':<12}{'RMSD vs RKS':>22}{'RMSD vs BS':>22}")
print(f"{'':<12}{'median   >0.3':>22}{'median   >0.3':>22}")
for m in MODELS:
    s = [r for r in rows if r['model'] == m]
    if not s:
        continue
    a = np.array([r['rmsd_ref'] for r in s])
    b = np.array([r['rmsd_bs'] for r in s])
    print(f'{m:<12}'
          f"{f'{np.median(a):.4f}   {int((a>THR_RMSD).sum())}/{len(s)}':>22}"
          f"{f'{np.median(b):.4f}   {int((b>THR_RMSD).sum())}/{len(s)}':>22}")

print('\n=== Wird das Modell besser, wenn man gegen die BS-Referenz misst? ===')
for m in MODELS:
    s = [r for r in rows if r['model'] == m]
    if not s:
        continue
    better_rc = sum(1 for r in s if r['rc_bs'] < r['rc_ref'])
    better_rm = sum(1 for r in s if r['rmsd_bs'] < r['rmsd_ref'])
    print(f'  {m:<12} reaktive Koordinate {better_rc}/{len(s)}   '
          f'All-Atom-RMSD {better_rm}/{len(s)}')

print('\n=== Diagnostik: wo die Masse auseinanderlaufen (gegen BS) ===')
print(f"{'rxn':<10}{'Modell':<12}{'RC':>8}{'RMSD':>9}{'schwer':>9}{'kov.':>8}")
for r in sorted(rows, key=lambda x: -abs(x['rmsd_bs'] - x['rc_bs']))[:10]:
    print(f"{r['rxn']:<10}{r['model']:<12}{r['rc_bs']:>8.3f}"
          f"{r['rmsd_bs']:>9.3f}{r['heavy_bs']:>9.3f}{r['cov_bs']:>8.3f}")

json.dump(rows, open(f'{H}/model_vs_bsts.json', 'w'), indent=1)
print(f'\ngeschrieben: {H}/model_vs_bsts.json')
