"""
Comprehensive local analysis of full_benchmark_results.json.
Covers all 30 benchmark reactions grouped by MR character (High/Mid/Low).

Sections:
1. RMSD of TS geometries (T1x vs NEB wB97M-V)
2. Barrier heights — all methods vs CCSD(T)
3. Functional-gap decomposition
4. NEVPT2 agreement with CCSD(T) (reliable subset)
5. Per-group summaries
"""
import json
import numpy as np

with open('full_benchmark_results.json') as f:
    data = json.load(f)

rxns = data['reactions']

TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
            'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
BOTTOM10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
            'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}
MIDDLE10 = {'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
            'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155'}

groups = {
    'High MR': [r for r in rxns if r['rxn'] in TOP10],
    'Low MR':  [r for r in rxns if r['rxn'] in BOTTOM10],
    'Mid MR':  [r for r in rxns if r['rxn'] in MIDDLE10],
}

def mae(arr):  return np.abs(arr).mean()
def rmse(arr): return np.sqrt((arr**2).mean())
def bias(arr): return arr.mean()

def stats_str(pred, ref):
    err = np.array(pred) - np.array(ref)
    return f'MAE={mae(err):.0f}  RMSE={rmse(err):.0f}  Bias={bias(err):+.0f}'

def gather(rxn_list, key):
    return [(r[key], r) for r in rxn_list if r.get(key) is not None]

SEP = '=' * 110


# ===========================================================================
print(SEP)
print('FULL BENCHMARK ANALYSIS — 30 REACTIONS (grouped by MR character)')
print(SEP)


# ---------------------------------------------------------------------------
# 1. RMSD of TS geometries
# ---------------------------------------------------------------------------
print('\n--- 1. RMSD OF TS GEOMETRIES: T1x wB97X-D3 vs NEB wB97M-V (Angstrom) ---')
print(f'{"Rxn":12s}  {"Group":7s}  {"RMSD (A)":>9}')
print('-' * 35)
for label, grp in groups.items():
    for r in grp:
        rmsd = r.get('ts_rmsd_ang')
        rmsd_s = f'{rmsd:.4f}' if rmsd is not None else 'N/A'
        print(f'{r["rxn"]:12s}  {label:7s}  {rmsd_s:>9}')
    rmsds = [r['ts_rmsd_ang'] for r in grp if r.get('ts_rmsd_ang') is not None]
    print(f'  --> {label}: mean RMSD = {np.mean(rmsds):.4f} A  max = {np.max(rmsds):.4f} A')
    print()


# ---------------------------------------------------------------------------
# 2. Barrier heights — all methods
# ---------------------------------------------------------------------------
print('\n--- 2. BARRIER HEIGHTS (meV, forward) — ALL METHODS ---')
print(f'{"Rxn":12s} {"Grp":5s} {"T1x-X":>7} {"NEB-M":>7} {"NEB-X":>7} {"CCSD(T)":>8} {"NEVPT2":>7} {"MACE":>6} {"delta":>6}')
print('-' * 70)
for label, grp in groups.items():
    print(f'\n  [{label}]')
    for r in grp:
        nev = r.get('nevpt2_fwd_meV')
        nev_s = f'{nev:7.0f}' if nev is not None else '    N/A'
        def f(k, w=7):
            v = r.get(k)
            return f'{v:>{w}.0f}' if v is not None else f'{"N/A":>{w}}'
        print(f'  {r["rxn"]:12s} {label[:3]:5s} {f("t1x_wb97x_fwd_meV")} {f("neb_wb97m_fwd_meV")} '
              f'{f("neb_wb97x_fwd_meV")} {f("ccsdt_fwd_meV",8)} {nev_s} '
              f'{f("mace_fwd_meV",6)} {f("delta_fwd_meV",6)}')


# ---------------------------------------------------------------------------
# 3. Method comparison vs CCSD(T) — per group and overall
# ---------------------------------------------------------------------------
METHODS = [
    ('wB97M-V (NEB)',  'neb_wb97m_fwd_meV'),
    ('wB97X-D3 (NEB)', 'neb_wb97x_fwd_meV'),
    ('wB97X-D3 (T1x)', 't1x_wb97x_fwd_meV'),
    ('MACE',           'mace_fwd_meV'),
    ('MACE+delta',     'delta_fwd_meV'),
    ('NEVPT2',         'nevpt2_fwd_meV'),
]

def compare_vs_ccsdt(rxn_list, label):
    ref_avail = [r for r in rxn_list if r.get('ccsdt_fwd_meV') is not None]
    n = len(ref_avail)
    if n == 0:
        return
    print(f'\n  [{label}, n={n}]')
    print(f'  {"Method":20s}  {"n":>3}  {"MAE":>7}  {"RMSE":>7}  {"Bias":>8}')
    print('  ' + '-' * 50)
    for mname, key in METHODS:
        pairs = [(r[key], r['ccsdt_fwd_meV']) for r in ref_avail if r.get(key) is not None]
        if not pairs:
            continue
        pred, ref = zip(*pairs)
        err = np.array(pred) - np.array(ref)
        print(f'  {mname:20s}  {len(err):>3}  {mae(err):>7.1f}  {rmse(err):>7.1f}  {bias(err):>+8.1f}')

print('\n\n--- 3. METHOD COMPARISON vs CCSD(T) (meV) ---')
compare_vs_ccsdt(rxns, 'All 30')
compare_vs_ccsdt(groups['High MR'], 'High MR (top 10)')
compare_vs_ccsdt(groups['Low MR'],  'Low MR (bottom 10)')
compare_vs_ccsdt(groups['Mid MR'],  'Mid MR (middle 10)')


# ---------------------------------------------------------------------------
# 4. Functional-gap decomposition
# ---------------------------------------------------------------------------
print('\n\n--- 4. FUNCTIONAL-GAP DECOMPOSITION ---')
print('  Geometry effect:  wB97X-D3(T1x) - wB97X-D3(NEB)  [geometry re-optimisation]')
print('  Functional effect: wB97X-D3(NEB) - wB97M-V(NEB)   [functional change at same geom]')
print()

all_with_both = [r for r in rxns if all(r.get(k) for k in
                 ['t1x_wb97x_fwd_meV','neb_wb97x_fwd_meV','neb_wb97m_fwd_meV'])]

geom_effect = np.array([r['t1x_wb97x_fwd_meV'] - r['neb_wb97x_fwd_meV'] for r in all_with_both])
func_effect = np.array([r['neb_wb97x_fwd_meV'] - r['neb_wb97m_fwd_meV'] for r in all_with_both])
total_effect = np.array([r['t1x_wb97x_fwd_meV'] - r['neb_wb97m_fwd_meV'] for r in all_with_both])

print(f'  {"Effect":30s}  {"Mean":>8}  {"Std":>8}  {"Min":>8}  {"Max":>8}')
print('  ' + '-' * 68)
for name, arr in [('Geometry (T1x->NEB)', geom_effect),
                  ('Functional (wB97X->wB97M)', func_effect),
                  ('Total (T1x-X -> NEB-M)', total_effect)]:
    print(f'  {name:30s}  {arr.mean():>+8.1f}  {arr.std():>8.1f}  {arr.min():>+8.1f}  {arr.max():>+8.1f}')

# per group
for label, grp in groups.items():
    subset = [r for r in grp if all(r.get(k) for k in
              ['t1x_wb97x_fwd_meV','neb_wb97x_fwd_meV','neb_wb97m_fwd_meV'])]
    if not subset:
        continue
    g = np.array([r['t1x_wb97x_fwd_meV'] - r['neb_wb97x_fwd_meV'] for r in subset])
    f = np.array([r['neb_wb97x_fwd_meV'] - r['neb_wb97m_fwd_meV'] for r in subset])
    print(f'\n  {label}: geom={g.mean():+.0f}+/-{g.std():.0f}  func={f.mean():+.0f}+/-{f.std():.0f} meV')


# ---------------------------------------------------------------------------
# 5. NEVPT2 vs CCSD(T) (reliable top-10 subset)
# ---------------------------------------------------------------------------
nev_reliable = [r for r in rxns if r.get('nevpt2_reliable') and r.get('nevpt2_fwd_meV') and r.get('ccsdt_fwd_meV')]
print(f'\n\n--- 5. NEVPT2 vs CCSD(T) ({len(nev_reliable)} reliable reactions) ---')
print(f'{"Rxn":12s}  {"CCSD(T)":>8}  {"NEVPT2":>7}  {"diff":>7}')
print('-' * 40)
diffs = []
for r in nev_reliable:
    diff = r['nevpt2_fwd_meV'] - r['ccsdt_fwd_meV']
    diffs.append(diff)
    print(f'{r["rxn"]:12s}  {r["ccsdt_fwd_meV"]:>8.0f}  {r["nevpt2_fwd_meV"]:>7.0f}  {diff:>+7.0f}')
diffs = np.array(diffs)
print(f'\n  MAE={mae(diffs):.0f}  RMSE={rmse(diffs):.0f}  Bias={bias(diffs):+.0f} meV')


# ---------------------------------------------------------------------------
# 6. RMSD outlier note
# ---------------------------------------------------------------------------
print('\n\n--- 6. GEOMETRY QUALITY CHECK ---')
outliers = [(r['rxn'], r['ts_rmsd_ang']) for r in rxns
            if r.get('ts_rmsd_ang') is not None and r['ts_rmsd_ang'] > 0.05]
if outliers:
    print(f'  Reactions with TS RMSD > 0.05 A (possible conformational change):')
    for rxn, rm in sorted(outliers, key=lambda x: -x[1]):
        print(f'    {rxn}: {rm:.4f} A')
else:
    print('  All TS RMSDs < 0.05 A -- geometries tightly clustered.')

print('\n' + SEP)
