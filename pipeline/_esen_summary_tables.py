import json, numpy as np

with open('full_benchmark_results.json') as f:
    rxns = json.load(f)['reactions']

TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
MIDDLE10 = {'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955','rxn4519','rxn4500','rxn2553','rxn8829','rxn1155'}
BOTTOM10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}

def grp(rxn): return 'High MR' if rxn in TOP10 else ('Mid MR' if rxn in MIDDLE10 else 'Low MR')
def mae(a):  return np.abs(a).mean()
def rmse(a): return np.sqrt((a**2).mean())
def bias(a): return a.mean()

METHODS = [
    ('eSEN (OMol25)',  'esen_neb_fwd_meV'),
    ('wB97M-V (ORCA)', 'neb_wb97m_fwd_meV'),
]

groups = [
    ('Low MR',  BOTTOM10),
    ('Mid MR',  MIDDLE10),
    ('High MR', TOP10),
    ('All 30',  TOP10 | MIDDLE10 | BOTTOM10),
]

# ── Table 1: barrier MAE vs CCSD(T) ─────────────────────────────────────────
print('Table 1: Barrier MAE vs CCSD(T) (meV, forward)')
print(f'{"Group":<10} {"Method":<20} {"n":>3} {"MAE":>8} {"Bias":>8} {"RMSE":>8}')
print('-' * 62)

for glabel, gset in groups:
    ref = [r for r in rxns if r['rxn'] in gset and r.get('ccsdt_fwd_meV')]
    for mname, key in METHODS:
        pairs = [(r[key], r['ccsdt_fwd_meV']) for r in ref if r.get(key)]
        if not pairs: continue
        pred, cct = zip(*pairs)
        err = np.array(pred) - np.array(cct)
        print(f'{glabel:<10} {mname:<20} {len(err):>3} {mae(err):>8.1f} {bias(err):>+8.1f} {rmse(err):>8.1f}')
    print()

# ── Table 2: eSEN vs wB97M-V directly ───────────────────────────────────────
print('Table 2: eSEN vs wB97M-V NEB (meV, forward) — same functional level')
print(f'{"Group":<10} {"n":>3} {"MAE":>8} {"Bias":>8} {"RMSE":>8}')
print('-' * 42)

for glabel, gset in groups:
    pairs = [(r['esen_neb_fwd_meV'], r['neb_wb97m_fwd_meV'])
             for r in rxns if r['rxn'] in gset
             and r.get('esen_neb_fwd_meV') and r.get('neb_wb97m_fwd_meV')]
    if not pairs: continue
    pred, ref = zip(*pairs)
    err = np.array(pred) - np.array(ref)
    print(f'{glabel:<10} {len(err):>3} {mae(err):>8.1f} {bias(err):>+8.1f} {rmse(err):>8.1f}')
