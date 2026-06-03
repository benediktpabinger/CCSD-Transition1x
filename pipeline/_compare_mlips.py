import json, numpy as np

with open('full_benchmark_results.json') as f:
    rxns = json.load(f)['reactions']

TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
MIDDLE10 = {'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955','rxn4519','rxn4500','rxn2553','rxn8829','rxn1155'}
BOTTOM10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}

def mae(a):  return np.abs(a).mean()
def rmse(a): return np.sqrt((a**2).mean())
def bias(a): return a.mean()

groups = [
    ('Low MR',  BOTTOM10),
    ('Mid MR',  MIDDLE10),
    ('High MR', TOP10),
    ('All 30',  TOP10 | MIDDLE10 | BOTTOM10),
]

METHODS = [
    ('eSEN (OMol25)',   'esen_neb_fwd_meV'),
    ('UMA-s',          'uma_neb_fwd_meV'),
    ('wB97M-V (ORCA)', 'neb_wb97m_fwd_meV'),
    ('MACE+delta',     'delta_fwd_meV'),
    ('MACE',           'mace_fwd_meV'),
]

# ── vs CCSD(T) ───────────────────────────────────────────────────────────────
print('Method comparison vs CCSD(T) (meV, forward barrier)')
print()

for mname, key in METHODS:
    print(f'{mname}')
    print(f'  {"Group":<10} {"MAE":>10} {"Bias":>8} {"RMSE":>8}')
    print(f'  {"-"*38}')
    for glabel, gset in groups:
        ref = [r for r in rxns if r['rxn'] in gset and r.get('ccsdt_fwd_meV') and r.get(key)]
        if not ref: continue
        err = np.array([r[key] - r['ccsdt_fwd_meV'] for r in ref])
        print(f'  {glabel:<10} {mae(err):>8.1f}   {bias(err):>+7.1f}  {rmse(err):>8.1f}')
    print()

# ── vs wB97M-V ───────────────────────────────────────────────────────────────
print()
print('eSEN and UMA vs wB97M-V (same functional — geometry effect only)')
print()
for mname, key in [('eSEN', 'esen_neb_fwd_meV'), ('UMA-s', 'uma_neb_fwd_meV')]:
    print(f'{mname}')
    print(f'  {"Group":<10} {"MAE":>10} {"Bias":>8} {"RMSE":>8}')
    print(f'  {"-"*38}')
    for glabel, gset in groups:
        pairs = [(r[key], r['neb_wb97m_fwd_meV']) for r in rxns
                 if r['rxn'] in gset and r.get(key) and r.get('neb_wb97m_fwd_meV')]
        if not pairs: continue
        pred, ref = zip(*pairs)
        err = np.array(pred) - np.array(ref)
        print(f'  {glabel:<10} {mae(err):>8.1f}   {bias(err):>+7.1f}  {rmse(err):>8.1f}')
    print()
