"""
Analysis of eSEN NEB results vs CCSD(T), wB97M-V, and MACE.
Reads from full_benchmark_results.json (local copy).
"""
import json
import numpy as np

with open('full_benchmark_results.json') as f:
    rxns = json.load(f)['reactions']

TOP10    = {'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885','rxn7945','rxn7937','rxn6196','rxn0346','rxn1150'}
MIDDLE10 = {'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955','rxn4519','rxn4500','rxn2553','rxn8829','rxn1155'}
BOTTOM10 = {'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004','rxn4063','rxn4114','rxn4060','rxn1961','rxn1962'}

def group(rxn): return 'High' if rxn in TOP10 else ('Mid' if rxn in MIDDLE10 else 'Low')

def mae(a):  return np.abs(a).mean()
def rmse(a): return np.sqrt((a**2).mean())
def bias(a): return a.mean()

SEP = '=' * 90

# ── 1. Full table ────────────────────────────────────────────────────────────
print(SEP)
print('eSEN NEB RESULTS — all 30 reactions')
print(f'{"Rxn":<12} {"Grp":<5} {"eSEN fwd":>9} {"wB97M fwd":>10} {"CCSD(T)":>8} {"MACE":>7} '
      f'{"eSEN-CCT":>9} {"eSEN-M":>8} {"fmax":>7} {"flag"}')
print('-' * 90)

flags = []
for r in rxns:
    rxn   = r['rxn']
    grp   = group(rxn)
    esen  = r.get('esen_neb_fwd_meV')
    nebm  = r.get('neb_wb97m_fwd_meV')
    cct   = r.get('ccsdt_fwd_meV')
    mace  = r.get('mace_fwd_meV')
    fmax  = r.get('esen_neb_fmax')

    if esen is None:
        print(f'{rxn:<12} {grp:<5} {"N/A":>9}')
        continue

    diff_cct  = f'{esen - cct:+.0f}'  if cct  else 'N/A'
    diff_mace = f'{esen - mace:+.0f}' if mace else 'N/A'
    nebm_s    = f'{nebm:.0f}' if nebm else 'N/A'
    cct_s     = f'{cct:.0f}'  if cct  else 'N/A'
    mace_s    = f'{mace:.0f}' if mace else 'N/A'
    fmax_s    = f'{fmax:.3f}' if fmax else '?'

    flag = ''
    if fmax and fmax > 0.10: flag = f'HIGH FMAX={fmax:.2f}'
    elif fmax and fmax > 0.05: flag = f'fmax={fmax:.3f}'
    if flag: flags.append(f'{rxn}: {flag}')

    print(f'{rxn:<12} {grp:<5} {esen:>9.0f} {nebm_s:>10} {cct_s:>8} {mace_s:>7} '
          f'{diff_cct:>9} {diff_mace:>8} {fmax_s:>7}  {flag}')

# ── 2. Method comparison vs CCSD(T) ─────────────────────────────────────────
print()
print(SEP)
print('METHOD COMPARISON vs CCSD(T) (meV, forward barrier)')
print()

METHODS = [
    ('eSEN (OMol25)',    'esen_neb_fwd_meV'),
    ('wB97M-V (NEB)',    'neb_wb97m_fwd_meV'),
    ('MACE (wB97X-D3)', 'mace_fwd_meV'),
    ('MACE+delta',      'delta_fwd_meV'),
]

groups = [
    ('All 30',   rxns),
    ('High MR',  [r for r in rxns if r['rxn'] in TOP10]),
    ('Mid MR',   [r for r in rxns if r['rxn'] in MIDDLE10]),
    ('Low MR',   [r for r in rxns if r['rxn'] in BOTTOM10]),
]

for glabel, grxns in groups:
    ref_avail = [r for r in grxns if r.get('ccsdt_fwd_meV') is not None]
    print(f'  [{glabel}, n={len(ref_avail)} with CCSD(T)]')
    print(f'  {"Method":<20}  {"n":>3}  {"MAE":>7}  {"RMSE":>7}  {"Bias":>8}')
    print('  ' + '-' * 45)
    for mname, key in METHODS:
        pairs = [(r[key], r['ccsdt_fwd_meV']) for r in ref_avail if r.get(key) is not None]
        if not pairs: continue
        pred, ref = zip(*pairs)
        err = np.array(pred) - np.array(ref)
        print(f'  {mname:<20}  {len(err):>3}  {mae(err):>7.1f}  {rmse(err):>7.1f}  {bias(err):>+8.1f}')
    print()

# ── 3. eSEN vs wB97M-V ───────────────────────────────────────────────────────
print(SEP)
print('eSEN vs wB97M-V (same functional level — geometry difference only)')
print()
for glabel, grxns in groups:
    pairs = [(r['esen_neb_fwd_meV'], r['neb_wb97m_fwd_meV'])
             for r in grxns if r.get('esen_neb_fwd_meV') and r.get('neb_wb97m_fwd_meV')]
    if not pairs: continue
    pred, ref = zip(*pairs)
    err = np.array(pred) - np.array(ref)
    print(f'  {glabel:<10}: n={len(err)}  MAE={mae(err):.1f}  Bias={bias(err):+.1f}  RMSE={rmse(err):.1f} meV')

# ── 4. Convergence flags ─────────────────────────────────────────────────────
print()
print(SEP)
print('CONVERGENCE FLAGS (fmax > 0.05)')
for f in flags:
    print(f'  {f}')
if not flags:
    print('  None.')
