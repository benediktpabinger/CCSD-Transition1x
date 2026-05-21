"""
Comprehensive 4-way analysis of eval_benchmark_sp.json.
Compares: wB97X-D3 true, MACE, MACE+delta, wB97M-V (reference) on
energies (relative) and forces (absolute) for 30 benchmark reactions.
"""
import json
import numpy as np

with open('eval_benchmark_sp.json') as f:
    data = json.load(f)

rxns = data['reactions']
N = len(rxns)

# -- Collect per-geometry arrays ----------------------------------------------
e_wb97m, e_wb97x, e_mace, e_delta = [], [], [], []
f_wb97m, f_wb97x, f_mace, f_delta = [], [], [], []

for r in rxns:
    ref = np.array(r['e_wb97m_eV'])
    def rel(e): return (np.array(e) - np.array(e)[0]) * 1000

    e_wb97m.extend(rel(r['e_wb97m_eV']))
    e_wb97x.extend(rel(r['e_wb97x_eV']))
    e_mace.extend( rel(r['e_mace_eV']))
    e_delta.extend(rel(r['e_delta_eV']))

    for fi, key in enumerate(['f_wb97m_eV_per_ang','f_wb97x_eV_per_ang',
                               'f_mace_eV_per_ang', 'f_delta_eV_per_ang']):
        for fmat in r[key]:
            if fmat is not None:
                arr = np.array(fmat).flatten() * 1000  # eV/Ang -> meV/Ang
                [f_wb97m, f_wb97x, f_mace, f_delta][fi].extend(arr.tolist())

e_wb97m = np.array(e_wb97m); e_wb97x = np.array(e_wb97x)
e_mace  = np.array(e_mace);  e_delta = np.array(e_delta)
f_wb97m_arr = np.array(f_wb97m); f_wb97x_arr = np.array(f_wb97x)
f_mace_arr  = np.array(f_mace);  f_delta_arr = np.array(f_delta)

METHODS = ['wB97X-D3', 'MACE', 'MACE+delta']
e_preds = [e_wb97x, e_mace, e_delta]
f_preds = [f_wb97x_arr, f_mace_arr, f_delta_arr]

def stats(err):
    return dict(mae=np.abs(err).mean(), rmse=np.sqrt((err**2).mean()),
                bias=err.mean(), std=err.std(),
                p50=np.median(np.abs(err)), p95=np.percentile(np.abs(err), 95))


print('=' * 70)
print(f'BENCHMARK SP EVALUATION  ({N} reactions, {len(e_wb97m)} geoms)')
print('Reference: wB97M-V/def2-TZVP')
print('=' * 70)

# -- 1. Overall energy stats ---------------------------------------------------
print('\n-- RELATIVE ENERGY (meV, ref to first image per reaction) ----------')
print(f'{"Method":15s}  {"MAE":>7}  {"RMSE":>7}  {"Bias":>7}  {"Std":>7}  {"P50":>7}  {"P95":>7}')
print('-' * 65)
for name, ep in zip(METHODS, e_preds):
    err = ep - e_wb97m
    s = stats(err)
    print(f'{name:15s}  {s["mae"]:7.1f}  {s["rmse"]:7.1f}  {s["bias"]:+7.1f}  '
          f'{s["std"]:7.1f}  {s["p50"]:7.1f}  {s["p95"]:7.1f}')

# -- 2. Overall force stats ----------------------------------------------------
print('\n-- FORCES (meV/Ang per component, vs wB97M-V) ----------------------')
print(f'{"Method":15s}  {"MAE":>7}  {"RMSE":>7}  {"Bias":>7}  {"Std":>7}  {"P95":>7}')
print('-' * 57)
for name, fp in zip(METHODS, f_preds):
    n = min(len(fp), len(f_wb97m_arr))
    err = fp[:n] - f_wb97m_arr[:n]
    s = stats(err)
    print(f'{name:15s}  {s["mae"]:7.1f}  {s["rmse"]:7.1f}  {s["bias"]:+7.1f}  '
          f'{s["std"]:7.1f}  {s["p95"]:7.1f}')

# -- 3. Per-reaction energy MAE ------------------------------------------------
print('\n-- PER-REACTION ENERGY MAE (meV) -----------------------------------')
print(f'{"Rxn":12s}  {"wb97x":>7}  {"MACE":>7}  {"delta":>7}  {"best":>10}')
print('-' * 52)
for r in rxns:
    print(f'{r["rxn"]:12s}  {r["emae_wb97x_meV"]:>7.1f}  {r["emae_mace_meV"]:>7.1f}  '
          f'{r["emae_delta_meV"]:>7.1f}  ', end='')
    best = min([('wb97x', r['emae_wb97x_meV']),
                ('mace',  r['emae_mace_meV']),
                ('delta', r['emae_delta_meV'])], key=lambda x: x[1])
    print(f'{best[0]:>10}')

# -- 4. Per-reaction force MAE -------------------------------------------------
print('\n-- PER-REACTION FORCE MAE (meV/Ang) -------------------------------')
print(f'{"Rxn":12s}  {"wb97x":>7}  {"MACE":>7}  {"delta":>7}  {"best":>10}')
print('-' * 52)
for r in rxns:
    fw = r.get('fmae_wb97x_meVA'); fm = r.get('fmae_mace_meVA'); fd = r.get('fmae_delta_meVA')
    if None in (fw, fm, fd): continue
    print(f'{r["rxn"]:12s}  {fw:>7.1f}  {fm:>7.1f}  {fd:>7.1f}  ', end='')
    best = min([('wb97x', fw), ('mace', fm), ('delta', fd)], key=lambda x: x[1])
    print(f'{best[0]:>10}')

# -- 5. Win/loss tally ---------------------------------------------------------
print('\n-- WIN/LOSS: how often each method is best -------------------------')
e_wins = {m: 0 for m in METHODS}
f_wins = {m: 0 for m in METHODS}
for r in rxns:
    candidates_e = [('wB97X-D3', r['emae_wb97x_meV']),
                    ('MACE',     r['emae_mace_meV']),
                    ('MACE+delta', r['emae_delta_meV'])]
    e_wins[min(candidates_e, key=lambda x: x[1])[0]] += 1
    fw = r.get('fmae_wb97x_meVA'); fm = r.get('fmae_mace_meVA'); fd = r.get('fmae_delta_meVA')
    if None not in (fw, fm, fd):
        candidates_f = [('wB97X-D3', fw), ('MACE', fm), ('MACE+delta', fd)]
        f_wins[min(candidates_f, key=lambda x: x[1])[0]] += 1

print(f'{"Method":15s}  {"Energy wins":>12}  {"Force wins":>10}')
print('-' * 42)
for m in METHODS:
    print(f'{m:15s}  {e_wins[m]:>12}  {f_wins[m]:>10}')

# -- 6. Delta improvement over MACE -------------------------------------------
print('\n-- DELTA vs MACE: improvement per reaction -------------------------')
e_improved = sum(1 for r in rxns if r['emae_delta_meV'] < r['emae_mace_meV'])
f_improved = sum(1 for r in rxns
                 if r.get('fmae_delta_meVA') and r.get('fmae_mace_meVA')
                 and r['fmae_delta_meVA'] < r['fmae_mace_meVA'])
e_delta_gain = np.mean([r['emae_mace_meV'] - r['emae_delta_meV'] for r in rxns])
f_delta_gain = np.mean([r['fmae_mace_meVA'] - r['fmae_delta_meVA']
                        for r in rxns if r.get('fmae_mace_meVA') and r.get('fmae_delta_meVA')])
print(f'Energy: delta better in {e_improved}/{N} reactions, mean gain = {e_delta_gain:+.1f} meV')
print(f'Forces: delta better in {f_improved}/{N} reactions, mean gain = {f_delta_gain:+.1f} meV/Ang')

# -- 7. Functional gap (wb97x true vs wb97m) ----------------------------------
print('\n-- FUNCTIONAL GAP: wB97X-D3 true vs wB97M-V -----------------------')
gap_e = e_wb97x - e_wb97m
print(f'  Energy gap MAE:  {np.abs(gap_e).mean():.1f} meV')
print(f'  Energy gap bias: {gap_e.mean():+.1f} meV  (wB97X-D3 systematically {"higher" if gap_e.mean()>0 else "lower"})')
gap_f = f_wb97x_arr - f_wb97m_arr
print(f'  Force  gap MAE:  {np.abs(gap_f).mean():.1f} meV/Ang')
print(f'  Force  gap bias: {gap_f.mean():+.1f} meV/Ang')

print('\n' + '=' * 70)
