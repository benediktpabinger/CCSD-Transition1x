"""
Comprehensive benchmark analysis — all 30 reactions, 3 parts.

DATA SOURCES
------------
full_benchmark_results.json  — barrier heights from all methods on stationary points
eval_benchmark_sp.json       — per-image energies + forces (last 10 NEB images, all 4 methods)

REFERENCE CHOICES
-----------------
Part 1  RMSD comparison:
        T1x wB97X-D3 optimised TS  vs  NEB wB97M-V optimised TS
        Both are stationary-point geometries; RMSD measures how much the TS
        moved when re-optimised at the higher level.

Part 2  Barrier heights:
        All barriers are *forward* (TS − reactant) in meV.
        Gold standard = CCSD(T)/def2-TZVP on NEB wB97M-V stationary points.
        Errors are (method − CCSD(T)).  Positive = overestimate.
        Two sub-questions:
          2a) Geometry effect   — same functional (wB97X-D3), T1x geom vs NEB geom
          2b) Method comparison — all methods on NEB geoms vs CCSD(T)

Part 3  Single-point energies + forces on 10 NEB images per reaction:
        Reference = wB97M-V/def2-TZVP (highest quality, same geometries).
        Methods compared:
          wB97X-D3   — the functional MACE was trained on (functional gap baseline)
          MACE       — ML surrogate of wB97X-D3, no correction
          MACE+delta — ML surrogate with delta-learning correction toward wB97M-V
        wB97M-V vs itself is trivially zero and excluded from error tables.

        ENERGY METRICS (relative per reaction, ref to first image)
          MAE          — mean absolute error of relative energies (meV)
          R^2          — profile shape agreement; 1 = perfect shape, 0 = flat line
          TS image     — fraction of reactions where argmax(E_pred) == argmax(E_ref)
          TS |delta E| — mean |error| at the TS image specifically

        FORCE METRICS (per Cartesian component, vs wB97M-V)
          MAE          — mean absolute error (meV/A)
          Cosine sim   — mean cos(angle) between predicted and reference force vectors
                         per atom; 1 = same direction, 0 = perpendicular, -1 = reversed
                         Most critical for NEB: wrong direction => wrong path
          Mag ratio    — mean |F_pred| / |F_ref| per atom; 1 = same magnitude
                         >1 = overestimates stiffness, <1 = underestimates
          TS force norm— RMS force at the wB97M-V TS image (should be ~0 at saddle point)
                         Lower = method sees a genuine saddle near the reference TS
"""
import json
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────
with open('full_benchmark_results.json') as f:
    bm = {r['rxn']: r for r in json.load(f)['reactions']}

with open('eval_benchmark_sp.json') as f:
    sp_raw = json.load(f)['reactions']
sp = {r['rxn']: r for r in sp_raw}

TOP10    = ['rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
            'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150']
BOTTOM10 = ['rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
            'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962']
MIDDLE10 = ['rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
            'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155']
ALL30    = TOP10 + BOTTOM10 + MIDDLE10

GROUPS = [('High MR', TOP10), ('Low MR', BOTTOM10), ('Mid MR', MIDDLE10)]
GRP_LABEL = {rxn: g for g, rxns in GROUPS for rxn in rxns}

SEP  = '=' * 100
SEP2 = '-' * 100

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def nz(arr): return np.abs(arr).mean()
def rms(arr): return np.sqrt((arr**2).mean())

# ── Part 3 helpers ───────────────────────────────────────────────────────────

def cosine_sim(f_pred_imgs, f_ref_imgs):
    """Mean cosine similarity per atom across all images. Skip near-zero forces."""
    sims = []
    for fp_img, fr_img in zip(f_pred_imgs, f_ref_imgs):
        if fp_img is None or fr_img is None:
            continue
        fp = np.array(fp_img)   # (N, 3)
        fr = np.array(fr_img)
        norm_p = np.linalg.norm(fp, axis=1)
        norm_r = np.linalg.norm(fr, axis=1)
        valid = (norm_p > 1e-6) & (norm_r > 1e-6)
        if not valid.any():
            continue
        dots = np.sum(fp[valid] * fr[valid], axis=1)
        cos  = dots / (norm_p[valid] * norm_r[valid])
        sims.extend(cos.tolist())
    return np.mean(sims) if sims else np.nan

def mag_ratio(f_pred_imgs, f_ref_imgs):
    """Mean |F_pred| / |F_ref| per atom. Excludes near-zero reference forces."""
    ratios = []
    for fp_img, fr_img in zip(f_pred_imgs, f_ref_imgs):
        if fp_img is None or fr_img is None:
            continue
        fp = np.array(fp_img)
        fr = np.array(fr_img)
        norm_r = np.linalg.norm(fr, axis=1)
        norm_p = np.linalg.norm(fp, axis=1)
        valid = norm_r > 0.01  # eV/A
        if not valid.any():
            continue
        ratios.extend((norm_p[valid] / norm_r[valid]).tolist())
    return np.mean(ratios) if ratios else np.nan

def ts_force_norm(f_pred_imgs, ts_idx):
    """RMS force at the TS image (index ts_idx)."""
    if f_pred_imgs[ts_idx] is None:
        return np.nan
    return float(np.sqrt(np.mean(np.array(f_pred_imgs[ts_idx])**2)))

def r_squared(e_pred_rel, e_ref_rel):
    """R² of predicted vs reference relative energy profile."""
    e_pred = np.array(e_pred_rel)
    e_ref  = np.array(e_ref_rel)
    ss_res = np.sum((e_pred - e_ref)**2)
    ss_tot = np.sum((e_ref - e_ref.mean())**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ────────────────────────────────────────────────────────────────────────────
print(SEP)
print('BENCHMARK ANALYSIS — 30 REACTIONS')
print(SEP)

# ════════════════════════════════════════════════════════════════════════════
# PART 1: RMSD
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*100)
print('PART 1  TS GEOMETRY RMSD -- T1x wB97X-D3 optimised  vs  NEB wB97M-V optimised  (Angstrom)')
print('        Measures how much the TS moved when re-optimised at higher level + better geometry')
print('='*100)

print(f'\n  {"Rxn":12s}  {"Group":7s}  {"RMSD (A)":>9}')
print(f'  {"-"*35}')
for label, rxns in GROUPS:
    vals = []
    for rxn in rxns:
        r = bm[rxn]
        v = r.get('ts_rmsd_ang')
        s = f'{v:.4f}' if v is not None else 'N/A'
        print(f'  {rxn:12s}  {label:7s}  {s:>9}')
        if v: vals.append(v)
    print(f'  {"":12s}  {"MEAN":7s}  {np.mean(vals):>9.4f}   (max {np.max(vals):.4f})')
    print()

rmsd_all = [r['ts_rmsd_ang'] for r in bm.values() if r.get('ts_rmsd_ang')]
print(f'  Overall mean RMSD: {np.mean(rmsd_all):.4f} A   max: {np.max(rmsd_all):.4f} A  (rxn5690)')
print(f'  --> Geometries are essentially unchanged. T1x and NEB TS are the same structure.')
print(f'      The 0.086 A outlier (rxn5690) is the only case with a real conformational difference.')


# ════════════════════════════════════════════════════════════════════════════
# PART 2: BARRIER HEIGHTS
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*100)
print('PART 2  BARRIER HEIGHTS — forward barrier (TS − reactant) in meV')
print('        Reference: CCSD(T)/def2-TZVP on NEB wB97M-V stationary-point geometries')
print('        Error = method − CCSD(T).  Positive = overestimate.')
print('='*100)

# --- 2a: geometry effect ---
print('\n  [2a] GEOMETRY EFFECT — same functional (wB97X-D3), two different geometries')
print(f'       Does re-optimising the geometry from T1x to NEB wB97M-V change the barrier?')
print()
print(f'  {"Rxn":12s}  {"Grp":5s}  {"T1x geom":>9}  {"NEB geom":>9}  {"geom delta":>10}')
print(f'  {"-"*52}')

geom_deltas = {g: [] for g, _ in GROUPS}
for label, rxns in GROUPS:
    for rxn in rxns:
        r = bm[rxn]
        t1x = r.get('t1x_wb97x_fwd_meV')
        neb = r.get('neb_wb97x_fwd_meV')
        if t1x and neb:
            d = neb - t1x
            geom_deltas[label].append(d)
            print(f'  {rxn:12s}  {label[:3]:5s}  {t1x:>9.0f}  {neb:>9.0f}  {d:>+10.0f}')
    vals = geom_deltas[label]
    print(f'  {"":12s}  {"MEAN":5s}  {"":>9}  {"":>9}  {np.mean(vals):>+10.0f}   (std {np.std(vals):.0f})')
    print()

all_gd = [v for vals in geom_deltas.values() for v in vals]
print(f'  --> Re-optimising geometry lowers the wB97X-D3 barrier by {-np.mean(all_gd):.0f} meV on average.')
print(f'      This is a small effect (~{abs(np.mean(all_gd)/4000*100):.0f}% of barrier height). Geometry is not the main source of error.')

# --- 2b: method comparison ---
print()
print(f'\n  [2b] METHOD COMPARISON — all methods on NEB wB97M-V geometries vs CCSD(T)')
print(f'       Columns: raw barrier (meV) | error vs CCSD(T) (meV)')
print()
print(f'  {"Rxn":12s}  {"Grp":5s}  {"CCSD(T)":>8}  {"wB97M-V":>8}  {"err":>6}  {"wB97X-D3":>9}  {"err":>6}  '
      f'{"MACE":>6}  {"err":>6}  {"delta":>6}  {"err":>6}  {"NEVPT2":>7}  {"err":>6}')
print(f'  {"-"*105}')

method_errs = {
    'wB97M-V':   {g: [] for g, _ in GROUPS},
    'wB97X-D3':  {g: [] for g, _ in GROUPS},
    'MACE':      {g: [] for g, _ in GROUPS},
    'delta':     {g: [] for g, _ in GROUPS},
    'NEVPT2':    {g: [] for g, _ in GROUPS},
}

for label, rxns in GROUPS:
    for rxn in rxns:
        r = bm[rxn]
        ref = r.get('ccsdt_fwd_meV')
        if ref is None:
            continue
        def e(key): return r.get(key)
        def ferr(key):
            v = e(key)
            return f'{v-ref:>+6.0f}' if v else f'{"N/A":>6}'
        def fval(key, w=8):
            v = e(key)
            return f'{v:>{w}.0f}' if v else f'{"N/A":>{w}}'

        nev = r.get('nevpt2_fwd_meV')

        print(f'  {rxn:12s}  {label[:3]:5s}  {ref:>8.0f}  '
              f'{fval("neb_wb97m_fwd_meV")}  {ferr("neb_wb97m_fwd_meV")}  '
              f'{fval("neb_wb97x_fwd_meV",9)}  {ferr("neb_wb97x_fwd_meV")}  '
              f'{fval("mace_fwd_meV",6)}  {ferr("mace_fwd_meV")}  '
              f'{fval("delta_fwd_meV",6)}  {ferr("delta_fwd_meV")}  '
              f'{fval("nevpt2_fwd_meV",7)}  {ferr("nevpt2_fwd_meV")}')

        for key, mname in [('neb_wb97m_fwd_meV','wB97M-V'),('neb_wb97x_fwd_meV','wB97X-D3'),
                           ('mace_fwd_meV','MACE'),('delta_fwd_meV','delta'),('nevpt2_fwd_meV','NEVPT2')]:
            v = r.get(key)
            if v: method_errs[mname][label].append(v - ref)
    print()

# Summary table
print(f'\n  SUMMARY vs CCSD(T):')
print(f'  {"Method":12s}  {"Ref":6s}  {"n":>3}  {"MAE":>7}  {"RMSE":>7}  {"Bias":>8}')
print(f'  {"-"*50}')
for mname in ['wB97M-V','wB97X-D3','MACE','delta','NEVPT2']:
    all_e = [v for vals in method_errs[mname].values() for v in vals]
    if not all_e: continue
    e = np.array(all_e)
    print(f'  {mname:12s}  {"CCSD(T)":6s}  {len(e):>3}  {nz(e):>7.0f}  {rms(e):>7.0f}  {e.mean():>+8.0f}')

print()
print(f'  Per-group breakdown (MAE / Bias meV):')
print(f'  {"Method":12s}  ', end='')
for label, _ in GROUPS:
    print(f'  {label} (MAE/Bias)  ', end='')
print()
print(f'  {"-"*75}')
for mname in ['wB97M-V','wB97X-D3','MACE','delta','NEVPT2']:
    print(f'  {mname:12s}  ', end='')
    for label, _ in GROUPS:
        e = np.array(method_errs[mname][label])
        if len(e):
            print(f'  {nz(e):>5.0f} / {e.mean():>+5.0f}    ', end='')
        else:
            print(f'  {"N/A":>5}   {"N/A":>5}    ', end='')
    print()

print(f'\n  --> wB97M-V is the best DFT reference (MAE ~100 meV overall, ~27 meV for low-MR).')
print(f'      wB97X-D3 systematically overestimates by ~230 meV across all groups.')
print(f'      MACE+delta halves the bias vs MACE (+53 vs +198 meV) at similar MAE.')
print(f'      NEVPT2 (8 reactions) is closest to CCSD(T): MAE 80 meV, slight underestimate.')


# ════════════════════════════════════════════════════════════════════════════
# PART 3: SINGLE-POINT COMPARISON — two focused questions
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*100)
print('PART 3  SINGLE-POINT COMPARISON — 10 NEB wB97M-V images per reaction')
print()
print('  Q1: Does MACE track wB97X-D3?  (reference = wB97X-D3, its training target)')
print('      Tells you how faithfully the ML model reproduces the functional it was trained on.')
print()
print('  Q2: Does MACE+delta track wB97M-V better than MACE alone?')
print('      (reference = wB97M-V for both; wB97X-D3 shown as baseline for the functional gap)')
print('      Tells you whether the correction head closes the gap toward the better reference.')
print()
print('  ENERGY: relative to first image per reaction (meV)')
print('  FORCES: per-atom 3D vectors vs reference')
print('    MAE        — mean absolute error per Cartesian component (meV/A)')
print('    Cosine sim — mean cos(angle) between predicted and reference force per atom')
print('                 1=same direction, 0=perpendicular, -1=opposite')
print('    Mag ratio  — mean |F_pred|/|F_ref| per atom; 1=same magnitude')
print('='*100)

# ── Compute metrics for a given (pred_ekey, pred_fkey) vs (ref_ekey, ref_fkey) ──

def compute_metrics(pred_ekey, pred_fkey, ref_ekey, ref_fkey, rxn_list):
    """
    Returns per-reaction dicts with energy and force metrics.
    ref_ekey/fkey = the reference method's data keys.
    """
    results = {}
    for rxn in rxn_list:
        r = sp.get(rxn)
        if r is None:
            continue

        e_ref_abs = np.array(r[ref_ekey])
        e_ref_rel = (e_ref_abs - e_ref_abs[0]) * 1000
        f_ref = r[ref_fkey]
        ts_idx_ref = int(e_ref_rel.argmax())

        e_abs = np.array(r[pred_ekey])
        e_rel = (e_abs - e_abs[0]) * 1000
        f_pred = r[pred_fkey]

        # Energy
        e_err = e_rel - e_ref_rel
        ts_idx_pred = int(e_rel.argmax())

        # Force MAE
        f_errs = []
        for fp, fr in zip(f_pred, f_ref):
            if fp is None or fr is None: continue
            f_errs.extend((np.abs(np.array(fp) - np.array(fr)) * 1000).flatten().tolist())

        results[rxn] = {
            'e_mae':      float(np.abs(e_err).mean()),
            'e_bias':     float(e_err.mean()),
            'e_r2':       r_squared(e_rel, e_ref_rel),
            'ts_match':   int(ts_idx_pred == ts_idx_ref),
            'ts_e_err':   float(abs(e_rel[ts_idx_ref] - e_ref_rel[ts_idx_ref])),
            'f_mae':      float(np.mean(f_errs)) if f_errs else np.nan,
            'f_cos':      cosine_sim(f_pred, f_ref),
            'f_ratio':    mag_ratio(f_pred, f_ref),
        }
    return results


def summarise(metrics_dict, rxn_list, label='All'):
    """Print a clean summary table row for one method comparison."""
    vals = [metrics_dict[rxn] for rxn in rxn_list if rxn in metrics_dict]
    if not vals:
        return
    def agg(key):
        v = [x[key] for x in vals if not np.isnan(x[key])]
        return np.mean(v) if v else np.nan

    print(f'  {label:9s}  '
          f'{agg("e_mae"):>8.1f}  {agg("e_bias"):>+8.1f}  {agg("e_r2"):>7.4f}  '
          f'{agg("ts_match")*100:>8.0f}  '
          f'{agg("f_mae"):>8.1f}  {agg("f_cos"):>8.4f}  {agg("f_ratio"):>8.3f}')


HDR = (f'  {"Group":9s}  {"eMAE":>8}  {"eBias":>8}  {"eR2":>7}  '
       f'{"TS%":>8}  {"fMAE":>8}  {"fCos":>8}  {"fRatio":>8}')
HDR_UNITS = (f'  {"":9s}  {"(meV)":>8}  {"(meV)":>8}  {"":>7}  '
             f'{"match":>8}  {"(meV/A)":>8}  {"(0-1)":>8}  {"(1=same)":>8}')


# ── Q1: MACE vs wB97X-D3 (training target) ──────────────────────────────────
q1 = compute_metrics('e_mace_eV', 'f_mace_eV_per_ang',
                     'e_wb97x_eV', 'f_wb97x_eV_per_ang', ALL30)

# ── Q2: vs wB97M-V — three methods compared ─────────────────────────────────
q2_x   = compute_metrics('e_wb97x_eV', 'f_wb97x_eV_per_ang',
                          'e_wb97m_eV', 'f_wb97m_eV_per_ang', ALL30)
q2_m   = compute_metrics('e_mace_eV',  'f_mace_eV_per_ang',
                          'e_wb97m_eV', 'f_wb97m_eV_per_ang', ALL30)
q2_d   = compute_metrics('e_delta_eV', 'f_delta_eV_per_ang',
                          'e_wb97m_eV', 'f_wb97m_eV_per_ang', ALL30)


# ── Print ────────────────────────────────────────────────────────────────────

print(f'\n{"="*100}')
print(f'Q1: Does MACE track wB97X-D3?  (reference = wB97X-D3, its training target)')
print(f'    A perfect ML model would score MAE~0, R2~1, cosine~1, ratio~1.')
print(f'{"="*100}')
print()
print(HDR)
print(HDR_UNITS)
print(f'  {"-"*80}')
summarise(q1, ALL30,    'All 30')
for glabel, grxns in GROUPS:
    summarise(q1, grxns, glabel[:7])

print()
print(f'  Columns:')
print(f'    eMAE   = mean absolute error of relative energies (meV)')
print(f'    eBias  = mean signed error (positive = MACE predicts higher than wB97X-D3)')
print(f'    eR2    = R2 of energy profile shape (1=perfect)')
print(f'    TS%    = % reactions where highest-energy image agrees with wB97X-D3')
print(f'    fMAE   = force MAE per Cartesian component (meV/A)')
print(f'    fCos   = mean cosine similarity per atom (1=same dir, 0=perp)')
print(f'    fRatio = mean |F_MACE|/|F_wB97X-D3| (1=same magnitude)')


print(f'\n{"="*100}')
print(f'Q2: Does MACE+delta track wB97M-V better than MACE?  (reference = wB97M-V)')
print(f'    wB97X-D3 shown as the functional-gap baseline (it has the same gap MACE inherits).')
print(f'    Improvement = delta is better than MACE on a given metric.')
print(f'{"="*100}')

for glabel, grxns in GROUPS:
    print(f'\n  [{glabel}]')
    print(HDR)
    print(HDR_UNITS)
    print(f'  {"-"*80}')
    summarise(q2_x, grxns, 'wB97X-D3')
    summarise(q2_m, grxns, 'MACE')
    summarise(q2_d, grxns, 'MACE+delta')

print(f'\n  [All 30]')
print(HDR)
print(HDR_UNITS)
print(f'  {"-"*80}')
summarise(q2_x, ALL30, 'wB97X-D3')
summarise(q2_m, ALL30, 'MACE')
summarise(q2_d, ALL30, 'MACE+delta')

print()
print(f'  Reading guide:')
print(f'    - MACE vs wB97M-V error = functional gap (inherited from training) + ML error')
print(f'    - MACE+delta vs wB97M-V error = what remains after the delta correction')
print(f'    - Improvement in eMAE/fCos/fRatio when going MACE -> MACE+delta = delta working')
print(f'    - If MACE+delta scores similar to wB97X-D3 -> delta recovered the functional gap')
print(f'    - If MACE+delta scores better than wB97X-D3 -> delta went further than just bridging')

print('\n' + SEP)
