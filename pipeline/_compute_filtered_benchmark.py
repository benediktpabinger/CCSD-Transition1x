"""
Compute filtered TS-RMSD and barrier-error statistics for the 23-reaction
CASSCF OptTS benchmark, stratified by reliability class.

Reads locally:
  ts_rmsd_final.json          — RMSD of each method's NEB TS vs CASSCF OptTS
  barrier_comparison_optts.json — barriers: NEVPT2(OptTS) ref + all models

Outputs a clean summary table for the paper.
"""
import json, math, statistics, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('ts_rmsd_final.json') as f:
    rmsd_data = {d['rxn']: d for d in json.load(f)}

with open('barrier_comparison_optts.json') as f:
    barrier_data = {d['rxn']: d for d in json.load(f)}

# ── Reliability classification ──────────────────────────────────────────────
RELIABLE   = {'rxn7949','rxn8832','rxn8885','rxn7945','rxn6196',
              'rxn3107','rxn7936','rxn7957',
              'rxn7937','rxn0346',              # upgraded: intruder validated by CCSD(T) Δ<150 meV
              'rxn7060'}                        # upgraded: intruder validated Δ=+97 meV (R>TS pattern noted)
RELIABLE_S = {'rxn1320','rxn1147','rxn1150'}   # geometry OK, NEVPT2 biased (0@R)
CAVEAT     = {'rxn0896','rxn8827','rxn10005'}
BORDERLINE = {'rxn8837'}
EXCL_GEO   = {'rxn4518','rxn0101','rxn4522',
              'rxn10054'}                       # negative NEVPT2 barrier (−30 meV) → wrong saddle point
EXCL_NEV   = {'rxn4113'}

def cls(rxn):
    if rxn in RELIABLE:   return 'reliable'
    if rxn in RELIABLE_S: return 'reliable*'
    if rxn in CAVEAT:     return 'caveat'
    if rxn in BORDERLINE: return 'borderline'
    if rxn in EXCL_GEO:   return 'excl-geo'
    if rxn in EXCL_NEV:   return 'excl-nevpt2'
    return '?'

# Reactions where NEVPT2(OptTS) barrier is an unreliable reference
# (rxn10054 now in EXCL_GEO and already excluded from all stats)
BARRIER_UNRELIABLE = {'rxn1150', 'rxn4113',
                      'rxn1147',   # 0@R: CCSD(T)=4023 vs NEVPT2=2114 (Δ=+1909 meV)
                      'rxn0896'}   # intruder: CCSD(T)@OptTS=4548 vs NEVPT2=2484 (Δ=+2064 meV)

RMSD_METHODS = [
    ('orca',       'orca_vs_optts',       'ORCA DFT'),
    ('t1x',        't1x_vs_optts',        'T1x wB97X'),
    ('mace_bare',  'mace_bare_vs_optts',  'MACE bare'),
    ('mace_delta', 'mace_delta_vs_optts', 'MACE+delta'),
    ('uma_s',      'uma_s_vs_optts',      'UMA-s'),
    ('uma_m',      'uma_m_vs_optts',      'UMA-m'),
    ('esen',       'esen_vs_optts',       'eSEN'),
]

BARRIER_METHODS = [
    ('orca',       'orca_fwd_meV',       'ORCA DFT'),
    ('mace_bare',  'mace_bare_fwd_meV',  'MACE bare'),
    ('mace_delta', 'mace_delta_fwd_meV', 'MACE+delta'),
    ('uma_s',      'uma_s_fwd_meV',      'UMA-s'),
    ('uma_m',      'uma_m_fwd_meV',      'UMA-m'),
    ('esen',       'esen_fwd_meV',       'eSEN'),
]

all_rxns = sorted(rmsd_data.keys())

# ── Per-reaction table ───────────────────────────────────────────────────────
print('=' * 130)
print('PER-REACTION TS RMSD vs CASSCF OptTS [Å]')
print('=' * 130)
hdr = f"{'rxn':10s} {'class':12s} {'ORCA':7s} {'T1x':7s} {'MACE':7s} {'MACEd':7s} {'UMA-s':7s} {'UMA-m':7s} {'eSEN':7s}"
print(hdr)
print('-' * 80)
for rxn in all_rxns:
    c = cls(rxn)
    d = rmsd_data[rxn]
    vals = []
    for _, key, _ in RMSD_METHODS:
        v = d.get(key)
        if v is None or v == 'FRAG' or isinstance(v, str):
            vals.append(' FRAG ')
        else:
            vals.append(f'{v:6.3f}')
    flag = ' ← excl' if c in ('excl-geo','excl-nevpt2') else ''
    flag += ' ← bad-ref' if rxn in BARRIER_UNRELIABLE else ''
    print(f"{rxn:10s} {c:12s} {' '.join(vals)}{flag}")

# ── Aggregate RMSD stats by filter set ──────────────────────────────────────
print()
print('=' * 130)
print('AGGREGATE RMSD STATISTICS (mean ± std) [Å]')
print('=' * 130)

filter_sets = [
    ('All 23',          all_rxns),
    ('No excl-geo (19)',  [r for r in all_rxns if r not in EXCL_GEO]),
    ('Reliable only (11)', [r for r in all_rxns if r in RELIABLE]),
    ('Rel+Rel* (14)',      [r for r in all_rxns if r in RELIABLE | RELIABLE_S]),
    ('Rel+Rel*+Cav (17)', [r for r in all_rxns if r in RELIABLE|RELIABLE_S|CAVEAT]),
]

print(f"{'Filter':25s}", end='')
for _, _, label in RMSD_METHODS:
    print(f'  {label:12s}', end='')
print()
print('-' * 130)

for fname, rxns in filter_sets:
    print(f'{fname:25s}', end='')
    for _, key, _ in RMSD_METHODS:
        vals = []
        for rxn in rxns:
            v = rmsd_data[rxn].get(key)
            if v is not None and not isinstance(v, str):
                vals.append(float(v))
        if vals:
            print(f'  {statistics.mean(vals):5.3f}±{statistics.stdev(vals) if len(vals)>1 else 0:.3f}', end='')
        else:
            print(f'  {"—":12s}', end='')
    print()

# ── Per-reaction barrier errors ──────────────────────────────────────────────
print()
print('=' * 130)
print('PER-REACTION BARRIER ERROR vs NEVPT2(OptTS) [meV]')
print('(positive = model barrier higher than reference)')
print('=' * 130)
hdr = f"{'rxn':10s} {'class':12s} {'ref meV':8s} {'ORCA':8s} {'MACE':8s} {'MACEd':8s} {'UMA-s':8s} {'UMA-m':8s} {'eSEN':8s}"
print(hdr)
print('-' * 100)
for rxn in all_rxns:
    c = cls(rxn)
    d = barrier_data.get(rxn, {})
    ref = d.get('optts_fwd_meV')
    if ref is None:
        continue
    ref_s = f'{ref:8.0f}'
    errs = []
    for _, key, _ in BARRIER_METHODS[1:]:  # skip ORCA as method
        v = d.get(key)
        if v is None or d.get(key.replace('fwd_meV','frag'), False):
            errs.append('  FRAG  ')
        else:
            errs.append(f'{v - ref:+8.0f}')
    # ORCA error
    orca_v = d.get('orca_fwd_meV')
    orca_e = f'{orca_v - ref:+8.0f}' if orca_v else '       —'
    flag = ''
    if c == 'excl-geo': flag = ' ← excl-geo (CASSCF ref invalid)'
    if rxn in BARRIER_UNRELIABLE: flag = ' ← NEVPT2 ref unreliable'
    print(f"{rxn:10s} {c:12s} {ref_s} {orca_e} {' '.join(errs)}{flag}")

# ── Aggregate barrier MAE ────────────────────────────────────────────────────
print()
print('=' * 130)
print('AGGREGATE BARRIER MAE [meV] — errors vs NEVPT2(OptTS)')
print('=' * 130)

# Valid barrier reference: exclude excl-geo AND barrier-unreliable
valid_barrier = [r for r in all_rxns
                 if r not in EXCL_GEO and r not in BARRIER_UNRELIABLE]
valid_reliable = [r for r in valid_barrier if r in RELIABLE]
valid_rel_relS = [r for r in valid_barrier if r in RELIABLE | RELIABLE_S]
valid_full     = [r for r in valid_barrier if r in RELIABLE|RELIABLE_S|CAVEAT|BORDERLINE]

filter_sets_b = [
    ('All valid (15)',         valid_barrier),
    ('Reliable only (11)',    valid_reliable),
    ('Rel+Rel* (12)',         valid_rel_relS),
    ('Rel+Rel*+Cav+BL (15)', valid_full),
]

print(f"{'Filter':25s}", end='')
for _, _, label in BARRIER_METHODS:
    print(f'  {label:12s}', end='')
print()
print('-' * 130)

for fname, rxns in filter_sets_b:
    print(f'{fname:25s}', end='')
    for _, key, _ in BARRIER_METHODS:
        d_vals = []
        for rxn in rxns:
            d = barrier_data.get(rxn, {})
            ref = d.get('optts_fwd_meV')
            v   = d.get(key)
            frag_key = key.replace('fwd_meV', 'frag')
            if (ref is None or v is None or d.get(frag_key, False)):
                continue
            d_vals.append(abs(float(v) - float(ref)))
        if d_vals:
            print(f'  {statistics.mean(d_vals):8.0f}    ', end='')
        else:
            print(f'  {"—":12s}', end='')
    print(f'  (n={len(rxns)})')

print()
print('Notes:')
print('  excl-geo excluded from all stats: rxn4518, rxn0101, rxn4522')
print('  barrier-unreliable excluded from barrier stats:')
print('    rxn1150 (0@R, NEVPT2=1679 vs CCSD(T)=3460, Δ=+1781 meV)')
print('    rxn1147 (0@R, NEVPT2=2114 vs CCSD(T)=4023, Δ=+1909 meV)')
print('    rxn0896 (intruder, NEVPT2=2484 vs CCSD(T)@OptTS=4548, Δ=+2064 meV)')
print('    rxn4113 (0@R, use CCSD(T)=5346 meV)')
print('  rxn10054 moved to excl-geo (NEVPT2=-30 meV, wrong saddle point)')
print('  FRAG = method produced fragmented/unphysical NEB path for that reaction')
