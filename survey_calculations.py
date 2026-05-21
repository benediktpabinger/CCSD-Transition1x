import os, json, csv

HOME        = '/home/energy/s242862'
KCAL_TO_MEV = 43.3641

def exists(path): return os.path.isfile(path)
def isdir(path):  return os.path.isdir(path)

def load_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

test_rxns  = load_list(f'{HOME}/ccsd_dataset/test_reactions.txt')
val_rxns   = load_list(f'{HOME}/ccsd_dataset/val_reactions.txt')
vault_rxns = set(load_list(f'{HOME}/ccsd_dataset/vault_reactions.txt'))

# ── Lookup tables built once ─────────────────────────────────────────────────
barrier_p10 = set()
mace_p10_ep291 = {}
with open(f'{HOME}/barrier_comparison_p10.json') as f:
    for r in json.load(f)['reactions']:
        barrier_p10.add(r['rxn'])
        mace_p10_ep291[r['rxn']] = round(r['mace_meV'], 1)

failed_neb = set()
with open(f'{HOME}/failed_neb_list.txt') as f:
    for line in f:
        parts = line.strip().split()
        if parts: failed_neb.add(parts[0])

neb_info = {}
with open(f'{HOME}/neb_analysis/neb_analysis.json') as f:
    for r in json.load(f): neb_info[r['rxn']] = r

painn_rxns    = set()
painn_barrier = {}
with open(f'{HOME}/eval_painn_orca.json') as f:
    for r in json.load(f).get('results', []):
        if 'rxn' in r:
            painn_rxns.add(r['rxn'])
            painn_barrier[r['rxn']] = round(r['barrier_painn'] * KCAL_TO_MEV, 1)

fod_nfod = {}
fod_ranking_path = f'{HOME}/fod_results/fod_ranking.json'
if exists(fod_ranking_path):
    with open(fod_ranking_path) as f:
        for r in json.load(f)['results']:
            fod_nfod[r['rxn']] = round(r['nfod'], 4)

def load_nevpt2_avas(rxn):
    """Load validated NEVPT2 AVAS result from _pyscf_avas directory."""
    path = f'{HOME}/nevpt2_results/{rxn}_pyscf_avas/nevpt2_results.json'
    if not exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# ── Build rows ───────────────────────────────────────────────────────────────
rows = []
for split, rxns in [('test', test_rxns), ('val', val_rxns)]:
    neb_dir = 'orca_neb_results' if split == 'test' else 'orca_neb_val_results'
    for rxn in rxns:
        ni   = neb_info.get(rxn, {})
        nevpt2 = load_nevpt2_avas(rxn)

        # geometry source for validated NEVPT2 AVAS result
        if nevpt2:
            xyz = nevpt2['ts']['xyz']
            if   'orca_neb_results'   in xyz: nevpt2_geom = 'orca_neb'
            elif 'ccsd_pyscf_results' in xyz: nevpt2_geom = 'ccsd_pyscf'
            else:                             nevpt2_geom = 'other'
        else:
            nevpt2_geom = ''

        row = {
            'rxn':   rxn,
            'split': split,
            'in_vault': rxn in vault_rxns,
            # ORCA NEB wB97M-V/def2-TZVP
            'orca_neb_ts_exists':    exists(f'{HOME}/{neb_dir}/{rxn}/transition_state.xyz'),
            'orca_neb_converged':    ni.get('converged', ''),
            'orca_neb_fmax':         round(ni['last_fmax'], 4) if ni.get('last_fmax') is not None else '',
            'orca_neb_in_faillist':  rxn in failed_neb,
            'orca_barrier_eV':       round(ni['b_orca'], 4) if ni.get('b_orca') is not None else '',
            't1x_barrier_eV':        round(ni['b_t1x'], 4) if ni.get('b_t1x') is not None else '',
            # CCSD PySCF NEB (DZ)
            'ccsd_pyscf_neb':        exists(f'{HOME}/ccsd_pyscf_results/{rxn}/converged'),
            # Per-image CCSD DZ SPs
            'ccsd_dz_sp_images':     (rxn == 'rxn0103' and isdir(f'{HOME}/ccsd_rxn0103')) or
                                     (rxn == 'rxn1961' and isdir(f'{HOME}/ccsd_rxn1961')),
            'ccsd_dz_compiled':      (rxn == 'rxn0103' and exists(f'{HOME}/ccsd_rxn0103/results_0000_0012.json')) or
                                     (rxn == 'rxn1961' and exists(f'{HOME}/ccsd_rxn1961/ccsd_results_final.json')),
            # Per-image CCSD(T) TZ SPs
            'ccsdt_tz_sp_images':    rxn == 'rxn0103' and isdir(f'{HOME}/ccsd_tz_rxn0103'),
            'ccsdt_tz_compiled':     exists(f'{HOME}/data/ccsd_tz_rxn0103.json') and rxn == 'rxn0103',
            'ccsdt_tz_neb_db':       exists(f'{HOME}/ccsd_tz_results/{rxn}/neb.db'),
            # Model evaluations — boolean presence
            'in_barrier_comp_p10':   rxn in barrier_p10,
            'painn_barrier':         rxn in painn_rxns,
            # Model evaluations — numerical barriers
            'mace_p10_ep291_meV':    mace_p10_ep291.get(rxn, ''),
            'painn_barrier_meV':     painn_barrier.get(rxn, ''),
            # FOD multireference screening
            'nfod':                  fod_nfod.get(rxn, ''),
            # Delta experiment
            'delta_sp':              exists(f'{HOME}/delta_experiment/results/{rxn}.json'),
            # NEVPT2 — earlier/experimental runs (kept for historical record)
            'nevpt2_sp_66_orca':     exists(f'{HOME}/nevpt2_results/{rxn}/nevpt2_results.json'),
            'nevpt2_optts_66_orca':  exists(f'{HOME}/nevpt2_results/{rxn}_optts/nevpt2_results.json'),
            'ccsdt_sp_pyscf':        exists(f'{HOME}/nevpt2_results/{rxn}_ccsd_pyscf/ccsd_results.json'),
            'nevpt2_avas_pyscf':     exists(f'{HOME}/nevpt2_results/{rxn}_pyscf/nevpt2_results.json'),
            'nevpt2_avas_fixed':     exists(f'{HOME}/nevpt2_results/{rxn}_pyscf_fixed/nevpt2_results.json'),
            # NEVPT2 AVAS validated — _pyscf_avas: AVAS at TS, MOs projected to R/P
            'nevpt2_avas_validated':     nevpt2 is not None,
            'nevpt2_avas_fwd_meV':       round(nevpt2['barrier_forward_meV'],  1) if nevpt2 else '',
            'nevpt2_avas_rev_meV':       round(nevpt2['barrier_reverse_meV'],  1) if nevpt2 else '',
            'nevpt2_avas_geometry':      nevpt2_geom,
            # Attempted, all failed
            'mp2_neb_crashed':       rxn == 'rxn0103',
            'mp2_neb_succeeded':     exists(f'{HOME}/ccsd_neb_results/{rxn}/ccsd_singlepoints.json'),
        }
        rows.append(row)

with open(f'{HOME}/calculation_inventory.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f'Total: {len(rows)} ({sum(r["split"]=="test" for r in rows)} test, {sum(r["split"]=="val" for r in rows)} val)')
print()
skip = {'rxn', 'split', 'orca_neb_fmax', 'orca_barrier_eV', 't1x_barrier_eV',
        'mace_p10_ep291_meV', 'painn_barrier_meV', 'nevpt2_avas_fwd_meV',
        'nevpt2_avas_rev_meV', 'nevpt2_avas_geometry', 'nfod'}
for col in rows[0].keys():
    if col in skip: continue
    n   = sum(bool(r[col]) for r in rows)
    n_t = sum(bool(r[col]) for r in rows if r['split'] == 'test')
    n_v = sum(bool(r[col]) for r in rows if r['split'] == 'val')
    print(f'  {col:30s}: {n:4d}  ({n_t} test, {n_v} val)')

print(f'\nSaved: ~/calculation_inventory.csv')
