"""Collect UMA NEB barriers and patch full_benchmark_results.json."""
import sqlite3, json, numpy as np

REACTIONS = [
    'rxn7949','rxn8832','rxn1320','rxn4113','rxn8885',
    'rxn7945','rxn7937','rxn6196','rxn0346','rxn1150',
    'rxn0896','rxn1154','rxn5690','rxn4513','rxn7955',
    'rxn4519','rxn4500','rxn2553','rxn8829','rxn1155',
    'rxn9246','rxn4498','rxn1061','rxn4003','rxn4004',
    'rxn4063','rxn4114','rxn4060','rxn1961','rxn1962',
]

BM_PATH = '/home/energy/s242862/delta_head/full_benchmark_results.json'

with open(BM_PATH) as f:
    data = json.load(f)
bm = {r['rxn']: r for r in data['reactions']}

for rxn in REACTIONS:
    base = '/home/energy/s242862/uma_neb_results/' + rxn
    try:
        con = sqlite3.connect(base + '/neb.db')
        all_e = [row[0] for row in con.execute('SELECT energy FROM systems ORDER BY id').fetchall()]
        con.close()
        e_last = np.array(all_e[-10:])
        rel = (e_last - e_last[0]) * 1000
        fwd = float(rel.max())
        rev = float(rel.max() - rel[-1])

        with open(base + '/fmaxs.json') as f2:
            fmaxs = json.load(f2)
        final_fmax = fmaxs[-1] if fmaxs else None

        bm[rxn]['uma_neb_fwd_meV'] = round(fwd, 1)
        bm[rxn]['uma_neb_rev_meV'] = round(rev, 1)
        bm[rxn]['uma_neb_fmax']    = round(final_fmax, 4) if final_fmax else None
        print(f'{rxn}: fwd={fwd:.0f}  rev={rev:.0f}  fmax={final_fmax:.3f}  rows={len(all_e)}')
    except Exception as e:
        bm[rxn]['uma_neb_fwd_meV'] = None
        bm[rxn]['uma_neb_rev_meV'] = None
        bm[rxn]['uma_neb_fmax']    = None
        print(f'{rxn}: ERROR {e}')

with open(BM_PATH, 'w') as f:
    json.dump(data, f, indent=2)
print(f'\nPatched {BM_PATH}')
