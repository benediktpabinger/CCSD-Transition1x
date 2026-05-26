import json

path = 'c:/Transition 1X/Transition 1x/Transition1x/full_benchmark_results.json'
with open(path) as f:
    data = json.load(f)

updates = {
    'rxn9246': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 1433,  'nevpt2_rev_meV': 894,   'nevpt2_flag': '0 frac@TS; |NEV-CCT|=343meV'},
    'rxn4498': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': None,   'nevpt2_rev_meV': None,  'nevpt2_flag': 'MISSING'},
    'rxn1061': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': None,   'nevpt2_rev_meV': None,  'nevpt2_flag': 'MISSING'},
    'rxn4003': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': None,   'nevpt2_rev_meV': None,  'nevpt2_flag': 'MISSING'},
    'rxn4004': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 2185,  'nevpt2_rev_meV': 42,    'nevpt2_flag': '0 frac@R; 0 frac@TS'},
    'rxn4063': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 2031,  'nevpt2_rev_meV': 699,   'nevpt2_flag': '0 frac@R; 0 frac@TS; 0 frac@P'},
    'rxn4114': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 2434,  'nevpt2_rev_meV': 1663,  'nevpt2_flag': '0 frac@R; 0 frac@TS'},
    'rxn4060': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 1688,  'nevpt2_rev_meV': 832,   'nevpt2_flag': '0 frac@R; 0 frac@TS; 0 frac@P'},
    'rxn1961': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 869,   'nevpt2_rev_meV': -1370, 'nevpt2_flag': '0 frac@R; 0 frac@TS; 0 frac@P; neg rev; |NEV-CCT|=1400meV'},
    'rxn1962': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 2019,  'nevpt2_rev_meV': 1439,  'nevpt2_flag': '0 frac@R; 0 frac@TS; 0 frac@P; |NEV-CCT|=393meV'},
    'rxn0896': {'nevpt2_reliable': True,  'nevpt2_fwd_meV': 5145,  'nevpt2_rev_meV': None,  'nevpt2_flag': ''},
    'rxn1154': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 4295,  'nevpt2_rev_meV': 3444,  'nevpt2_flag': '0 frac@P; |NEV-CCT|=448meV'},
    'rxn5690': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 3274,  'nevpt2_rev_meV': -72,   'nevpt2_flag': 'neg rev (-72meV)'},
    'rxn4513': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 1929,  'nevpt2_rev_meV': 2831,  'nevpt2_flag': '0 frac@TS; 0 frac@P'},
    'rxn7955': {'nevpt2_reliable': True,  'nevpt2_fwd_meV': 3100,  'nevpt2_rev_meV': None,  'nevpt2_flag': ''},
    'rxn4519': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': None,   'nevpt2_rev_meV': None,  'nevpt2_flag': 'MISSING'},
    'rxn4500': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': None,   'nevpt2_rev_meV': None,  'nevpt2_flag': 'MISSING'},
    'rxn2553': {'nevpt2_reliable': True,  'nevpt2_fwd_meV': 2009,  'nevpt2_rev_meV': 2011,  'nevpt2_flag': ''},
    'rxn8829': {'nevpt2_reliable': True,  'nevpt2_fwd_meV': 2994,  'nevpt2_rev_meV': 2938,  'nevpt2_flag': ''},
    'rxn1155': {'nevpt2_reliable': False, 'nevpt2_fwd_meV': 2453,  'nevpt2_rev_meV': 1874,  'nevpt2_flag': '0 frac@R; 0 frac@TS; |NEV-CCT|=344meV'},
}

for rxn_data in data['reactions']:
    rxn = rxn_data['rxn']
    if rxn in updates:
        rxn_data.update(updates[rxn])

with open(path, 'w') as f:
    json.dump(data, f, indent=2)

reliable = [r for r in data['reactions'] if r.get('nevpt2_reliable')]
print(f'Total reliable across all 30: {len(reliable)}')
for r in data['reactions']:
    if r['rxn'] in updates:
        flag = r.get('nevpt2_flag', '')
        print(f'  {r["rxn"]}: reliable={r["nevpt2_reliable"]}  fwd={r.get("nevpt2_fwd_meV")}  flag={flag}')
