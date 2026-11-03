"""Added reactions of decision D29, on top of results/d24_set.csv.

    baseline   100 more of the 200 verified closed-shell reactions with N_FOD < 0.2
               (results/base_selection.csv)
    high       100 reactions with N_FOD >= 1.0 and our stability analysis, drawn at
               random from those not used, no twin, class from the analysis (bin)

Seed 0. Output results/d29_set.csv, ~/neb_d29_rxns.txt

    python pipeline/mk_d29.py
"""
import csv
import os
import random

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
rows = {}
for name in ('stab_s2pos_orca6', 'stab_cshigh_orca6', 'stab_twins_orca6', 'stab_base_orca6'):
    for r in csv.DictReader(open(f'{RES}/{name}.csv')):
        rows.setdefault(r['rxn'], r)
used = {r['rxn'] for r in csv.DictReader(open(f'{RES}/d24_set.csv'))}
rng = random.Random(0)
out = []
base = sorted(r['rxn'] for r in csv.DictReader(open(f'{RES}/base_selection.csv')) if r['rxn'] not in used)
for rx in rng.sample(base, 100):
    out.append(dict(rxn=rx, group='baseline', bin='', partner='', nfod=fod[rx], root='neb_d29'))
pool = sorted(rx for rx, r in rows.items() if fod[rx] >= 1.0 and rx not in used)
print(len(base), 'baseline left,', len(pool), 'with N_FOD >= 1.0')
for rx in rng.sample(pool, 100):
    cls = 'bs' if float(rows[rx]['s2_ours']) > 0.05 and float(rows[rx]['depth_mev']) > 1 else 'closed'
    out.append(dict(rxn=rx, group='high', bin=cls, partner='', nfod=fod[rx], root='neb_d29'))
with open(f'{RES}/d29_set.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    for r in out:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
with open(f'{H}/neb_d29_rxns.txt', 'w') as fh:
    fh.write('\n'.join(r['rxn'] for r in out) + '\n')
print(len(out), 'reactions,', sum(r['bin'] == 'bs' for r in out), 'high N_FOD broken-symmetry')
