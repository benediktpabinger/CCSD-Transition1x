"""Reaction set of the DFT test at found structures (docs/paper_decisions.md D24).

    pilot      the 200 reactions of results/pilot_pairs.csv
    onset      30 broken-symmetry reactions with our breaking depth 1-10 meV at the
               stored TS, drawn at random (seed 0), each with a closed-shell twin by
               the rule of mk_pilot.py, no reaction used twice
    baseline   50 of the 200 verified closed-shell reactions with N_FOD < 0.2
               (results/base_selection.csv), drawn at random (seed 0)

Output results/d24_set.csv, ~/neb_d24_rxns.txt (the 110 reactions that need a NEB)

    python pipeline/mk_d24.py
"""
import csv
import json
import os
import random

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
TOL = 0.05

fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
ts = json.load(open(f'{H}/t1x_ts_all.json'))
heavy = lambda rx: sum(z > 1 for z in ts[rx]['numbers'])    # noqa: E731
rows = {}
for name in ('stab_s2pos_orca6', 'stab_cshigh_orca6', 'stab_twins_orca6', 'stab_base_orca6'):
    for r in csv.DictReader(open(f'{RES}/{name}.csv')):
        rows.setdefault(r['rxn'], r)
cs = [r for r in rows.values() if float(r['lambda_min']) > 0 and float(r['depth_mev']) < 1
      and abs(float(r['s2_ours'])) < 1e-3 and not int(r['uks_direct_bs'])]
pairs = list(csv.DictReader(open(f'{RES}/pilot_pairs.csv')))
out = []
for p in pairs:
    out.append(dict(rxn=p['rxn_bs'], group='bs', bin=p['bin'], partner=p['rxn_cs'], nfod=fod[p['rxn_bs']], root='neb_pilot'))
    out.append(dict(rxn=p['rxn_cs'], group='closed', bin=p['bin'], partner=p['rxn_bs'], nfod=fod[p['rxn_cs']], root='neb_pilot'))
taken = {r['rxn'] for r in out}
rng = random.Random(0)
pool = sorted((r for r in rows.values() if 1 < float(r['depth_mev']) <= 10 and float(r['s2_ours']) > 0.05
               and r['rxn'] not in taken), key=lambda r: r['rxn'])
rng.shuffle(pool)
n = tried = 0
for b in pool:
    if n == 30:
        break
    tried += 1
    x, nh = fod[b['rxn']], heavy(b['rxn'])
    win = [c for c in cs if c['rxn'] not in taken and abs(fod[c['rxn']] - x) <= TOL]
    same = [c for c in win if heavy(c['rxn']) == nh]
    if not win:
        continue
    c = min(same or win, key=lambda c: abs(fod[c['rxn']] - x))
    taken |= {b['rxn'], c['rxn']}
    out.append(dict(rxn=b['rxn'], group='bs', bin='1-10', partner=c['rxn'], nfod=x, root='neb_d24'))
    out.append(dict(rxn=c['rxn'], group='closed', bin='1-10', partner=b['rxn'], nfod=fod[c['rxn']], root='neb_d24'))
    n += 1
print(f'onset: {n} pairs, {tried} drawn, {len(pool)} in pool')
base = sorted(r['rxn'] for r in csv.DictReader(open(f'{RES}/base_selection.csv')) if r['rxn'] not in taken)
for rx in rng.sample(base, 50):
    out.append(dict(rxn=rx, group='baseline', bin='', partner='', nfod=fod[rx], root='neb_d24'))
assert len({r['rxn'] for r in out}) == len(out)
with open(f'{RES}/d24_set.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    for r in out:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})
new = [r['rxn'] for r in out if r['root'] == 'neb_d24']
with open(f'{H}/neb_d24_rxns.txt', 'w') as fh:
    fh.write('\n'.join(new) + '\n')
print(f'{len(out)} reactions, {len(new)} need a NEB')
