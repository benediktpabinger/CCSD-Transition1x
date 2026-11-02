"""Reaction list of the twin pilot (docs/paper_decisions.md D22).

Broken-symmetry side: reactions with our breaking depth above 10 meV at the stored
transition state (ORCA 6.0.1, results/stab_s2pos_orca6.csv and stab_cshigh_orca6.csv,
the latter holds labels OMol25 called closed-shell), drawn at random (seed 0) in three
depth bins: 10-43, 43-300 and above 300 meV.

Closed-shell side: each drawn reaction gets the twin of nearest N_FOD at the stored TS
within 0.05, same heavy-atom count preferred, no reuse, from all reactions our analysis
finds stable there: lambda_min > 0, depth < 1 meV, <S^2> < 0.001 (stab_cshigh, stab_twins,
stab_base). A drawn reaction with no twin is skipped and the next one in its bin is taken,
so high N_FOD is under-represented; this is reported.

Output results/pilot_pairs.csv, ~/neb_pilot_rxns.txt (200 reactions)

    python pipeline/mk_pilot.py
"""
import csv
import json
import os
import random

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
BINS = [('10-43', 10, 43, 34), ('43-300', 43, 300, 33), ('>300', 300, 1e9, 33)]
TOL = 0.05

fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
ts = json.load(open(f'{H}/t1x_ts_all.json'))
heavy = lambda rx: sum(z > 1 for z in ts[rx]['numbers'])    # noqa: E731
rows = {}
for name in ('stab_s2pos_orca6', 'stab_cshigh_orca6', 'stab_twins_orca6', 'stab_base_orca6'):
    for r in csv.DictReader(open(f'{RES}/{name}.csv')):
        rows.setdefault(r['rxn'], r)
bs = [r for r in rows.values() if float(r['depth_mev']) > 10 and float(r['s2_ours']) > 0.05]
cs = [r for r in rows.values() if float(r['lambda_min']) > 0 and float(r['depth_mev']) < 1
      and abs(float(r['s2_ours'])) < 1e-3 and not int(r['uks_direct_bs'])]
print(f'{len(bs)} broken-symmetry candidates, {len(cs)} verified closed-shell candidates')
rng = random.Random(0)
taken, pairs = set(), []
for name, lo, hi, want in BINS:
    pool = sorted((r for r in bs if lo < float(r['depth_mev']) <= hi), key=lambda r: r['rxn'])
    rng.shuffle(pool)
    n, tried = 0, 0
    for b in pool:
        if n == want:
            break
        tried += 1
        x, nh = fod[b['rxn']], heavy(b['rxn'])
        win = [c for c in cs if c['rxn'] not in taken and abs(fod[c['rxn']] - x) <= TOL]
        same = [c for c in win if heavy(c['rxn']) == nh]
        if not win:
            continue
        c = min(same or win, key=lambda c: abs(fod[c['rxn']] - x))
        taken.add(c['rxn'])
        pairs.append(dict(bin=name, rxn_bs=b['rxn'], rxn_cs=c['rxn'], nfod_bs=x, nfod_cs=fod[c['rxn']],
                          depth_bs_mev=float(b['depth_mev']), s2_bs=float(b['s2_ours']),
                          heavy_bs=nh, heavy_cs=heavy(c['rxn'])))
        n += 1
    print(f'  bin {name}: {n} pairs of {want}, {tried} drawn, {len(pool)} in pool')
assert len({p['rxn_bs'] for p in pairs} | {p['rxn_cs'] for p in pairs}) == 2 * len(pairs)
with open(f'{RES}/pilot_pairs.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
    w.writeheader()
    for p in pairs:
        w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in p.items()})
with open(f'{H}/neb_pilot_rxns.txt', 'w') as fh:
    for p in pairs:
        fh.write(f"{p['rxn_bs']}\n{p['rxn_cs']}\n")
d = [abs(p['nfod_bs'] - p['nfod_cs']) for p in pairs]
print('%d pairs, |dN_FOD| median %.4f max %.4f, same heavy atoms %d' % (len(pairs), sorted(d)[len(d) // 2], max(d), sum(p['heavy_bs'] == p['heavy_cs'] for p in pairs)))
print('N_FOD range bs %.2f..%.2f' % (min(p['nfod_bs'] for p in pairs), max(p['nfod_bs'] for p in pairs)))
