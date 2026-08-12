"""Task list for the endpoint stability check.

Both endpoints of every reaction in the benchmark, not only the 19: the split
into 19 and 26 was made on stability at the transition state alone, so a
reaction with an unremarkable transition state and an unstable reactant is
currently in the wrong group and nobody would notice.

The geometries are the relaxed endpoints the reference NEB actually used --
orca_neb_results/<rxn>/reactant.xyz and product.xyz -- because the question is
about the path that was computed, not about some other reactant.
"""
import json
import os

H = '/home/energy/s242862'
OUT = f'{H}/endpoint_tasks.txt'

res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
sel = ([res[i]['rxn'] for i in range(26)]
       + [res[i - 1]['rxn'] for i in
          [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
       + [res[i]['rxn'] for i in range(n - 10, n)])
seen, rxns = set(), []
for r in sel:
    if r not in seen:
        seen.add(r); rxns.append(r)


def cls(rx):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return '?'
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    if not g or g.get('ext_stable') is None:
        return '?'
    return 'MR' if g['ext_stable'] is False else 'simple'


tasks, miss = [], []
for rx in rxns:
    for end in ('reactant', 'product'):
        g = f'{H}/orca_neb_results/{rx}/{end}.xyz'
        lbl = f'{rx}_{end}'
        if not os.path.exists(g):
            miss.append(lbl)
            continue
        if os.path.exists(f'{H}/orca_endpoint/{lbl}/sp.out'):
            continue
        tasks.append((lbl, g, cls(rx)))

with open(OUT, 'w') as fh:
    for lbl, g, c in tasks:
        fh.write(f'{lbl} {g}\n')

print(f'{len(tasks)} tasks -> {OUT}')
print(f'array range: 0-{len(tasks) - 1}')
print(f'  multireference group  {sum(1 for t in tasks if t[2] == "MR")}')
print(f'  single-reference      {sum(1 for t in tasks if t[2] == "simple")}')
if miss:
    print(f'\nmissing geometries: {len(miss)}')
    for m in miss:
        print('   ', m)
