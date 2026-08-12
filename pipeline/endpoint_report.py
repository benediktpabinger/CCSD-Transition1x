"""Are the reference NEB endpoints on the right surface?

The premise the whole benchmark rests on and the one thing nobody checked: the
stability analysis was run at 180 geometries and every one of them was a
transition state. A barrier is E(TS) minus E(reactant), so an unstable reactant
puts the wrong zero under every barrier -- and it would do so for the
single-reference reactions too, since that split was made on the transition
state alone.

The expected answer is that they are all stable. What the report has to make
easy is spotting the exception if there is one.

Result: 45 of 45 reactants are closed-shell, 5 of 45 products are not. Which
side breaks decides what is affected, so the report distinguishes them rather
than counting "broken endpoints" -- the conclusion originally written here
assumed a broken endpoint meant a broken barrier zero, and that is wrong when
only products break, because a barrier is measured from the reactant.
"""
import glob
import json
import os
import re

import numpy as np

H = '/home/energy/s242862'
HA_MEV = 27211.386
S2RE = re.compile(r'<S\*\*2>\s*:\s*([-\d.]+)')
ERE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')


def read(label):
    p = f'{H}/orca_endpoint/{label}/sp.out'
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return {'state': 'running or failed'}
    s2 = S2RE.findall(t)
    e = ERE.findall(t)
    unstable = ('is unstable' in t) or ('UNSTABLE' in t)
    stable = 'current solution is stable' in t or 'is stable' in t
    return {'state': 'done', 's2': float(s2[-1]) if s2 else None,
            'e': float(e[-1]) if e else None,
            'unstable': unstable, 'stable': stable,
            'raw': [l.strip() for l in t.split('\n')
                    if 'stable' in l.lower() or 'STABILITY' in l]}


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
nf = {x['rxn']: x['nfod'] for x in res}


def cls(rx):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return '?'
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    if not g or g.get('ext_stable') is None:
        return '?'
    return 'MR' if g['ext_stable'] is False else 'simple'


def ts_s2(rx):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return None, None
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    b = (g or {}).get('bs') or {}
    return b.get('s2'), b.get('de_meV')


print('ARE THE REFERENCE ENDPOINTS ON THE GROUND-STATE SURFACE?')
print('=' * 92)
print('One single point with STABPerform at each relaxed reactant and product')
print('of the reference NEB. <S^2> > 0 after the analysis means the restricted')
print('solution was unstable there and ORCA rotated into a broken one.')
print()
print(f'{"rxn":<9}{"grp":<8}{"reactant <S2>":>15}{"product <S2>":>14}'
      f'   {"TS <S2>":>9}{"TS dE_BS":>10}   flag')
print('-' * 92)

bad, done, missing = [], 0, 0
for rx in rxns:
    c = cls(rx)
    r = read(f'{rx}_reactant')
    p = read(f'{rx}_product')
    if not r or not p or r.get('state') != 'done' or p.get('state') != 'done':
        missing += 1
        continue
    done += 1
    s2r = r['s2'] if r['s2'] is not None else float('nan')
    s2p = p['s2'] if p['s2'] is not None else float('nan')
    s2t, det = ts_s2(rx)
    flag = ''
    if (s2r and abs(s2r) > 0.05) or (s2p and abs(s2p) > 0.05):
        flag = '*** ENDPOINT BROKEN ***'
        bad.append(rx)
    elif r['unstable'] or p['unstable']:
        flag = 'reported unstable'
        bad.append(rx)
    print(f'{rx:<9}{c:<8}{s2r:>15.4f}{s2p:>14.4f}   '
          f'{s2t if s2t is not None else float("nan"):>9.3f}'
          f'{det if det is not None else float("nan"):>10.1f}   {flag}')

print()
print(f'{done} reactions complete, {missing} still running or missing')
print()
if bad:
    nr = sum(1 for rx in bad if (read(f'{rx}_reactant') or {}).get('s2')
             and abs(read(f'{rx}_reactant')['s2']) > 0.05)
    print(f'{len(bad)} reactions with a broken-symmetry endpoint: '
          + ' '.join(bad))
    print(f'of those, {nr} on the reactant side')
    print()
    if nr == 0:
        print('Every reactant is closed-shell, and only products break. That')
        print('matters for what is and is not affected, and the distinction is')
        print('easy to get backwards:')
        print()
        print('  forward barrier  E(TS) - E(reactant).  UNAFFECTED. The zero')
        print('                   sits at the reactant and every reactant here')
        print('                   is on the right surface. This is the number')
        print('                   the benchmark scores.')
        print('  reaction energy  E(product) - E(reactant).  WRONG for these,')
        print('  reverse barrier  by the depth of the symmetry breaking at the')
        print('                   product.')
        print('  product geometry optimised on the restricted surface, so not a')
        print('                   minimum of the surface the reaction runs on.')
        print('                   The reference path ends at a point that is')
        print('                   not a stationary point.')
        print()
        print('A reaction whose transition state is stable but whose product is')
        print('not is still labelled single-reference by our split, which was')
        print('made on the transition state alone. The label is right about the')
        print('transition state and wrong about the reaction.')
    else:
        print('A broken reactant puts the wrong zero under the forward barrier')
        print('itself, which is the number the benchmark scores. This is the')
        print('case that changes the question rather than a detail of it.')
else:
    print('No endpoint is spin-broken.')
    print()
    print('The premise holds: the symmetry breaking of these reactions is a')
    print('transition-state phenomenon. The minima are closed-shell and')
    print('correctly described, the reference path starts and ends where it')
    print('should, and the failure is confined to the barrier top where a bond')
    print('is half broken. That also justifies, after the fact, having checked')
    print('only transition-state geometries for 180 calculations.')
