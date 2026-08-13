"""Does the instability survive at wB97X/6-31G(d)?

Builds the task list and, once the jobs are done, reports which reactions are
externally unstable at the cheap level and how that compares with
wB97M-V/def2-TZVP. Both roles in one file so the comparison cannot drift from
the list it was generated for.

  python cheap_stab_report.py list      write the task list
  python cheap_stab_report.py report    read the results

The question it answers is whether the cheap level is usable as a testbed. If
the same reactions break, it is. If different ones break, it still is, but the
reaction list does not transfer. If almost nothing breaks, it is not -- the
failure the method is meant to catch would not occur there.
"""
import json
import os
import re
import sys

import numpy as np

H = '/home/energy/s242862'
HA_MEV = 27211.386
ERE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2RE = re.compile(r'<S\*\*2>\s*:\s*([-\d.]+)')


def reactions():
    res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
                 key=lambda r: -r['nfod'])
    n = len(res)
    sel = ([res[i]['rxn'] for i in range(26)]
           + [res[i - 1]['rxn'] for i in
              [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
           + [res[i]['rxn'] for i in range(n - 10, n)])
    seen, out = set(), []
    for r in sel:
        if r not in seen:
            seen.add(r); out.append(r)
    return out, {x['rxn']: x['nfod'] for x in res}


def expensive(rx):
    """ext_stable and dE_BS at the production level, for comparison."""
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        return None, None
    g = {x['source']: x
         for x in json.load(open(p))['geometries']}.get('RKS-ref')
    if not g or g.get('ext_stable') is None:
        return None, None
    return (not g['ext_stable']), ((g.get('bs') or {}).get('de_meV'))


if sys.argv[1:2] == ['list']:
    rxns, _ = reactions()
    n = 0
    with open(f'{H}/cheap_stab_tasks.txt', 'w') as fh:
        for rx in rxns:
            g = f'{H}/orca_neb_results/{rx}/transition_state.xyz'
            if os.path.exists(g):
                fh.write(f'{rx} {g}\n')
                n += 1
    print(f'{n} tasks -> {H}/cheap_stab_tasks.txt')
    print(f'array range: 0-{n - 1}')
    raise SystemExit

rxns, nf = reactions()
print('IS THE INSTABILITY THERE AT wB97X/6-31G(d)?')
print('=' * 96)
print('One single point plus a stability analysis at each reference transition')
print('state, against the same quantity at wB97M-V/def2-TZVP.')
print()
print(f'{"rxn":<9}{"N_FOD":>7}   {"cheap":<26}   {"production":<22}  agree')
print(f'{"":<9}{"":>7}   {"<S^2>":>7}{"dE_BS meV":>12}      '
      f'{"unstable":>9}{"dE_BS meV":>12}')
print('-' * 96)

agree = dis = miss = 0
cheap_unstable, prod_unstable = [], []
both = []
for rx in rxns:
    d = f'{H}/cheap_stab/{rx}'
    pr, ps = f'{d}/rks.out', f'{d}/stab.out'
    if not (os.path.exists(pr) and os.path.exists(ps)):
        miss += 1
        continue
    tr, ts = (open(pr, errors='replace').read(),
              open(ps, errors='replace').read())
    if 'ORCA TERMINATED NORMALLY' not in ts:
        miss += 1
        continue
    er = ERE.findall(tr)
    eu = ERE.findall(ts)
    s2 = S2RE.findall(ts)
    if not (er and eu):
        miss += 1
        continue
    e_rks, e_bs = float(er[-1]), float(eu[-1])
    s2v = float(s2[-1]) if s2 else 0.0
    de = (e_bs - e_rks) * HA_MEV
    unstable_cheap = abs(s2v) > 0.05 or de < -1.0
    unst_p, de_p = expensive(rx)
    if unstable_cheap:
        cheap_unstable.append(rx)
    if unst_p:
        prod_unstable.append(rx)
    if unstable_cheap and unst_p:
        both.append(rx)
    ok = '' if unst_p is None else ('ja' if unstable_cheap == unst_p else 'NEIN')
    if ok == 'ja':
        agree += 1
    elif ok == 'NEIN':
        dis += 1
    print(f'{rx:<9}{nf.get(rx, float("nan")):>7.3f}   {s2v:>7.3f}{de:>12.1f}'
          f'      {str(unst_p):>9}'
          f'{de_p if de_p is not None else float("nan"):>12.1f}  {ok}')

print()
print(f'complete {agree + dis}, missing {miss}')
print(f'unstable at wB97X/6-31G(d)      {len(cheap_unstable):>3}')
print(f'unstable at wB97M-V/def2-TZVP   {len(prod_unstable):>3}')
print(f'both                            {len(both):>3}')
print(f'classification agrees           {agree:>3}   disagrees {dis}')

if cheap_unstable:
    dees = []
    for rx in cheap_unstable:
        d = f'{H}/cheap_stab/{rx}'
        er = ERE.findall(open(f'{d}/rks.out', errors='replace').read())
        eu = ERE.findall(open(f'{d}/stab.out', errors='replace').read())
        if er and eu:
            dees.append((float(eu[-1]) - float(er[-1])) * HA_MEV)
    if dees:
        print()
        print(f'depth of the breaking where it occurs, cheap level: '
              f'median {np.median(dees):.1f} meV, deepest {min(dees):.1f}')
        print('production level for comparison: -648.5 to -1.3 meV')

print()
if len(cheap_unstable) >= 8:
    print('The testbed carries the phenomenon: enough reactions break at the')
    print('cheap level to develop and test the method there. Whether the same')
    print('reactions break decides only whether the list transfers.')
else:
    print('The testbed does NOT carry the phenomenon well. Developing the')
    print('method here would mean testing a fix against a failure that barely')
    print('occurs. Use the production level, or a different cheap functional')
    print('with more exact exchange.')
