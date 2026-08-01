"""For every externally unstable row: does the geometry sit on the RKS surface
or on the BS surface?  Criterion is which gradient is smaller."""
import json

H = '/home/energy/s242862'
res = sorted(json.load(open(f'{H}/fod_ranking.json'))['results'],
             key=lambda r: -r['nfod'])
n = len(res)
TOP = [res[i]['rxn'] for i in range(26)]
MID = [res[i - 1]['rxn'] for i in [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]]
LOW = [res[i]['rxn'] for i in range(n - 10, n)]
grp = {}
for r in TOP: grp[r] = 'high'
for r in MID: grp.setdefault(r, 'mid')
for r in LOW: grp.setdefault(r, 'low')
nf = {x['rxn']: x['nfod'] for x in res}

rows = []
for rx in grp:
    d = json.load(open(f'{H}/stab_pipeline/{rx}/result.json'))
    for g in d['geometries']:
        if g.get('ext_stable') is not False:
            continue
        bs = g.get('bs') or {}
        gr = (g.get('rks_grad') or {}).get('max_evang')
        gb = (bs.get('bs_grad') or {}).get('max_evang')
        if gr is None or gb is None:
            continue
        rows.append({'rxn': rx, 'grp': grp[rx], 'src': g['source'],
                     'gr': gr, 'gb': gb, 'ratio': gr / gb,
                     'de': bs.get('de_meV'), 's2': bs.get('s2')})

MODEL = [r for r in rows if r['src'] != 'RKS-ref']
REF = [r for r in rows if r['src'] == 'RKS-ref']

def report(name, rs):
    on_rks = [r for r in rs if r['ratio'] < 1]
    on_bs = [r for r in rs if r['ratio'] >= 1]
    print(f'\n=== {name}  (n={len(rs)}) ===')
    print(f'  auf RKS-Flaeche (|g|_RKS < |g|_BS): {len(on_rks)}')
    print(f'  auf BS-Flaeche  (|g|_BS < |g|_RKS): {len(on_bs)}')
    if rs:
        q = sorted(r['ratio'] for r in rs)
        print(f'  ratio |g|_RKS/|g|_BS: min {q[0]:.2f}  median '
              f'{q[len(q)//2]:.2f}  max {q[-1]:.2f}')
    return on_rks

report('RKS-Referenzgeometrien', REF)
on_rks = report('Modellgeometrien (UMA-S / UMA-M / eSEN)', MODEL)

print('\n=== Modellgeometrien, die auf der RKS-Loesung sitzen ===')
if on_rks:
    print(f"{'rxn':<10}{'grp':<6}{'geom':<8}{'|g|_RKS':>9}{'|g|_BS':>9}"
          f"{'ratio':>7}{'dE[meV]':>10}{'S2':>8}")
    for r in sorted(on_rks, key=lambda x: x['ratio']):
        print(f"{r['rxn']:<10}{r['grp']:<6}{r['src']:<8}{r['gr']:>9.4f}"
              f"{r['gb']:>9.4f}{r['ratio']:>7.2f}{r['de']:>10.1f}{r['s2']:>8.4f}")
else:
    print('  keine')

print('\n=== je Modell ===')
for m in ('UMA-S', 'UMA-M', 'eSEN'):
    sub = [r for r in MODEL if r['src'] == m]
    k = [r for r in sub if r['ratio'] < 1]
    print(f'  {m:<7} {len(k)}/{len(sub)} auf RKS-Flaeche')
