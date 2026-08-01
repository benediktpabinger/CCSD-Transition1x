"""Full 180-row table: stability, broken symmetry, and which surface each
geometry sits on (ratio |g|_RKS / |g|_BS; <1 = RKS surface, >1 = BS surface)."""
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

def f(v, p=4):
    return '' if v is None or isinstance(v, str) else f'{v:.{p}f}'

print('| rxn | grp | N_FOD | geom | RKS | max\\|g\\|_RKS | ext | lmin_ext | '
      'route | dE_BS [meV] | S2 | max\\|g\\|_BS | ratio | UKS int | '
      'lmin_int(UKS) | lmin_ext(UKS) |')
print('|' + '---|' * 16)

for rx in sorted(grp, key=lambda r: -nf[r]):
    d = json.load(open(f'{H}/stab_pipeline/{rx}/result.json'))
    for g in d['geometries']:
        bs = g.get('bs') or {}
        gr = (g.get('rks_grad') or {}).get('max_evang')
        gb = (bs.get('bs_grad') or {}).get('max_evang')
        conv = g.get('rks_converged') is True
        ext = g.get('ext_stable')
        ratio = f'{gr/gb:.2f}' if (gr and gb) else ''
        print('| {} | {} | {:.4f} | {} | {} | {} | {} | {} | {} | {} | {} | '
              '{} | {} | {} | {} | {} |'.format(
            rx, grp[rx], nf[rx], g['source'],
            'ja' if conv else '**NEIN**', f(gr),
            '' if not conv else ('stabil' if ext else '**instabil**'),
            f(g.get('lmin_ext'), 5),
            bs.get('route', ''), f(bs.get('de_meV'), 1), f(bs.get('s2')),
            f(gb), ratio,
            {True: 'stabil', False: '**instabil**'}.get(bs.get('uks_int_stable'), ''),
            f(bs.get('uks_lmin_int'), 5), f(bs.get('uks_lmin_ext'), 5)))
