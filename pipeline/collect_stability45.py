"""Aggregate the 45-reaction x 4-geometry stability pipeline into one table."""
import json, os, glob

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

rows, missing, notes = [], [], []
for rx in sorted(grp, key=lambda r: -nf[r]):
    f = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(f):
        missing.append(rx); continue
    d = json.load(open(f))
    for g in d.get('geometries', d if isinstance(d, list) else []):
        rows.append((rx, grp[rx], nf[rx], g))

print(f'Reaktionen mit Datei: {len(grp)-len(missing)}/{len(grp)}   Zeilen: {len(rows)}')
if missing: print('FEHLT:', missing)

def fmt(v, p=3):
    return '' if v is None else (f'{v:.{p}f}' if isinstance(v, float) else str(v))

hdr = ('| rxn | grp | N_FOD | geom | RKS conv | max|g|_RKS | ext stab | '
       'lmin_ext | BS | route | dE [meV] | S2 | max|g|_BS | UKS int | '
       'lmin_int(UKS) | lmin_ext(UKS) |')
sep = '|' + '---|' * 16
print('\n' + hdr); print(sep)
stat = {'ext_unstable': 0, 'bs_found': 0, 'uks_int_stable': 0, 'route2': 0,
        'rks_fail': 0, 'breakdown': 0, 'total': 0}
for rx, gr, fod, g in rows:
    stat['total'] += 1
    bs = g.get('bs') or {}
    ext_ok = g.get('ext_stable')
    if ext_ok is False: stat['ext_unstable'] += 1
    if bs and 'invalid' not in bs: stat['bs_found'] += 1
    if bs.get('route') == 2: stat['route2'] += 1
    if bs.get('uks_int_stable') is True: stat['uks_int_stable'] += 1
    if g.get('rks_converged') is not True: stat['rks_fail'] += 1
    for k in ('lmin_ext', 'lmin_int'):
        if isinstance(g.get(k), str): stat['breakdown'] += 1
    rg = g.get('rks_grad') or {}
    bg = bs.get('bs_grad') or {}
    print('| {} | {} | {:.4f} | {} | {} | {} | {} | {} | {} | {} | {} | {} | '
          '{} | {} | {} | {} |'.format(
        rx, gr, fod, g.get('source', ''),
        'ja' if g.get('rks_converged') else 'NEIN',
        fmt(rg.get('max_evang'), 4),
        {True: 'stabil', False: 'INSTABIL'}.get(ext_ok, ''),
        fmt(g.get('lmin_ext'), 5),
        'ja' if (bs and 'invalid' not in bs) else ('-' if ext_ok else 'NEIN'),
        fmt(bs.get('route'), 0), fmt(bs.get('de_meV'), 1), fmt(bs.get('s2'), 4),
        fmt(bg.get('max_evang'), 4),
        {True: 'stabil', False: 'INSTABIL'}.get(bs.get('uks_int_stable'), ''),
        fmt(bs.get('uks_lmin_int'), 5), fmt(bs.get('uks_lmin_ext'), 5)))

print('\n=== Zusammenfassung ===')
for k, v in stat.items(): print(f'  {k:16s} {v}')
