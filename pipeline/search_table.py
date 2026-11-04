"""Valid transition states by search method, model and group (D27, D30, D31).

A structure passes if it is class route under the D23 check, as rerun with the D35
fixes (folders *_v2; unconverged downhill runs count as checked, not passed). Rates are given among
converged runs and among started runs. The paired test is BS against twin, all
pairs counted from the start, exact McNemar.

Output results/search_checks.csv

    python pipeline/search_table.py
"""
import csv
import glob
import json
import os
from math import comb

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
S = {r['rxn']: r for n in ('d24_set', 'd29_set') for r in csv.DictReader(open(f'{RES}/{n}.csv'))}
M = ('esen', 'uma-s', 'uma-m', 'mace-omol')
G = ('baseline', 'closed', 'bs', 'high')
METHODS = {
    'fsm': (lambda m: f'{H}/neb_fsm/fsm/{m}', lambda m: f'{H}/neb_fsm/check_v2', 'fsm'),
    'neb_idpp': (lambda m: f'{H}/neb_pilot/idpp/{m}', lambda m: f'{H}/neb_pilot/check_idpp_v2', 'idpp'),
    'neb_t1x': (None, None, 't1x'),
}


def mc(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def runs(method, m):
    """rxn -> (converged, class or None)."""
    out = {}
    if method == 'neb_t1x':
        if m != 'esen':
            return out
        chk = {}
        for d in (f'{H}/neb_d24/check_v2/esen', f'{H}/neb_d29/check_v2/esen'):
            for f in glob.glob(f'{d}/*.json'):
                j = json.load(open(f))
                chk[j['rxn']] = j['cls']
        for rx, r in S.items():
            p = f"{H}/{r['root']}/t1x/esen/{rx}/result.json"
            if os.path.exists(p):
                out[rx] = (bool(json.load(open(p))['criterion_met']), chk.get(rx))
        return out
    rdir, cdir, start = METHODS[method]
    chk = {}
    for f in glob.glob(f'{cdir(m)}/{m}/*_{start}.json'):
        j = json.load(open(f))
        chk[j['rxn']] = j['cls']
    for f in glob.glob(f'{rdir(m)}/*/result.json'):
        j = json.load(open(f))
        if j['rxn'] in S:
            out[j['rxn']] = (bool(j['criterion_met']), chk.get(j['rxn']))
    return out


rows = []
for method in METHODS:
    for m in M:
        R = runs(method, m)
        if not R:
            continue
        for g in G:
            A = [rx for rx in R if S[rx]['group'] == g]
            if not A:
                continue
            conv = [rx for rx in A if R[rx][0]]
            chk = [rx for rx in conv if R[rx][1] is not None]
            ok = sum(R[rx][1] == 'route' for rx in chk)
            rows.append(dict(method=method, model=m, group=g, started=len(A), converged=len(conv), checked=len(chk),
                             passed=ok, unconverged=sum(R[rx][1] == 'unconverged' for rx in chk), pct_checked=100 * ok / len(chk) if chk else float('nan'),
                             pct_started=100 * ok / len(A) if len(chk) == len(conv) else float('nan')))
        P = [(rx, S[rx]['partner']) for rx in R if S[rx]['group'] == 'bs' and S[rx]['partner'] in R]
        if P and all(R[rx][1] is not None or not R[rx][0] for rx in R):
            okf = lambda rx: R[rx][1] == 'route'    # noqa: E731
            b = sum(okf(a) and not okf(t) for a, t in P)
            c = sum(okf(t) and not okf(a) for a, t in P)
            print(f'{method:9s} {m:10s} paired BS/twin: {len(P)} pairs, BS {sum(okf(a) for a,t in P)}, twin {sum(okf(t) for a,t in P)}, only BS {b}, only twin {c}, p = {mc(b, c):.3f}')
print('\nmethod    model      group     started conv chk pass  %conv  %started')
for r in rows:
    print('%-9s %-10s %-9s %6d %4d %4d %4d  %5.0f  %6s' % (r['method'], r['model'], r['group'], r['started'], r['converged'], r['checked'],
          r['passed'], r['pct_checked'], '%.0f' % r['pct_started'] if r['pct_started'] == r['pct_started'] else '-'))
with open(f'{RES}/search_checks.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow({k: ('%.1f' % v if isinstance(v, float) else v) for k, v in r.items()})
