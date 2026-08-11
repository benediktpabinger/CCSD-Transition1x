"""How far above the correct saddle does the model geometry sit, in each group?

Same quantity on both sides, so the two groups are directly comparable. For the
externally stable reactions the RKS reference is the correct saddle; for the
multireference ones it is our confirmed broken-symmetry saddle. Energies are
ground-state energies at the model geometry, on whichever surface applies.

A model that has found the saddle sits a few meV above it -- moving off a saddle
raises the energy in every direction except along the reaction coordinate. A
large positive value means the geometry is far off; a large negative value means
it is downhill of the saddle, so not a transition state at all.

Only the eleven reactions where our saddle survived all three stages enter the
multireference group. Where our own structure is unconfirmed there is nothing to
measure against.
"""
import json
import os

import numpy as np

H = '/home/energy/s242862'
HA_MEV = 27211.386
MODELS = ['UMA-S', 'UMA-M', 'eSEN']
CONFIRMED = ['rxn8832', 'rxn4113', 'rxn8885', 'rxn6196', 'rxn0346', 'rxn3107',
             'rxn8837', 'rxn7060', 'rxn8827', 'rxn1147', 'rxn0894']

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


def our_ts_e(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if j.get('e_uks_final') is not None:
                return j['e_uks_final']
    return None


def ground(g):
    if g is None or g.get('ext_stable') is None:
        return None, None
    if g['ext_stable']:
        return g.get('e_rks'), (g.get('rks_grad') or {}).get('max_evang')
    b = g.get('bs') or {}
    return b.get('e_uks'), (b.get('bs_grad') or {}).get('max_evang')


rows = []
for rx in grp:
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    geo = {g['source']: g for g in d['geometries']}
    ref = geo.get('RKS-ref')
    if not ref or ref.get('ext_stable') is None:
        continue
    unstable = ref['ext_stable'] is False

    if unstable:
        if rx not in CONFIRMED:
            continue                      # no trustworthy saddle to measure from
        e0 = our_ts_e(rx)
        cls = 'MR'
    else:
        e0, _ = ground(ref)
        cls = 'einfach'
    if e0 is None:
        continue

    for m in MODELS:
        e, gr = ground(geo.get(m))
        if e is None:
            continue
        rows.append({'rxn': rx, 'cls': cls, 'model': m,
                     'de': (e - e0) * HA_MEV, 'grad': gr})

print(f'{len(rows)} Modellgeometrien\n')
print('dE = Energie an der Modellgeometrie minus Energie am richtigen '
      'Sattelpunkt, in meV.\n')

print(f"{'Klasse':<10}{'n':>5}{'median':>10}{'|median|':>10}{'Q1':>10}"
      f"{'Q3':>10}{'min':>11}{'max':>11}")
print('-' * 77)
for cls in ('einfach', 'MR'):
    v = np.array([r['de'] for r in rows if r['cls'] == cls])
    if not len(v):
        continue
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    print(f'{cls:<10}{len(v):>5}{q2:>10.1f}{np.median(np.abs(v)):>10.1f}'
          f'{q1:>10.1f}{q3:>10.1f}{v.min():>11.1f}{v.max():>11.1f}')

print(f"\n{'Modell':<12}{'einfach |dE|':>14}{'MR |dE|':>10}{'Faktor':>9}")
for m in MODELS:
    a = np.abs([r['de'] for r in rows if r['model'] == m and r['cls'] == 'einfach'])
    b = np.abs([r['de'] for r in rows if r['model'] == m and r['cls'] == 'MR'])
    if not (len(a) and len(b)):
        continue
    ma, mb = float(np.median(a)), float(np.median(b))
    print(f'{m:<12}{ma:>14.1f}{mb:>10.1f}{mb/ma:>9.1f}')

print('\n=== Verteilung |dE| ===')
for cls in ('einfach', 'MR'):
    v = np.abs([r['de'] for r in rows if r['cls'] == cls])
    if not len(v):
        continue
    bins = [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, 1e9)]
    line = f'  {cls:<9}'
    for lo, hi in bins:
        k = int(((v >= lo) & (v < hi)).sum())
        line += f'  {lo}-{hi if hi < 1e8 else "∞"}: {k:>3}'
    print(line)

print('\n=== MR-Gruppe je Reaktion, |dE| in meV ===')
print(f"{'rxn':<10}" + ''.join(f'{m:>10}' for m in MODELS))
for rx in CONFIRMED:
    sub = {r['model']: r['de'] for r in rows if r['rxn'] == rx}
    if not sub:
        continue
    print(f'{rx:<10}' + ''.join(
        f"{sub[m]:>10.1f}" if m in sub else f"{'—':>10}" for m in MODELS))
