"""Per reaction: which point is the more correct transition state, as far as the
data allows an answer.

Where two structures are both first-order saddles connecting the same reactant
and product, the reaction takes the lower one, so energy decides. The catch is
that a near-stationary point lying below our saddle can equally be a minimum --
an intermediate downhill of the transition state -- and a minimum is not a
candidate at all. Only a frequency calculation separates the two, so the table
marks every case where that check is missing rather than guessing.

A model geometry counts as a serious candidate when it is nearly stationary
(gradient below 0.15 eV/A, the level at which our own reference geometries sit)
and lies below our saddle. Anything with a large gradient is not a transition
state whatever its energy.
"""
import glob
import json
import os

H = '/home/energy/s242862'
HA_MEV = 27211.386
MODELS = ['UMA-S', 'UMA-M', 'eSEN']
GRAD_OK = 0.15
BAD_MODE = {'rxn1320', 'rxn4518', 'rxn5691'}   # imaginary mode not reactive

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


def our_ts(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if j.get('e_uks_final') is not None:
                return j['e_uks_final']
    return None


def our_freq(rx):
    for d in ('bs_freq', 'bs_freq_v2', 'bs_freq_fromneb'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if 'n_imag' in j:
                return j['n_imag']
    return None


def model_freq(rx, m):
    p = f'{H}/freq_at_model/{rx}_{m}/result.json'
    if os.path.exists(p):
        j = json.load(open(p))
        return j.get('n_imag')
    return None


rows = []
for rx in sorted(grp, key=lambda r: -nf[r]):
    p = f'{H}/stab_pipeline/{rx}/result.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    geo = {g['source']: g for g in d['geometries']}
    ref = geo.get('RKS-ref')
    if not ref or ref.get('ext_stable') is not False:
        continue

    def ground(g):
        if g is None or g.get('ext_stable') is None:
            return None, None
        if g['ext_stable']:
            return g.get('e_rks'), (g.get('rks_grad') or {}).get('max_evang')
        b = g.get('bs') or {}
        return b.get('e_uks'), (b.get('bs_grad') or {}).get('max_evang')

    e_ours = our_ts(rx)
    ni_ours = our_freq(rx)
    ours_ok = (ni_ours == 1) and (rx not in BAD_MODE)

    best = None
    for m in MODELS:
        e, gr = ground(geo.get(m))
        if e is None or gr is None or e_ours is None:
            continue
        de = (e - e_ours) * HA_MEV
        if gr <= GRAD_OK and de < -20:            # nearly stationary and lower
            if best is None or de < best['de']:
                best = {'model': m, 'de': de, 'grad': gr,
                        'nimag': model_freq(rx, m)}

    # verdict
    if not ours_ok and best is None:
        v = 'ungeklaert'
        why = ('unser Sattelpunkt unbestaetigt, kein Modellkandidat'
               if ni_ours != 1 else 'unser Sattelpunkt hat die falsche Mode')
    elif not ours_ok and best is not None:
        v = 'Modell besser?'
        why = (f'unser Sattelpunkt {"unbestaetigt" if ni_ours != 1 else "falsche Mode"}, '
               f'{best["model"]} liegt {abs(best["de"]):.0f} meV tiefer bei '
               f'Gradient {best["grad"]:.3f}')
    elif best is None:
        v = 'unser TS'
        why = 'bestaetigt, kein Modellkandidat tiefer und stationaer'
    elif best['nimag'] == 1:
        v = 'MODELL'
        why = (f'{best["model"]} ist bestaetigter Sattelpunkt und liegt '
               f'{abs(best["de"]):.0f} meV tiefer')
    elif best['nimag'] == 0:
        v = 'unser TS'
        why = f'{best["model"]} ist ein Minimum, kein Kandidat'
    else:
        v = 'offen'
        why = (f'{best["model"]} {abs(best["de"]):.0f} meV tiefer bei Gradient '
               f'{best["grad"]:.3f} — Frequenz fehlt')
    rows.append({'rxn': rx, 'grp': grp[rx], 'nfod': nf[rx],
                 'ni': ni_ours, 'ours_ok': ours_ok, 'best': best,
                 'verdict': v, 'why': why})

ORD = {'unser TS': 0, 'MODELL': 1, 'offen': 2, 'Modell besser?': 3,
       'ungeklaert': 4}
rows.sort(key=lambda r: (ORD[r['verdict']], -r['nfod']))

print(f"{'rxn':<9}{'Gr':<5}{'unser TS':>10}{'bester Modellkandidat':>26}"
      f"  Urteil")
print('-' * 104)
for r in rows:
    ours = ('bestaetigt' if r['ours_ok'] else
            ('falsche Mode' if r['ni'] == 1 else 'keiner'))
    b = r['best']
    bt = '—' if b is None else (f"{b['model']} {b['de']:+.0f} meV "
                                f"(g {b['grad']:.3f}"
                                + (f", {b['nimag']} imag)" if b['nimag'] is not None
                                   else ', Freq fehlt)'))
    print(f"{r['rxn']:<9}{r['grp']:<5}{ours:>10}{bt:>26}  "
          f"{r['verdict']:<15} {r['why']}")

print()
c = {}
for r in rows:
    c[r['verdict']] = c.get(r['verdict'], 0) + 1
for k in ORD:
    if k in c:
        print(f'  {k:<16}{c[k]:>3}   ' +
              ' '.join(r['rxn'] for r in rows if r['verdict'] == k))
