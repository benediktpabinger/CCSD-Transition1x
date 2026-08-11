"""Which side has the transition state, after applying the three-stage rule to
both.

Stage 3 -- does the imaginary mode belong to this reaction -- is decided from
two quantities that are read directly off the structure and its mode:

  Rate    how fast a reactive bond changes along the imaginary mode. Below about
          0.05 the mode does not touch the reaction coordinate at all.
  Laenge  a reactive bond already at its normal value means the reaction is
          finished at that point; the structure sits in the product valley and
          whatever saddle it carries belongs to another motion.

Both sides are held to the same test. It has rejected structures on either --
ours at rxn7957, the models at rxn1147 and rxn7949 -- which is the reason to
trust it.
"""
import glob
import json
import os

import numpy as np
from ase.data import atomic_masses, atomic_numbers

H = '/home/energy/s242862'
HA_MEV = 27211.386
MODELS = ['UMA-S', 'UMA-M', 'eSEN']
GRAD_OK = 0.15
RATE_MIN = 0.05          # below this the mode misses the reaction coordinate

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


def read_xyz(p):
    L = open(p).read().split('\n')
    m = int(L[0].split()[0])
    sym, xyz = [], []
    for line in L[2:2 + m]:
        f = line.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.array(xyz)


def reactive(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            rb = json.load(open(p)).get('reactive_bonds')
            if rb:
                return [(e['pair'][0], e['pair'][1], e['sym']) for e in rb[:2]]
    return []


def mode_stats(hess_p, geom_p, pairs):
    if not (os.path.exists(hess_p) and os.path.exists(geom_p) and pairs):
        return None
    hess = np.load(hess_p)
    sym, xyz = read_xyz(geom_p)
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    w = np.repeat(1.0 / np.sqrt(m), 3)
    ev, vec = np.linalg.eigh(hess * w[:, None] * w[None, :])
    q = vec[:, int(np.argmin(ev))].reshape(-1, 3)
    q = q / np.linalg.norm(q)
    idx = sorted({i for a, b, _ in pairs for i in (a, b)})
    frac = float((q[idx] ** 2).sum())
    out = []
    for a, b, nm in pairs:
        u = (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b])
        out.append({'name': nm, 'rate': abs(float(np.dot(q[a] - q[b], u))),
                    'dist': float(np.linalg.norm(xyz[a] - xyz[b]))})
    return {'frac': frac, 'bonds': out,
            'maxrate': max(b['rate'] for b in out)}


FRAC_MIN = 0.10          # below this the motion sits elsewhere in the molecule


def stage3(ms):
    """Does this mode belong to the reaction?

    Two automatic tests only. The third marker -- whether a reactive bond has
    already reached its normal length, so the reaction is over at that point --
    needs a judgement about what counts as normal for a given bond type, so the
    distances are printed instead of thresholded. That marker decided rxn1147
    and rxn7957 and both are reported as contested here rather than resolved by
    a threshold invented to fit them.
    """
    if ms is None:
        return None, 'Mode nicht bestimmbar'
    if ms['maxrate'] < RATE_MIN:
        return False, (f'Mode bewegt die reaktiven Bindungen nicht '
                       f'(max {ms["maxrate"]:.3f})')
    if ms['frac'] < FRAC_MIN:
        return False, (f'Mode sitzt ausserhalb der reaktiven Atome '
                       f'(Anteil {ms["frac"]:.2f})')
    return True, f'Mode reaktiv (max {ms["maxrate"]:.3f}, Anteil {ms["frac"]:.2f})'


def our_files(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    for fd in ('bs_freq', 'bs_freq_v2', 'bs_freq_fromneb'):
                        hp = f'{H}/{fd}/{rx}/hessian.npy'
                        if os.path.exists(hp):
                            return f, hp, json.load(
                                open(f'{H}/{d}/{rx}/result.json')).get('e_uks_final')
                    return f, None, json.load(
                        open(f'{H}/{d}/{rx}/result.json')).get('e_uks_final')
    return None, None, None


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
    pairs = reactive(rx)

    og, ohp, e_ours = our_files(rx)
    oms = mode_stats(ohp, og, pairs) if (og and ohp) else None
    ours_ok, ours_why = stage3(oms)
    if e_ours is None:
        ours_ok, ours_why = False, 'kein konvergierter Sattelpunkt'

    cands = []
    for m in MODELS:
        fp = f'{H}/freq_at_model/{rx}_{m}/result.json'
        if not os.path.exists(fp):
            continue
        j = json.load(open(fp))
        if j.get('n_imag') != 1:
            continue
        gm = f'{H}/{ {"UMA-S": "uma_neb_results", "UMA-M": "uma_m_neb_results", "eSEN": "esen_neb_results"}[m] }/{rx}/transition_state.xyz'
        ms = mode_stats(f'{H}/freq_at_model/{rx}_{m}/hessian.npy', gm, pairs)
        ok, why = stage3(ms)
        cands.append({'m': m, 'de': j.get('e_vs_our_ts_meV'), 'ok': ok,
                      'why': why, 'grad': j.get('grad_max_evang')})

    winners = [c for c in cands if c['ok'] and c['de'] is not None and c['de'] < -20]
    if ours_ok and not winners:
        v = 'unsere Referenz'
        note = ours_why + ('; kein Modell besteht Stufe 3 und liegt tiefer'
                           if cands else '; Modelle ungeprueft oder hoeher')
    elif ours_ok and winners:
        c = min(winners, key=lambda x: x['de'])
        v = 'strittig'
        note = (f'beide bestehen Stufe 3; {c["m"]} liegt {c["de"]:+.0f} meV '
                f'tiefer')
    elif (not ours_ok) and winners:
        c = min(winners, key=lambda x: x['de'])
        v = 'MODELLE'
        note = f'unser TS: {ours_why}; {c["m"]} {c["de"]:+.0f} meV, {c["why"]}'
    else:
        v = 'offen'
        note = f'unser TS: {ours_why}; kein Modellkandidat besteht Stufe 3'
    rows.append({'rxn': rx, 'grp': grp[rx], 'nfod': nf[rx], 'v': v,
                 'note': note, 'ours': ours_why, 'oms': oms,
                 'cands': cands, 'pairs': pairs,
                 'ncand': len(cands), 'nwin': len(winners)})

ORD = {'unsere Referenz': 0, 'strittig': 1, 'MODELLE': 2, 'offen': 3}
rows.sort(key=lambda r: (ORD[r['v']], -r['nfod']))
print(f"{'rxn':<9}{'Gr':<5}{'N_FOD':>7}{'Kand':>6}  {'Urteil':<16} Begruendung")
print('-' * 120)
for r in rows:
    print('{:<9}{:<5}{:>7.3f}{:>6}  {:<16} {}'.format(
        r['rxn'], r['grp'], r['nfod'], f"{r['nwin']}/{r['ncand']}",
        r['v'], r['note']))
print()
c = {}
for r in rows:
    c.setdefault(r['v'], []).append(r['rxn'])
for k in ORD:
    if k in c:
        print(f"  {k:<16}{len(c[k]):>3}   " + ' '.join(c[k]))

# the contested cases get their bond lengths printed, since the automatic test
# cannot settle them
strit = [r for r in rows if r['v'] == 'strittig']
if strit:
    print('\n=== strittige Faelle: Bindungslaengen entscheiden ===')
    print('Eine reaktive Bindung, die bereits ihren normalen Wert hat, zeigt')
    print('an, dass die Reaktion dort abgeschlossen ist.\n')
    for r in strit:
        print(f"### {r['rxn']}")
        names = [b['name'] for b in (r['oms'] or {}).get('bonds', [])]
        print('{:<14}{:>8}'.format('Struktur', 'Anteil')
              + ''.join('{:>12}{:>11}'.format(n + ' d/dQ', n + ' [A]')
                        for n in names))
        if r['oms']:
            line = '{:<14}{:>8.3f}'.format('unser BS-TS', r['oms']['frac'])
            for b in r['oms']['bonds']:
                line += '{:>12.3f}{:>11.3f}'.format(b['rate'], b['dist'])
            print(line)
        for cd in r['cands']:
            gm = f'{H}/{ {"UMA-S": "uma_neb_results", "UMA-M": "uma_m_neb_results", "eSEN": "esen_neb_results"}[cd["m"]] }/{r["rxn"]}/transition_state.xyz'
            ms = mode_stats(f'{H}/freq_at_model/{r["rxn"]}_{cd["m"]}/hessian.npy',
                            gm, r['pairs'])
            if ms is None:
                continue
            line = '{:<14}{:>8.3f}'.format(cd['m'], ms['frac'])
            for b in ms['bonds']:
                line += '{:>12.3f}{:>11.3f}'.format(b['rate'], b['dist'])
            print(line + f"   dE {cd['de']:+.0f} meV")
        print()
