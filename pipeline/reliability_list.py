"""Reliability list for the 19 reactions whose reference transition state is
externally unstable: who has the transition state, and on what evidence.

A structure is the transition state of a given reaction only if it clears three
stages. Both sides are held to the same test, and it has rejected structures on
either -- ours at rxn7957, the models at rxn1147 and rxn7949.

  Stage 1  stationary, and lower in energy than the competing candidate
  Stage 2  exactly one imaginary frequency
  Stage 3  that mode moves this reaction's bonds, and neither reactive bond has
           already reached its normal length -- a bond that is done means the
           reaction is over at that point and the saddle belongs elsewhere

Stage 3 is reported with its raw numbers rather than a threshold, because the
bond-length part needs a judgement about what counts as a finished bond.
"""
import glob
import json
import os

import numpy as np
from ase.data import atomic_masses, atomic_numbers

H = '/home/energy/s242862'
HA_MEV = 27211.386
MODELS = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
          'eSEN': 'esen_neb_results'}

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

# verdicts that needed the bond-length judgement, with the reason
MANUAL = {
    'rxn1147': ('ours',
                'models sit past the transition state: the forming C1-O5 bond '
                'is at 1.497 A, a finished single bond, against 1.864 A at '
                'ours, and their mode moves it at 0.06 against our 0.94'),
    'rxn7957': ('models',
                'we sit past the transition state: C5-H7 is at 1.120 A, a '
                'finished C-H bond, and C1-H7 at 2.462 A is already detached; '
                'the models have 1.87 and 1.19 with mode rates up to 0.57 '
                'against our 0.06'),
}


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


def mode_stats(hp, gp, pairs):
    if not (hp and gp and pairs and os.path.exists(hp) and os.path.exists(gp)):
        return None
    hess = np.load(hp)
    sym, xyz = read_xyz(gp)
    m = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    w = np.repeat(1.0 / np.sqrt(m), 3)
    ev, vec = np.linalg.eigh(hess * w[:, None] * w[None, :])
    q = vec[:, int(np.argmin(ev))].reshape(-1, 3)
    q = q / np.linalg.norm(q)
    idx = sorted({i for a, b, _ in pairs for i in (a, b)})
    bonds = []
    for a, b, nm in pairs:
        u = (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b])
        bonds.append((nm, abs(float(np.dot(q[a] - q[b], u))),
                      float(np.linalg.norm(xyz[a] - xyz[b]))))
    return {'frac': float((q[idx] ** 2).sum()), 'bonds': bonds,
            'maxrate': max(b[1] for b in bonds)}


def ours(rx):
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        rp = f'{H}/{d}/{rx}/result.json'
        if not os.path.exists(rp):
            continue
        j = json.load(open(rp))
        if j.get('e_uks_final') is None:
            continue
        g = None
        for pat in ('ts', 'final', 'opt'):
            for f in glob.glob(f'{H}/{d}/{rx}/*.xyz'):
                if pat in os.path.basename(f).lower():
                    g = f
                    break
            if g:
                break
        hp = None
        ni = None
        for fd in ('bs_freq', 'bs_freq_v2', 'bs_freq_fromneb'):
            q = f'{H}/{fd}/{rx}/result.json'
            if os.path.exists(q):
                jj = json.load(open(q))
                if 'n_imag' in jj:
                    ni = jj['n_imag']
                    hp = f'{H}/{fd}/{rx}/hessian.npy'
        return {'e': j['e_uks_final'], 'geom': g, 'hess': hp, 'nimag': ni,
                'origin': d.replace('bs_tsopt_', '')}
    return None


out = []
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
    o = ours(rx)
    oms = mode_stats(o['hess'], o['geom'], pairs) if o else None

    cands = []
    for m, dn in MODELS.items():
        fp = f'{H}/freq_at_model/{rx}_{m}/result.json'
        g = geo.get(m)
        e = grad = None
        if g and g.get('ext_stable') is not None:
            e = (g.get('e_rks') if g['ext_stable']
                 else (g.get('bs') or {}).get('e_uks'))
            grad = ((g.get('rks_grad') or {}).get('max_evang') if g['ext_stable']
                    else ((g.get('bs') or {}).get('bs_grad') or {}).get('max_evang'))
        de = ((e - o['e']) * HA_MEV) if (e is not None and o) else None
        ni = ms = None
        if os.path.exists(fp):
            jj = json.load(open(fp))
            ni = jj.get('n_imag')
            ms = mode_stats(f'{H}/freq_at_model/{rx}_{m}/hessian.npy',
                            f'{H}/{dn}/{rx}/transition_state.xyz', pairs)
        cands.append({'m': m, 'de': de, 'grad': grad, 'nimag': ni, 'ms': ms})
    out.append({'rxn': rx, 'grp': grp[rx], 'nfod': nf[rx], 'pairs': pairs,
                'ours': o, 'oms': oms, 'cands': cands})


def classify(r):
    rx = r['rxn']
    if rx in MANUAL:
        who, why = MANUAL[rx]
        return ('OUR REFERENCE' if who == 'ours' else 'MODELS'), why
    o, oms = r['ours'], r['oms']
    ours_ok = o and o.get('nimag') == 1 and oms and oms['maxrate'] >= 0.05 \
        and oms['frac'] >= 0.10
    riv = [c for c in r['cands']
           if c['nimag'] == 1 and c['ms'] and c['ms']['maxrate'] >= 0.05
           and c['ms']['frac'] >= 0.10 and c['de'] is not None and c['de'] < -20]
    if ours_ok and not riv:
        near = [c for c in r['cands'] if c['de'] is not None]
        if not near:
            return 'OUR REFERENCE', 'ours clears all three stages; no model data'
        best = min(near, key=lambda c: c['de'])
        if best['de'] >= -20:
            return ('OUR REFERENCE',
                    f'ours clears all three stages; no model lies lower '
                    f'(closest {best["m"]} {best["de"]:+.0f} meV)')
        if best['grad'] is not None and best['grad'] > 0.15:
            return ('OUR REFERENCE',
                    f'ours clears all three stages; {best["m"]} lies '
                    f'{best["de"]:+.0f} meV lower but is not stationary '
                    f'(gradient {best["grad"]:.3f} eV/A)')
        return ('OUR REFERENCE',
                f'ours clears all three stages; {best["m"]} lies lower but its '
                f'mode does not belong to this reaction')
    if ours_ok and riv:
        c = min(riv, key=lambda x: x['de'])
        return 'CONTESTED', (f'both clear all three stages; {c["m"]} lies '
                             f'{c["de"]:+.0f} meV lower')
    if (not ours_ok) and riv:
        c = min(riv, key=lambda x: x['de'])
        bad = ('no converged saddle of ours' if not (o and o.get('nimag') == 1)
               else f'our mode misses the reactive bonds '
                    f'(max rate {oms["maxrate"]:.3f}, fraction {oms["frac"]:.2f})')
        return 'MODELS', (f'{bad}; {c["m"]} clears all three stages and lies '
                          f'{c["de"]:+.0f} meV lower')
    bad = ('no converged saddle of ours' if not (o and o.get('nimag') == 1)
           else (f'our mode misses the reactive bonds (max rate '
                 f'{oms["maxrate"]:.3f}, fraction {oms["frac"]:.2f})'
                 if oms else 'our mode could not be evaluated'))
    return 'UNRESOLVED', f'{bad}; no model candidate clears all three stages'


for r in out:
    r['v'], r['why'] = classify(r)

ORD = {'OUR REFERENCE': 0, 'MODELS': 1, 'CONTESTED': 2, 'UNRESOLVED': 3}
out.sort(key=lambda r: (ORD[r['v']], -r['nfod']))

print('RELIABILITY LIST — 19 reactions with an externally unstable reference')
print('=' * 100)
print()
for r in out:
    o, oms = r['ours'], r['oms']
    print(f"### {r['rxn']}   N_FOD {r['nfod']:.3f}   [{r['v']}]")
    print(f"    {r['why']}")
    print(f"    reactive bonds: " + ', '.join(nm for _, _, nm in r['pairs']))
    if o:
        s = (f"    ours       n_imag {o['nimag']}" if o['nimag'] is not None
             else "    ours       no frequency")
        if oms:
            s += (f"   mode fraction {oms['frac']:.2f}   " +
                  '  '.join(f'{nm} rate {rt:.3f} at {dd:.3f} A'
                            for nm, rt, dd in oms['bonds']))
        print(s + f"   [from {o['origin']}]")
    else:
        print('    ours       no converged saddle')
    for c in r['cands']:
        bits = []
        if c['de'] is not None:
            bits.append(f"dE {c['de']:+.0f} meV")
        if c['grad'] is not None:
            bits.append(f"grad {c['grad']:.3f}")
        if c['nimag'] is not None:
            bits.append(f"n_imag {c['nimag']}")
        if c['ms']:
            bits.append(f"fraction {c['ms']['frac']:.2f}")
            bits.append('  '.join(f'{nm} rate {rt:.3f} at {dd:.3f} A'
                                  for nm, rt, dd in c['ms']['bonds']))
        elif c['nimag'] is None:
            bits.append('frequency not computed')
        print(f"    {c['m']:<10} " + '   '.join(bits))
    print()

print('=' * 100)
c = {}
for r in out:
    c.setdefault(r['v'], []).append(r['rxn'])
for k in ORD:
    if k in c:
        print(f'{k:<16}{len(c[k]):>3}   ' + ' '.join(c[k]))
