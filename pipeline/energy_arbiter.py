"""Which candidate point is the more correct transition state?

Where two structures are both genuine first-order saddles connecting the same
reactant and product, the reaction takes the lower one. Energy is therefore a
valid arbiter -- but only between genuine saddles. At a point that is not
stationary the energy says nothing about transition states, so the table carries
the gradient and the frequency status alongside every energy.

All energies are ground-state energies at the given geometry, wB97M-V/def2-TZVP
in PySCF: the broken-symmetry solution where the restricted one is externally
unstable, RKS where it is stable. Same code and basis throughout, so differences
are meaningful at the meV level.

The NEB transition states are the gap: their energies exist only from ORCA and
are not comparable at that resolution. Marked accordingly.
"""
import glob
import json
import os

import numpy as np

H = '/home/energy/s242862'
HA_MEV = 27211.386
MODELS = ['UMA-S', 'UMA-M', 'eSEN']

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


def freq_of(rx, sub=None):
    """(n_imag, mode fraction, largest bond rate) for our saddle."""
    for d in ('bs_freq', 'bs_freq_v2', 'bs_freq_fromneb'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if 'n_imag' in j:
                return j['n_imag'], j.get('imag_freq', [None])[0]
    return None, None


def model_freq(rx, m):
    p = f'{H}/freq_at_model/{rx}_{m}/result.json'
    if os.path.exists(p):
        j = json.load(open(p))
        if 'n_imag' in j:
            return j
    return None


def our_ts(rx):
    """(energy, source) of our optimised saddle."""
    for d in ('bs_tsopt_fromneb', 'bs_tsopt_v2', 'bs_tsopt_batch'):
        p = f'{H}/{d}/{rx}/result.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if j.get('e_uks_final') is not None:
                return j['e_uks_final'], d
    return None, None


def mode_ok(rx):
    """Did the imaginary mode of our saddle stretch the reactive bonds?
    Recomputed here would need the Hessian; the verdicts are taken from the
    earlier run and hard-coded only where they were negative."""
    return rx not in ('rxn1320', 'rxn4518', 'rxn5691')


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
        """(energy, surface, gradient) of the ground state at this geometry."""
        if g is None or g.get('ext_stable') is None:
            return None, None, None
        if g['ext_stable']:
            return (g.get('e_rks'), 'RKS',
                    (g.get('rks_grad') or {}).get('max_evang'))
        b = g.get('bs') or {}
        return (b.get('e_uks'), 'BS', (b.get('bs_grad') or {}).get('max_evang'))

    cands = []
    e, s, gr = ground(ref)
    cands.append({'name': 'RKS-Referenz', 'e': e, 'surf': s, 'grad': gr,
                  'nimag': None, 'note': 'Ausgangsgeometrie des Benchmarks'})

    e_ours, src = our_ts(rx)
    ni, im = freq_of(rx)
    cands.append({'name': 'unser BS-TS', 'e': e_ours, 'surf': 'BS',
                  'grad': None, 'nimag': ni,
                  'note': ('Mode nicht reaktiv' if not mode_ok(rx)
                           else ('' if src is None else src.replace('bs_tsopt_', '')))})

    for m in MODELS:
        e, s, gr = ground(geo.get(m))
        mf = model_freq(rx, m)
        cands.append({'name': m, 'e': e, 'surf': s, 'grad': gr,
                      'nimag': (mf or {}).get('n_imag'),
                      'note': ('Frequenz gerechnet' if mf else
                               'nicht als Sattelpunkt geprueft')})

    if os.path.exists(f'{H}/bs_uks_neb_results/{rx}/bs_uks_neb_NEB-TS_converged.xyz'):
        cands.append({'name': 'ORCA-NEB-TS', 'e': None, 'surf': None,
                      'grad': None, 'nimag': None,
                      'note': 'Energie nur aus ORCA, nicht vergleichbar'})

    rows.append({'rxn': rx, 'grp': grp[rx], 'nfod': nf[rx], 'cands': cands})

# ------------------------------------------------------------------ output
print('Energien in meV relativ zu unserem BS-Sattelpunkt. Negativ = tiefer.')
print('Ein Punkt ist nur dann TS-Kandidat, wenn er stationaer ist (Gradient')
print('nahe null) UND eine imaginaere Frequenz hat.\n')

for r in rows:
    e0 = next((c['e'] for c in r['cands'] if c['name'] == 'unser BS-TS'), None)
    print(f"### {r['rxn']}  ({r['grp']}, N_FOD {r['nfod']:.3f})")
    print(f"{'Kandidat':<16}{'Flaeche':>8}{'dE [meV]':>11}{'max|g|':>9}"
          f"{'n_imag':>8}  Anmerkung")
    for c in r['cands']:
        de = ('—' if (c['e'] is None or e0 is None)
              else f"{(c['e'] - e0) * HA_MEV:+.1f}")
        g = '—' if c['grad'] is None else '{:.3f}'.format(c['grad'])
        ni = '—' if c['nimag'] is None else str(c['nimag'])
        print('{:<16}{:>8}{:>11}{:>9}{:>8}  {}'.format(
            c['name'], c['surf'] or '—', de, g, ni, c['note']))
    print()
