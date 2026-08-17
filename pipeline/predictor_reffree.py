"""Reference-free predictor test.

The earlier AUC scored the predictor against the RMSD to the RKS-TS.  On the
unstable side that reference is not a saddle point of the surface the reaction
runs on (T0), so "deviation" there is distance, not error -- and the test came
close to circular: instability of the restricted solution predicting departure
from the restricted saddle point.

This version drops the reference entirely.  The label is whether the model's
OWN structure is a stationary point of the ground-state surface, measured as
max|F| at that geometry from the two-stage recipe (STABPerform -> EnGrad
MORead).  No comparison structure enters.
"""
import json, os, math, glob
import numpy as np

H = '/home/energy/s242862'
EVA = 51.42208
STAT = 0.15
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}
MR19 = set(('rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn6196 rxn0346 rxn4518 '
            'rxn3107 rxn8837 rxn7060 rxn5691 rxn1283 rxn8827 rxn4522 rxn1147 '
            'rxn0894 rxn7957 rxn5690').split())


def auc(scores, labels):
    """Mann-Whitney AUC, identical to pipeline/sep_analysis.py."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def maxforce(label):
    for d in (f'{H}/orca_freq/{label}', f'{H}/orca_irc/{label}'):
        p = f'{d}/engrad.out'
        if not os.path.exists(p):
            continue
        t = open(p, errors='replace').read()
        i = t.find('CARTESIAN GRADIENT')
        if i < 0:
            continue
        G = []
        for line in t[i:].split('\n')[3:]:
            f = line.split()
            if len(f) < 6:
                break
            try:
                G.append([float(v) for v in f[3:6]])
            except ValueError:
                break
        if G:
            G = np.array(G) * EVA
            return float(np.abs(G).max())
    return None


nfod = {r['rxn']: r['nfod']
        for r in json.load(open(f'{H}/fod_ranking.json'))['results']}

rows = []
for p in sorted(glob.glob(f'{H}/stab_pipeline/rxn*/result.json')):
    rx = os.path.basename(os.path.dirname(p))
    try:
        g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    except Exception:
        continue
    if not g or g.get('ext_stable') is None or g.get('lmin_ext') is None:
        continue
    for m, dn in MODELDIR.items():
        if not os.path.exists(f'{H}/{dn}/{rx}/transition_state.xyz'):
            continue
        f = maxforce(f'{rx}_{m}')
        if f is None:
            continue
        rows.append({'rx': rx, 'model': m, 'f': f,
                     'stable': g['ext_stable'], 'lmin': g['lmin_ext'],
                     'nfod': nfod.get(rx), 'mr': rx in MR19})

y = [r['f'] >= STAT for r in rows]
print('REFERENZFREIER PRAEDIKTORTEST')
print('=' * 78)
print('Ziel: die Modellstruktur ist KEIN Stationaerpunkt der Grundzustands-')
print('flaeche, also max|F| >= %.2f eV/A an ihrer eigenen Geometrie.' % STAT)
print('Kein Vergleichspunkt geht ein.')
print()
print('  Zeilen: %d   davon nicht stationaer: %d (%.0f %%)'
      % (len(rows), sum(y), 100 * sum(y) / len(rows)))
print('  Reaktionen: %d   Modelle: %d'
      % (len({r['rx'] for r in rows}), len({r['model'] for r in rows})))
print()

lam = [r for r in rows if r['lmin'] is not None]
fod = [r for r in rows if r['nfod'] is not None]
print('%-32s %8s %6s' % ('Praediktor', 'AUC', 'n'))
print('-' * 50)
print('%-32s %8.3f %6d'
      % ('-lambda_min_ext (kontinuierlich)',
         auc([-r['lmin'] for r in lam], [r['f'] >= STAT for r in lam]), len(lam)))
print('%-32s %8.3f %6d'
      % ('instabil ja/nein (binaer)',
         auc([0.0 if r['stable'] else 1.0 for r in rows], y), len(rows)))
print('%-32s %8.3f %6d'
      % ('N_FOD (kontinuierlich)',
         auc([r['nfod'] for r in fod], [r['f'] >= STAT for r in fod]), len(fod)))
print()
print('je Modell, AUC von -lambda_min_ext')
for m in MODELDIR:
    sub = [r for r in lam if r['model'] == m]
    a = auc([-r['lmin'] for r in sub], [r['f'] >= STAT for r in sub])
    print('   %-8s %s   n=%d, davon nicht stationaer %d'
          % (m, ('%.3f' % a) if a is not None else '   -', len(sub),
             sum(1 for r in sub if r['f'] >= STAT)))

print()
print('Anteil gueltiger Stationaerpunkte, nach Praediktor geteilt')
for lab, sel in (('RKS stabil', lambda r: r['stable']),
                 ('RKS instabil', lambda r: not r['stable'])):
    sub = [r for r in rows if sel(r)]
    ok = sum(1 for r in sub if r['f'] < STAT)
    med = float(np.median([r['f'] for r in sub]))
    print('   %-14s %3d Strukturen   stationaer %3d (%3.0f %%)   Median max|F| %.3f'
          % (lab, len(sub), ok, 100 * ok / len(sub), med))

print()
print('Gegenprobe zum Zirkel-Einwand: nur Zeilen mit hohem N_FOD')
thr = 0.5
hi = [r for r in rows if (r['nfod'] or 0) > thr]
for lab, sel in (('RKS stabil', lambda r: r['stable']),
                 ('RKS instabil', lambda r: not r['stable'])):
    sub = [r for r in hi if sel(r)]
    if not sub:
        continue
    ok = sum(1 for r in sub if r['f'] < STAT)
    print('   N_FOD > %.1f, %-14s %3d Strukturen   stationaer %3d (%3.0f %%)'
          % (thr, lab, len(sub), ok, 100 * ok / len(sub)))
