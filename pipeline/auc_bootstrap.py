"""Bootstrap-Konfidenzintervalle fuer den referenzfreien Praediktortest (S1).

Die Punktschaetzer stammen aus pipeline/predictor_reffree.py und werden hier
zuerst nachgerechnet -- stimmen sie nicht, bricht das Skript ab, denn dann ist
die Datenbasis eine andere.

Geclustert nach REAKTION, nicht nach Zeile: die drei Modellzeilen einer
Reaktion teilen denselben Praediktorwert und sind korreliert.  Naives
Zeilen-Resampling taete so, als lieferten sie drei unabhaengige Beobachtungen,
und lieferte zu enge Intervalle.
"""
import json, os, glob, sys
import numpy as np

H = '/home/energy/s242862'
EVA = 51.42208
STAT = 0.15
NBOOT = 10000
SEED = 20260819
EXPECT = {'lam': 0.836, 'bin': 0.829, 'fod': 0.776}
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}


def auc(scores, labels):
    """Mann-Whitney-AUC mit Bindungskorrektur, identisch zu sep_analysis.py."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


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
            return float(np.abs(np.array(G) * EVA).max())
    return None


# ---------------------------------------------------------------- Datenbasis
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
        rows.append({'rx': rx, 'model': m, 'f': f, 'stable': g['ext_stable'],
                     'lmin': g['lmin_ext'], 'nfod': nfod.get(rx)})

L = [i for i, r in enumerate(rows) if r['lmin'] is not None]
F = [i for i, r in enumerate(rows) if r['nfod'] is not None]
BOTH = sorted(set(L) & set(F))

y = np.array([r['f'] >= STAT for r in rows])
lam = np.array([-(r['lmin'] if r['lmin'] is not None else np.nan) for r in rows])
fod = np.array([(r['nfod'] if r['nfod'] is not None else np.nan) for r in rows])
bnr = np.array([0.0 if r['stable'] else 1.0 for r in rows])
rxa = np.array([r['rx'] for r in rows])
RXN = sorted(set(rxa))
IDX = {rx: np.flatnonzero(rxa == rx) for rx in RXN}

print('Datenbasis')
print('  Zeilen %d   Reaktionen %d   Modelle %d   Positive (max|F| >= %.2f) %d'
      % (len(rows), len(RXN), len(MODELDIR), STAT, int(y.sum())))

# ------------------------------------------------------------ Reproduktion
got = {'lam': auc(lam[L], y[L]), 'bin': auc(bnr, y), 'fod': auc(fod[F], y[F])}
print()
print('Reproduktion der Punktschaetzer')
bad = []
for k, want in EXPECT.items():
    ok = got[k] is not None and abs(got[k] - want) < 5e-4
    print('  %-5s erwartet %.3f   erhalten %.4f   %s'
          % (k, want, got[k], 'ok' if ok else 'ABWEICHUNG'))
    if not ok:
        bad.append(k)
if bad:
    sys.exit('ABBRUCH: %s weichen ab -- die Datenbasis ist nicht dieselbe.'
             % ', '.join(bad))

# Punktschaetzer auf der gemeinsamen Teilmenge, auf der ADelta definiert ist
a_lam_c, a_fod_c = auc(lam[BOTH], y[BOTH]), auc(fod[BOTH], y[BOTH])

# --------------------------------------------------------------- Bootstrap
rng = np.random.default_rng(SEED)
bl, bf, bd = [], [], []
skipped = 0
for _ in range(NBOOT):
    pick = rng.integers(0, len(RXN), len(RXN))
    idx = np.concatenate([IDX[RXN[j]] for j in pick])
    yy = y[idx]
    if yy.all() or not yy.any():        # nur eine Klasse -- AUC undefiniert
        skipped += 1
        continue
    al, af = auc(lam[idx], yy), auc(fod[idx], yy)
    bl.append(al)
    bf.append(af)
    bd.append(al - af)                  # gepaart: dasselbe Resample
bl, bf, bd = np.array(bl), np.array(bf), np.array(bd)


def ci(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


lo_l, hi_l = ci(bl)
lo_f, hi_f = ci(bf)
lo_d, hi_d = ci(bd)
frac = float((bd > 0).mean())

out = []
w = out.append
w('BOOTSTRAP-KONFIDENZINTERVALLE, referenzfreier Praediktortest')
w('=' * 72)
w('')
w('Zeilen %d = %d Reaktionen x %d Modelle, davon %d nicht stationaer'
  % (len(rows), len(RXN), len(MODELDIR), int(y.sum())))
w('Ziel y = 1, wenn max|F|_DFT >= %.2f eV/A an der Modellgeometrie' % STAT)
w('Resampling geclustert nach Reaktion (alle 3 Modellzeilen zusammen),')
w('%d Iterationen, Seed %d, %d wegen fehlender zweiter Klasse verworfen.'
  % (NBOOT, SEED, skipped))
w('')
w('%-34s %8s   %-16s' % ('Praediktor', 'AUC', '95 %-CI'))
w('-' * 62)
w('%-34s %8.3f   [%.3f, %.3f]'
  % ('-lambda_min_ext (kontinuierlich)', got['lam'], lo_l, hi_l))
w('%-34s %8.3f   [%.3f, %.3f]' % ('N_FOD (kontinuierlich)', got['fod'], lo_f, hi_f))
w('%-34s %8.3f   %-16s' % ('instabil ja/nein (binaer)', got['bin'], '(nicht gebootstrappt)'))
w('')
w('%-34s %8.3f   [%.3f, %.3f]'
  % ('Delta AUC = -lambda_min - N_FOD', a_lam_c - a_fod_c, lo_d, hi_d))
w('%-34s %8.3f' % ('Anteil Resamples mit Delta > 0', frac))
w('')
w('Das Delta-CI %s die Null.'
  % ('enthaelt' if lo_d <= 0 <= hi_d else 'enthaelt NICHT'))
w('')
w('Erzeugt von pipeline/auc_bootstrap.py; Punktschaetzer gegen')
w('pipeline/predictor_reffree.py geprueft (%.4f / %.4f / %.4f).'
  % (got['lam'], got['bin'], got['fod']))
txt = '\n'.join(out)
print()
print(txt)
os.makedirs(f'{H}/results', exist_ok=True)
open(f'{H}/results/auc_bootstrap.txt', 'w').write(txt + '\n')
