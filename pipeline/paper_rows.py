# DEPRECATED -- TZVP-era figures/tables, superseded by
# pipeline/omol25_model_geoms.py, pipeline/hinge_tables.py and
# pipeline/plot_omol25_figs.py. Do not run for the paper; retained as history.
# The 1.697 eV/A median is obsolete; successor numbers live in
# results/hinge_t1x.csv (1.636) and results/hinge_omol25.csv (1.870).

"""Assembliert die beiden CSVs fuer das Workshop-Paper.

results/paper_rows.csv   122 Zeilen = 42 Reaktionen x 3 Modelle, dieselbe
                         Datenbasis wie predictor_reffree.py und
                         auc_bootstrap.py -- die Zeilenauswahl ist dieselbe
                         Schleife, Zeichen fuer Zeichen.
results/hinge_rows.csv   19 MR-Reaktionen, max|F| des RKS-TS auf beiden
                         Flaechen (die Tabelle in Abschnitt 6).

Alles wird aus den Rohdaten gerechnet, nichts aus dem Kapitel abgeschrieben;
die Kapitelwerte stehen nur als Gegenprobe im Code. Schlaegt eine Pruefung
fehl, bricht das Skript ab.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

H = '/home/energy/s242862'
EVA = 51.42208
STAT = 0.15
OUT = f'{H}/results'
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}
SLUG = {'UMA-S': 'uma-s', 'UMA-M': 'uma-m', 'eSEN': 'esen'}
MR19 = ('rxn7949 rxn8832 rxn1320 rxn4113 rxn8885 rxn6196 rxn0346 rxn4518 '
        'rxn3107 rxn8837 rxn7060 rxn5691 rxn1283 rxn8827 rxn4522 rxn1147 '
        'rxn0894 rxn7957 rxn5690').split()

# Gegenprobe: die Tabelle, wie sie im Kapitel steht (Abschnitt 6).
CH_HINGE = {
    'rxn7949': (0.105, 1.686), 'rxn8832': (0.142, 2.733), 'rxn1320': (0.059, 2.073),
    'rxn4113': (0.079, 0.386), 'rxn8885': (0.042, 2.637), 'rxn6196': (0.179, 0.638),
    'rxn0346': (0.052, 2.613), 'rxn4518': (0.068, 2.949), 'rxn3107': (0.063, 1.646),
    'rxn8837': (0.057, 1.697), 'rxn7060': (0.033, 1.766), 'rxn5691': (0.041, 1.419),
    'rxn1283': (0.038, 2.386), 'rxn8827': (0.026, 1.128), 'rxn4522': (0.098, 1.875),
    'rxn1147': (0.065, 1.840), 'rxn0894': (0.062, 1.350), 'rxn7957': (0.026, 0.901),
    'rxn5690': (0.037, 0.162)}

fails, notes = [], []


def check(ok, msg):
    (notes if ok else fails).append(('ok   ' if ok else 'FEHL ') + msg)


def auc(scores, labels):
    """Mann-Whitney-AUC mit Bindungskorrektur, identisch zu sep_analysis.py."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
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


def model_maxforce(p):
    """Die Kraft, die das Modell selbst am Climbing Image gemeldet hat --
    die letzten drei Spalten des extxyz, das der ASE-NEB geschrieben hat."""
    L = open(p, errors='replace').read().split('\n')
    n = int(L[0].split()[0])
    if 'forces' not in L[1]:
        return None
    F = []
    for line in L[2:2 + n]:
        f = line.split()
        if len(f) < 7:
            return None
        F.append([float(x) for x in f[4:7]])
    return float(np.abs(np.array(F)).max()) if len(F) == n else None


def dft_maxforce(label):
    """max|F| per DFT an derselben unveraenderten Geometrie (Stufe 1).
    ORCA druckt dE/dx; die Kraft ist minus davon, fuer |max| gleichgueltig."""
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


# ----------------------------------------------------------- paper_rows.csv
nfod = {r['rxn']: r['nfod']
        for r in json.load(open(f'{H}/fod_ranking.json'))['results']}

rows, no_depth = [], []
for p in sorted(glob.glob(f'{H}/stab_pipeline/rxn*/result.json')):
    rx = os.path.basename(os.path.dirname(p))
    try:
        g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    except Exception:
        continue
    if not g or g.get('ext_stable') is None or g.get('lmin_ext') is None:
        continue
    # Brechungstiefe: E(RKS) - E(BS) am RKS-TS. bs.de_meV ist E(BS)-E(RKS),
    # also negativ, wenn die gebrochene Loesung tiefer liegt. Stabil -> 0.
    bs = g.get('bs')
    if g['ext_stable']:
        depth = 0.0
    elif bs and bs.get('de_meV') is not None:
        depth = -float(bs['de_meV'])
    else:
        depth = None
        no_depth.append(rx)
    for m, dn in MODELDIR.items():
        xyz = f'{H}/{dn}/{rx}/transition_state.xyz'
        if not os.path.exists(xyz):
            continue
        fd = dft_maxforce(f'{rx}_{m}')
        if fd is None:
            continue
        rows.append({'rxn': rx, 'model': SLUG[m], 'F_model': model_maxforce(xyz),
                     'F_dft': fd, 'unstable': 0 if g['ext_stable'] else 1,
                     'lambda_min': g['lmin_ext'], 'nfod': nfod.get(rx),
                     'breaking_depth': depth})

y = np.array([r['F_dft'] >= STAT for r in rows])
st = np.array([r['unstable'] == 0 for r in rows])
missing_fm = [r for r in rows if r['F_model'] is None]

# ---- Validierung 1: Umfang
check(len(rows) == 122, 'n = %d (erwartet 122)' % len(rows))
check(int(y.sum()) == 29, 'F_dft >= %.2f: %d (erwartet 29)' % (STAT, int(y.sum())))
# ---- Validierung 4: Gruppengroessen
check(int(st.sum()) == 78 and int((~st).sum()) == 44,
      'stabil %d / instabil %d (erwartet 78 / 44)' % (st.sum(), (~st).sum()))
# ---- Validierung 2: AUCs
a = {'lam': auc([-r['lambda_min'] for r in rows], y),
     'bin': auc([float(r['unstable']) for r in rows], y),
     'fod': auc([r['nfod'] for r in rows], y)}
for k, want in (('lam', 0.836), ('bin', 0.829), ('fod', 0.776)):
    check(abs(a[k] - want) < 5e-4, 'AUC %-3s = %.4f (erwartet %.3f)' % (k, a[k], want))
# ---- Validierung 3: Mediane
check(not missing_fm, 'F_model fuer alle Zeilen vorhanden (%d fehlen)' % len(missing_fm))
med = {}
for lab, sel in (('stabil', st), ('instabil', ~st)):
    for col in ('F_model', 'F_dft'):
        med[(col, lab)] = float(np.median([rows[i][col] for i in np.flatnonzero(sel)]))
for key, want in ((('F_model', 'stabil'), 0.032), (('F_model', 'instabil'), 0.032),
                  (('F_dft', 'stabil'), 0.067), (('F_dft', 'instabil'), 0.163)):
    check(abs(med[key] - want) < 5e-4,
          'Median %-7s %-8s = %.4f (erwartet %.3f)' % (key[0], key[1], med[key], want))

# ------------------------------------------------------------ hinge_rows.csv
hinge, hmis = [], []
for rx in MR19:
    g = {x['source']: x for x in
         json.load(open(f'{H}/stab_pipeline/{rx}/result.json'))['geometries']}['RKS-ref']
    f_rks = g['rks_grad']['max_evang']
    f_bs = g['bs']['bs_grad']['max_evang']
    hinge.append({'rxn': rx, 'F_rks': f_rks, 'F_bs': f_bs})
    c = CH_HINGE[rx]
    if abs(round(f_rks, 3) - c[0]) > 5e-4 or abs(round(f_bs, 3) - c[1]) > 5e-4:
        hmis.append('%s roh %.3f/%.3f gegen Kapitel %.3f/%.3f'
                    % (rx, f_rks, f_bs, c[0], c[1]))
med_bs = float(np.median([h['F_bs'] for h in hinge]))
check(len(hinge) == 19, 'hinge n = %d (erwartet 19)' % len(hinge))
check(abs(med_bs - 1.697) < 5e-4, 'Median F_bs = %.4f (erwartet 1.697)' % med_bs)
check(not hmis, 'Kapitelwerte Abschnitt 6 gegen Rohdaten: %s'
      % ('alle 19 identisch' if not hmis else '; '.join(hmis)))

# ------------------------------------------------------------------ Ausgabe
os.makedirs(OUT, exist_ok=True)
rows.sort(key=lambda r: (r['rxn'], r['model']))
with open(f'{OUT}/paper_rows.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'model', 'F_model', 'F_dft', 'unstable',
                'lambda_min', 'nfod', 'breaking_depth'])
    for r in rows:
        w.writerow([r['rxn'], r['model'],
                    '' if r['F_model'] is None else '%.6f' % r['F_model'],
                    '%.6f' % r['F_dft'], r['unstable'],
                    '%.8f' % r['lambda_min'],
                    '' if r['nfod'] is None else '%.6f' % r['nfod'],
                    '' if r['breaking_depth'] is None else '%.3f' % r['breaking_depth']])
with open(f'{OUT}/hinge_rows.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'F_rks', 'F_bs'])
    for h in hinge:
        w.writerow([h['rxn'], '%.6f' % h['F_rks'], '%.6f' % h['F_bs']])

print('VALIDIERUNG')
for line in notes + fails:
    print('  ' + line)
print()
print('  Brechungstiefe: %d Zeilen ohne Wert%s'
      % (sum(1 for r in rows if r['breaking_depth'] is None),
         ('' if not no_depth else '  (' + ', '.join(sorted(set(no_depth))) + ')')))
dep = [r['breaking_depth'] for r in rows if r['unstable'] and r['breaking_depth'] is not None]
if dep:
    print('  Median breaking_depth instabil: %.1f meV   Spanne %.1f bis %.1f'
          % (np.median(dep), min(dep), max(dep)))
print()
print('  geschrieben: results/paper_rows.csv (%d) und results/hinge_rows.csv (%d)'
      % (len(rows), len(hinge)))
if fails:
    sys.exit('\nABBRUCH: %d Pruefung(en) fehlgeschlagen.' % len(fails))
