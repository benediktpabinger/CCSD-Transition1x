"""Erweitert die Paper-Datenbasis um die Spalten, die die vier Figuren brauchen.

Dieselbe Zeilenschleife wie paper_rows.py und predictor_reffree.py, damit die
122 Zeilen dieselben bleiben; zusaetzlich je Zeile der Kraftfehler gegen DFT
(Definition woertlich aus force_error_at_ts.py) und die max-Komponente davon.

results/paper_rows_ext.csv   122 Zeilen, paper_rows.csv + mae_force + maxcomp_err
results/control_rks.csv      die stabilen Reaktionen mit max|F| des RKS-TS,
                             Kontrollpanel zu Figur 4

Bricht ab, wenn die bekannten Kennzahlen nicht mehr herauskommen.
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

fails, notes = [], []


def check(ok, msg):
    (notes if ok else fails).append(('ok   ' if ok else 'FEHL ') + msg)


def read_extxyz_forces(p):
    """Kraftvektoren des Modells: die letzten drei Spalten."""
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
    return np.array(F) if len(F) == n else None


def orca_gradient(label):
    """dE/dx in eV/A. ORCA druckt den Gradienten; die Kraft ist minus davon."""
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
            return np.array(G) * EVA
    return None


nfod = {r['rxn']: r['nfod']
        for r in json.load(open(f'{H}/fod_ranking.json'))['results']}

rows, control = [], []
for p in sorted(glob.glob(f'{H}/stab_pipeline/rxn*/result.json')):
    rx = os.path.basename(os.path.dirname(p))
    try:
        g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    except Exception:
        continue
    if not g or g.get('ext_stable') is None or g.get('lmin_ext') is None:
        continue

    if g['ext_stable']:
        depth = 0.0
        control.append({'rxn': rx, 'nfod': nfod.get(rx),
                        'F_rks': (g.get('rks_grad') or {}).get('max_evang')})
    else:
        bs = g.get('bs') or {}
        depth = -float(bs['de_meV']) if bs.get('de_meV') is not None else None

    for m, dn in MODELDIR.items():
        xyz = f'{H}/{dn}/{rx}/transition_state.xyz'
        if not os.path.exists(xyz):
            continue
        G = orca_gradient(f'{rx}_{m}')
        if G is None:
            continue
        Fm = read_extxyz_forces(xyz)
        mae = maxerr = None
        if Fm is not None and len(Fm) == len(G):
            d = Fm - (-G)                     # Modellkraft minus DFT-Kraft
            mae = float(np.abs(d).mean())
            maxerr = float(np.abs(d).max())
        # Zweiter Satz Spalten: dieselben Groessen AN DER MODELLGEOMETRIE.
        # Das ist die Variable, mit der Abschnitt 5 argumentiert -- ein Wert je
        # Zeile statt einer je Reaktion.  stab_pipeline fuehrt dafuer einen
        # eigenen Eintrag je Modell.
        gm = {x['source']: x for x in
              json.load(open(p))['geometries']}.get(m) or {}
        if gm.get('ext_stable') is None:
            d_mod, l_mod, u_mod = None, None, None
        elif gm['ext_stable']:
            d_mod, l_mod, u_mod = 0.0, gm.get('lmin_ext'), 0
        else:
            bsm = gm.get('bs') or {}
            d_mod = (-float(bsm['de_meV'])
                     if bsm.get('de_meV') is not None else None)
            l_mod, u_mod = gm.get('lmin_ext'), 1
        rows.append({
            'rxn': rx, 'model': SLUG[m],
            'F_model': None if Fm is None else float(np.abs(Fm).max()),
            'F_dft': float(np.abs(G).max()),
            'mae_force': mae, 'maxcomp_err': maxerr,
            'unstable': 0 if g['ext_stable'] else 1,
            'lambda_min': g['lmin_ext'], 'nfod': nfod.get(rx),
            'breaking_depth': depth,
            'unstable_model': u_mod, 'lambda_min_model': l_mod,
            'breaking_depth_model': d_mod})

y = np.array([r['F_dft'] >= STAT for r in rows])
st = np.array([r['unstable'] == 0 for r in rows])

check(len(rows) == 122, 'n = %d (erwartet 122)' % len(rows))
check(int(y.sum()) == 29, 'nicht stationaer %d (erwartet 29)' % int(y.sum()))
check(int(st.sum()) == 78 and int((~st).sum()) == 44,
      'stabil %d / instabil %d (erwartet 78 / 44)' % (st.sum(), (~st).sum()))
check(sum(1 for r in rows if r['mae_force'] is None) == 0,
      'mae_force fuer alle Zeilen (%d fehlen)'
      % sum(1 for r in rows if r['mae_force'] is None))
check(len(control) == 26, 'Kontrollreaktionen %d (erwartet 26)' % len(control))
check(all(c['F_rks'] is not None for c in control), 'F_rks fuer alle Kontrollen')

# Gegenprobe gegen die Tabelle in Abschnitt 5, die an der MODELLGEOMETRIE binnt.
dm = np.array([np.nan if r['breaking_depth_model'] is None
               else r['breaking_depth_model'] for r in rows])
have = ~np.isnan(dm)
fdv = np.array([r['F_dft'] for r in rows])
check(int(have.sum()) == 121,
      'Zeilen mit Tiefe an der Modellgeometrie: %d (erwartet 121)' % have.sum())
for a, b, want, nm in ((-1, 0, 0.069, 'stabil'), (0, 50, 0.160, '1-50 meV'),
                       (50, 200, 0.163, '50-200 meV'), (200, 1e9, 0.141, '>200 meV')):
    m = have & ((dm == 0) if b == 0 else ((dm > a) & (dm <= b)))
    got = float(np.median(fdv[m]))
    check(abs(got - want) < 1e-3,
          'Median |F|_DFT  Tiefe@Modell %-10s = %.3f (erwartet %.3f, n=%d)'
          % (nm, got, want, m.sum()))

# Gegenprobe gegen die Zahlen aus A.8 (nach Nachrechnung der 52 Luecken)
med = {}
for lab, sel in (('stabil', st), ('instabil', ~st)):
    for col in ('mae_force', 'maxcomp_err', 'F_model', 'F_dft'):
        med[(col, lab)] = float(np.median([rows[i][col] for i in np.flatnonzero(sel)]))
for key, want in ((('mae_force', 'stabil'), 0.013), (('mae_force', 'instabil'), 0.031),
                  (('maxcomp_err', 'stabil'), 0.058), (('maxcomp_err', 'instabil'), 0.142),
                  (('F_model', 'stabil'), 0.032), (('F_model', 'instabil'), 0.032),
                  (('F_dft', 'stabil'), 0.067), (('F_dft', 'instabil'), 0.163)):
    check(abs(med[key] - want) < 1e-3,
          'Median %-11s %-8s = %.4f (A.8 sagt %.3f)' % (key[0], key[1], med[key], want))

os.makedirs(OUT, exist_ok=True)
rows.sort(key=lambda r: (r['rxn'], r['model']))
COLS = ['rxn', 'model', 'F_model', 'F_dft', 'mae_force', 'maxcomp_err',
        'unstable', 'lambda_min', 'nfod', 'breaking_depth',
        'unstable_model', 'lambda_min_model', 'breaking_depth_model']
FMT = {'F_model': '%.6f', 'F_dft': '%.6f', 'mae_force': '%.6f',
       'maxcomp_err': '%.6f', 'lambda_min': '%.8f', 'nfod': '%.6f',
       'breaking_depth': '%.3f', 'lambda_min_model': '%.8f',
       'breaking_depth_model': '%.3f'}
with open(f'{OUT}/paper_rows_ext.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in rows:
        w.writerow(['' if r[c] is None else (FMT[c] % r[c] if c in FMT else r[c])
                    for c in COLS])

control.sort(key=lambda c: -(c['nfod'] or 0))
with open(f'{OUT}/control_rks.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'nfod', 'F_rks'])
    for c in control:
        w.writerow([c['rxn'], '%.6f' % (c['nfod'] or 0), '%.6f' % c['F_rks']])

print('VALIDIERUNG')
for line in notes + fails:
    print('  ' + line)
print()
print('  geschrieben: results/paper_rows_ext.csv (%d) und results/control_rks.csv (%d)'
      % (len(rows), len(control)))
if fails:
    sys.exit('\nABBRUCH: %d Pruefung(en) fehlgeschlagen.' % len(fails))
