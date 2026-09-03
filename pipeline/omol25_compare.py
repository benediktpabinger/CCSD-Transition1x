"""def2-TZVP gegen das OMol25-Niveau, dieselben Geometrien.

Die Modelle sind gegen wB97M-V/def2-TZVPD trainiert, gerechnet wurde bisher
def2-TZVP mit ORCA-Standardgitter und lockeren Integralschwellen. Ein Teil des
gemessenen Modellfehlers ist damit Basissatzdifferenz und nicht Modellfehler.
Hier stehen beide Niveaus an denselben, unveraenderten Strukturen nebeneinander.

    TZVP    orca_freq/<rxn>_<Modell>/         + orca_ep/<rxn>_<Modell>_{R,P}/
    TZVPD   orca_om25/<rxn>_<Modell>/         alle vier Groessen in einem Lauf

Verglichen werden vier Groessen:
    max|F| am Modell-TS      -> Stufe-1-Urteil
    MAE der Kraftvektoren    -> mittlerer Fehler je Komponente, Modell vs DFT
    Barriere  E(TS) - E(R)   -> Barrierenfehler bei eingefrorener Geometrie
    Reaktionsenergie E(P)-E(R)

results/omol25_compare.csv
"""
import csv
import glob
import os
import re

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
EVA = 51.42208
HA_EV = 27.211386
STAT = 0.15
SLUG = {'UMA-S': 'uma-s', 'UMA-M': 'uma-m', 'eSEN': 'esen'}
MODELDIR = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
            'eSEN': 'esen_neb_results'}
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')


def gradient(p):
    """dE/dx in eV/A, ganzer Vektor. Die Kraft ist minus davon."""
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    i = t.find('CARTESIAN GRADIENT')
    if i < 0:
        return None
    G = []
    for line in t[i:].split('\n')[3:]:
        f = line.split()
        if len(f) < 6:
            break
        try:
            G.append([float(v) for v in f[3:6]])
        except ValueError:
            break
    return np.array(G) * EVA if G else None


def model_forces(p):
    """Kraftvektoren des Modells aus der extxyz: die letzten drei Spalten."""
    if not os.path.exists(p):
        return None
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


def energy(p):
    if not os.path.exists(p):
        return None, None
    t = open(p, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return None, None
    e = E_RE.findall(t)
    if not e or float(e[-1]) == 0.0:
        return None, None
    s2 = S2_RE.findall(t)
    return float(e[-1]) * HA_EV, (float(s2[-1]) if s2 else None)


base = {(r['rxn'], r['model']): r
        for r in csv.DictReader(open(f'{OUT}/paper_rows_ext.csv'))}
bfz = {(r['rxn'], r['model']): r
       for r in csv.DictReader(open(f'{OUT}/barrier_frozen.csv'))}

rows = []
for d in sorted(glob.glob(f'{H}/orca_om25/rxn*')):
    rx, mm = os.path.basename(d).rsplit('_', 1)
    k = (rx, SLUG.get(mm, mm))
    if k not in base:
        continue
    G = gradient(f'{d}/ts_engrad.out')
    fn = None if G is None else float(np.abs(G).max())
    Fm = model_forces(f'{H}/{MODELDIR[mm]}/{rx}/transition_state.xyz'
                      if mm in MODELDIR else '')
    mae = mxe = None
    if G is not None and Fm is not None and len(Fm) == len(G):
        dF = Fm - (-G)                     # Modellkraft minus DFT-Kraft
        mae = float(np.abs(dF).mean())
        mxe = float(np.abs(dF).max())
    ts, s2ts = energy(f'{d}/ts_sp.out')
    r_, _ = energy(f'{d}/r_sp.out')
    p_, _ = energy(f'{d}/p_sp.out')
    b = base[k]
    o = bfz.get(k)
    rows.append({
        'rxn': rx, 'model': k[1], 'unstable': int(b['unstable']),
        'F_tzvp': float(b['F_dft']), 'F_tzvpd': fn,
        'F_model': float(b['F_model']),
        'mae_force': mae, 'maxcomp_err': mxe,
        'barr_tzvp': float(o['barr_dft']) if o else None,
        'barr_tzvpd': (ts - r_) if (ts is not None and r_ is not None) else None,
        'barr_model': float(o['barr_model']) if o else None,
        'rxne_tzvp': float(o['rxne_dft']) if o else None,
        'rxne_tzvpd': (p_ - r_) if (p_ is not None and r_ is not None) else None,
        'rxne_model': float(o['rxne_model']) if o else None,
        's2_ts_tzvpd': s2ts})

os.makedirs(OUT, exist_ok=True)
COLS = ['rxn', 'model', 'unstable', 'F_model', 'F_tzvp', 'F_tzvpd',
        'mae_force', 'maxcomp_err',
        'barr_model', 'barr_tzvp', 'barr_tzvpd',
        'rxne_model', 'rxne_tzvp', 'rxne_tzvpd', 's2_ts_tzvpd']
with open(f'{OUT}/omol25_compare.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in sorted(rows, key=lambda r: (r['rxn'], r['model'])):
        w.writerow([r[c] if c in ('rxn', 'model', 'unstable')
                    else ('' if r[c] is None else
                          ('%.6f' if c in ('mae_force', 'maxcomp_err')
                           else '%.4f') % r[c]) for c in COLS])

un = np.array([r['unstable'] == 1 for r in rows])
mdl = np.array([r['model'] for r in rows])


def arr(k):
    return np.array([np.nan if r[k] is None else r[k] for r in rows])


fo, fn, fm = arr('F_tzvp'), arr('F_tzvpd'), arr('F_model')
bo, bn, bm = arr('barr_tzvp'), arr('barr_tzvpd'), arr('barr_model')
ro, rn, rm = arr('rxne_tzvp'), arr('rxne_tzvpd'), arr('rxne_model')
okF = ~np.isnan(fn)
okB = ~np.isnan(bn) & ~np.isnan(bo)

print('def2-TZVP GEGEN OMol25-NIVEAU (def2-TZVPD, DEFGRID3, Thresh 1e-12)')
print('=' * 76)
print('%d Zeilen, davon %d mit Gradient, %d mit Barriere' % (len(rows), okF.sum(), okB.sum()))
print()
print('1  RESTKRAFT max|F| am Modell-TS   [eV/A]')
print('   %-12s %4s %10s %10s %10s' % ('', 'n', 'TZVP', 'TZVPD', 'Delta'))
for lab, sel in (('stabil', okF & ~un), ('instabil', okF & un), ('alle', okF)):
    print('   %-12s %4d %10.4f %10.4f %+10.4f'
          % (lab, sel.sum(), np.median(fo[sel]), np.median(fn[sel]),
             np.median(fn[sel]) - np.median(fo[sel])))
d = fn[okF] - fo[okF]
print('   je Zeile: Median %+.4f   |Delta| Median %.4f   max %.4f'
      % (np.median(d), np.median(np.abs(d)), np.abs(d).max()))
print()
print('   Trennung stabil/instabil')
print('      Modellkraft   %.4f  (niveau-unabhaengig)'
      % abs(np.median(fm[okF & ~un]) - np.median(fm[okF & un])))
print('      DFT  TZVP     %.4f' % abs(np.median(fo[okF & ~un]) - np.median(fo[okF & un])))
print('      DFT  TZVPD    %.4f' % abs(np.median(fn[okF & ~un]) - np.median(fn[okF & un])))
print()
print('   Stufe-1-Urteil (Schwelle %.2f)' % STAT)
print('      nicht stationaer  TZVP %d   TZVPD %d   von %d'
      % ((fo[okF] >= STAT).sum(), (fn[okF] >= STAT).sum(), okF.sum()))
flip = (fo[okF] >= STAT) != (fn[okF] >= STAT)
print('      Urteilswechsel    %d Zeilen' % flip.sum())
rxa = np.array([r['rxn'] for r in rows])
for i in np.flatnonzero(okF)[flip]:
    print('         %-9s %-6s  %.4f -> %.4f  %s'
          % (rxa[i], mdl[i], fo[i], fn[i],
             'wird gueltig' if fn[i] < STAT else 'wird Ausfall'))
print()
print('2  BARRIERENFEHLER bei eingefrorener Geometrie   [eV]')
print('   %-12s %4s %10s %10s' % ('', 'n', 'TZVP', 'TZVPD'))
for lab, sel in (('stabil', okB & ~un), ('instabil', okB & un), ('alle', okB)):
    print('   %-12s %4d %10.4f %10.4f'
          % (lab, sel.sum(), np.median(np.abs(bm[sel] - bo[sel])),
             np.median(np.abs(bm[sel] - bn[sel]))))
for m in ('uma-s', 'uma-m', 'esen'):
    sel = okB & (mdl == m)
    print('   %-12s %4d %10.4f %10.4f'
          % (m, sel.sum(), np.median(np.abs(bm[sel] - bo[sel])),
             np.median(np.abs(bm[sel] - bn[sel]))))
print('   DFT-Barriere verschiebt sich: Median %+.4f   |d| Median %.4f   max %.4f'
      % (np.median(bn[okB] - bo[okB]), np.median(np.abs(bn[okB] - bo[okB])),
         np.abs(bn[okB] - bo[okB]).max()))
print()
okR = ~np.isnan(rn) & ~np.isnan(ro)
print('3  REAKTIONSENERGIEFEHLER   [eV]')
print('   %-12s %4s %10s %10s' % ('', 'n', 'TZVP', 'TZVPD'))
for lab, sel in (('alle', okR),):
    print('   %-12s %4d %10.4f %10.4f'
          % (lab, sel.sum(), np.median(np.abs(rm[sel] - ro[sel])),
             np.median(np.abs(rm[sel] - rn[sel]))))
print()
me = arr('mae_force')
okM = ~np.isnan(me)
print('4  KRAFTFEHLER JE KOMPONENTE, MAE |F_Modell - F_DFT|   [eV/A]')
print('   %-16s %4s %10s %10s %10s' % ('', 'n', 'Median', 'p90', 'Max'))
for lab, sel in (('stabil', okM & ~un), ('instabil', okM & un), ('alle', okM)):
    print('   %-16s %4d %10.4f %10.4f %10.4f'
          % (lab, sel.sum(), np.median(me[sel]), np.percentile(me[sel], 90),
             me[sel].max()))
for m in ('uma-s', 'uma-m', 'esen'):
    for u, lb in ((False, 'stabil'), (True, 'instabil')):
        sel = okM & (mdl == m) & (un == u)
        print('   %-16s %4d %10.4f %10.4f %10.4f'
              % (m + ' / ' + lb, sel.sum(), np.median(me[sel]),
                 np.percentile(me[sel], 90), me[sel].max()))
print()
print('geschrieben: results/omol25_compare.csv (%d Zeilen)' % len(rows))
