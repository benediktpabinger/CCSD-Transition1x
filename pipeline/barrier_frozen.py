"""Barrierenfehler bei eingefrorener Geometrie.

Das Modell meldet eine Barriere aus seinen eigenen Strukturen. Dieselbe
Differenz wird hier aus DFT an genau denselben, unveraenderten Strukturen
gebildet. Die Geometrie faellt heraus -- was uebrig bleibt, ist reiner
Energiefehler.

    dE_Modell = E_Modell(TS) - E_Modell(R)      extxyz, Feld energy=  [eV]
    dE_DFT    = E_DFT(TS)    - E_DFT(R)         bs_sp, Hartree -> eV
    Fehler    = dE_Modell - dE_DFT

Dieselbe Rechnung fuer die Reaktionsenergie E(P) - E(R).

Quellen der DFT-Energien:
    TS   orca_freq/<rxn>_<Modell>/bs_sp.out     lag aus der Stufe-1-Kette vor
    R,P  orca_ep/<rxn>_<Modell>_{R,P}/bs_sp.out Job 10765094

In allen Faellen hat STABPerform die Grundzustandsloesung gewaehlt; <S^2> wird
mitgeschrieben, damit nachpruefbar bleibt, welche Loesung es war.

results/barrier_frozen.csv
"""
import csv
import glob
import json
import os
import re

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
HA_EV = 27.211386
MD = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
      'eSEN': 'esen_neb_results'}
SLUG = {'UMA-S': 'uma-s', 'UMA-M': 'uma-m', 'eSEN': 'esen'}
E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')


def model_energy(p):
    """ASE schreibt die Energie in eV in die Kommentarzeile."""
    if not os.path.exists(p):
        return None
    m = re.search(r'energy=(-?[\d.eE+]+)', open(p, errors='replace').read(4000))
    return float(m.group(1)) if m else None


def dft_energy(d):
    """Energie in eV und <S^2> aus einem bs_sp-Lauf."""
    p = f'{d}/bs_sp.out'
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


rows, incomplete = [], []
for p in sorted(glob.glob(f'{H}/stab_pipeline/rxn*/result.json')):
    rx = os.path.basename(os.path.dirname(p))
    g = {x['source']: x for x in json.load(open(p))['geometries']}.get('RKS-ref')
    if not g or g.get('ext_stable') is None:
        continue
    for m, dn in MD.items():
        if not os.path.exists(f'{H}/{dn}/{rx}/transition_state.xyz'):
            continue
        em = {k: model_energy(f'{H}/{dn}/{rx}/{f}.xyz')
              for k, f in (('R', 'reactant'), ('TS', 'transition_state'),
                           ('P', 'product'))}
        ed, s2 = {}, {}
        ed['TS'], s2['TS'] = dft_energy(f'{H}/orca_freq/{rx}_{m}')
        for k in ('R', 'P'):
            ed[k], s2[k] = dft_energy(f'{H}/orca_ep/{rx}_{m}_{k}')
        if any(v is None for v in em.values()) or any(v is None for v in ed.values()):
            incomplete.append('%s:%s  Modell %s  DFT %s'
                              % (rx, m,
                                 [k for k, v in em.items() if v is None],
                                 [k for k, v in ed.items() if v is None]))
            continue
        rows.append({
            'rxn': rx, 'model': SLUG[m],
            'unstable': 0 if g['ext_stable'] else 1,
            'barr_model': em['TS'] - em['R'],
            'barr_dft': ed['TS'] - ed['R'],
            'rxne_model': em['P'] - em['R'],
            'rxne_dft': ed['P'] - ed['R'],
            's2_R': s2['R'], 's2_TS': s2['TS'], 's2_P': s2['P']})
for r in rows:
    r['err_barr'] = r['barr_model'] - r['barr_dft']
    r['err_rxne'] = r['rxne_model'] - r['rxne_dft']

os.makedirs(OUT, exist_ok=True)
COLS = ['rxn', 'model', 'unstable', 'barr_model', 'barr_dft', 'err_barr',
        'rxne_model', 'rxne_dft', 'err_rxne', 's2_R', 's2_TS', 's2_P']
with open(f'{OUT}/barrier_frozen.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in sorted(rows, key=lambda r: (r['rxn'], r['model'])):
        w.writerow([r[c] if c in ('rxn', 'model', 'unstable')
                    else ('' if r[c] is None else '%.4f' % r[c]) for c in COLS])

eb = np.array([r['err_barr'] for r in rows])
er = np.array([r['err_rxne'] for r in rows])
un = np.array([r['unstable'] == 1 for r in rows])
mdl = np.array([r['model'] for r in rows])

print('BARRIERENFEHLER BEI EINGEFRORENER GEOMETRIE   [eV]')
print('=' * 74)
print('%d Zeilen' % len(rows) + ('   unvollstaendig: %d' % len(incomplete)
                                 if incomplete else ''))
for s in incomplete[:8]:
    print('   ', s)
print()


def block(title, sel_pairs):
    print(title)
    print('  %-14s %4s %9s %9s %9s %9s' % ('', 'n', 'Median', 'MedAbs', 'Min', 'Max'))
    for nm, sel in sel_pairs:
        if not sel.sum():
            continue
        v = eb[sel]
        print('  %-14s %4d %9.3f %9.3f %9.3f %9.3f'
              % (nm, sel.sum(), np.median(v), np.median(np.abs(v)), v.min(), v.max()))
    print()


block('Barrierenfehler, nach Modell',
      [(m, mdl == m) for m in ('uma-s', 'uma-m', 'esen')]
      + [('alle', np.ones(len(rows), bool))])

block('Barrierenfehler, nach Modell x RKS-Stabilitaet',
      [('%s / %s' % (m, 'instabil' if u else 'stabil'), (mdl == m) & (un == u))
       for m in ('uma-s', 'uma-m', 'esen') for u in (False, True)]
      + [('alle / stabil', ~un), ('alle / instabil', un)])

print('Reaktionsenergiefehler, nach Modell')
print('  %-14s %4s %9s %9s %9s %9s' % ('', 'n', 'Median', 'MedAbs', 'Min', 'Max'))
for m in ('uma-s', 'uma-m', 'esen'):
    v = er[mdl == m]
    print('  %-14s %4d %9.3f %9.3f %9.3f %9.3f'
          % (m, len(v), np.median(v), np.median(np.abs(v)), v.min(), v.max()))
v = er
print('  %-14s %4d %9.3f %9.3f %9.3f %9.3f'
      % ('alle', len(v), np.median(v), np.median(np.abs(v)), v.min(), v.max()))
print()
print('Die fuenf groessten Barrierenfehler')
for i in np.argsort(-np.abs(eb))[:5]:
    r = rows[i]
    print('   %-9s %-6s  Modell %+7.3f   DFT %+7.3f   Fehler %+7.3f   '
          '<S^2>(TS) %.3f'
          % (r['rxn'], r['model'], r['barr_model'], r['barr_dft'],
             r['err_barr'], r['s2_TS'] or 0))
print()
print('geschrieben: results/barrier_frozen.csv (%d Zeilen)' % len(rows))
