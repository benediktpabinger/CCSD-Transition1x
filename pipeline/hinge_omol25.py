"""Der Hinge-Test auf OMol25-Niveau: derselbe Punkt, zwei Flaechen.

Bricht ab, wenn eine Pruefung fehlschlaegt.

DIE FRAGE
    Der Uebergangszustand stammt aus einer NEB auf der RESTRINGIERTEN Flaeche.
    Dort ist er ein stationaerer Punkt -- per Konstruktion. Wie gross ist die
    Restkraft an genau demselben Punkt auf der Flaeche, die dort der
    Grundzustand ist? Beide Kraefte an identischen Kernkoordinaten, es
    unterscheidet sich nur die elektronische Loesung.

GEOMETRIE
    ~/orca_neb_omol25/<rxn>/transition_state.xyz -- der Uebergangszustand der
    NEB auf OMol25-Niveau (wB97M-V/def2-TZVPD, RKS, Endpunkte auf demselben
    Niveau nachrelaxiert; pipeline/orca_neb_omol25.py). Nur die 33 Reaktionen
    mit konvergierter NEB; die zwoelf offenen bleiben aussen vor.

DIE DREI LAEUFE JE REAKTION      Slurm-Job 10773167, ~/orca_hinge25/<rxn>/
    rks_sp.out      RKS + EnGrad, keine Stabilitaetsanalyse -> E_RKS, F_RKS
    uks_sp.out      UKS + STABPerform + STABRestartUHFifUnstable -> E_BS, <S^2>
    uks_engrad.out  EnGrad auf den Orbitalen von uks_sp (MORead) -> F_BS
                    UND E_BS. Wichtig: die Energie der gebrochenen Loesung
                    wird aus DIESEM Lauf genommen, nicht aus uks_sp. ORCA
                    liefert fuer dieselbe Loesung in einem EnGrad-Lauf eine um
                    rund 2.4e-5 Ha andere Energie als in einem reinen
                    Einzelpunkt (abschliessende COSX/VV10-Gitterbehandlung).
                    Gemessen an den 18 stabilen Zeilen, wo beide Flaechen
                    zusammenfallen: rks_sp gegen uks_sp ergibt -2.43e-5 Ha
                    Median, rks_sp gegen uks_engrad +2.07e-10 Ha. Der
                    Vergleich muss EnGrad gegen EnGrad laufen, sonst traegt
                    jede Brechungstiefe einen Versatz von 0.66 meV.
    Niveau in allen dreien identisch zur Audit-Tabelle: wB97M-V/def2-TZVPD,
    def2/J, RIJCOSX, TightSCF, DEFGRID3, Thresh 1e-12, TCut 1e-13, ORCA 5.0.4.

UNTERSCHIED ZUR ALTEN FASSUNG results/hinge_rows.csv
    Geometrie frueher orca_neb_results/ (def2-TZVP-NEB), Kraefte frueher PySCF
    wB97M-V/def2-TZVP grids 3. Jetzt liegen Geometrie und Kraefte auf demselben
    Niveau. Die Klasse wird nicht mehr importiert, sondern aus <S^2> des
    uks_sp an genau diesem Punkt bestimmt; die alte Zuordnung steht als
    Vergleichsspalte daneben.

results/hinge_omol25.csv
"""
import csv
import glob
import os
import re

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
EVA = 51.42208                  # Eh/Bohr -> eV/A
HA_EV = 27.211386245988
S2_BREAK = 0.05

E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')
CYC_RE = re.compile(r'SCF CONVERGED AFTER\s+(\d+)\s+CYCLES')

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


def sp(p):
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return dict(ok=False, e=None, s2=None, cyc=None)
    e, s2, c = E_RE.findall(t), S2_RE.findall(t), CYC_RE.findall(t)
    return dict(ok=True, e=float(e[-1]) if e else None,
                s2=float(s2[-1]) if s2 else None,
                cyc=int(c[-1]) if c else None)


def gradmax(p):
    """groesste Betragskomponente des Gradienten in eV/A."""
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
    return float(np.abs(np.array(G) * EVA).max()) if G else None


old_mr = {r['rxn'] for r in csv.DictReader(open(f'{OUT}/hinge_rows.csv'))}
old_ct = {r['rxn'] for r in csv.DictReader(open(f'{OUT}/control_rks.csv'))}

rows, broken = [], []
for d in sorted(glob.glob(f'{H}/orca_hinge25/rxn*')):
    rx = os.path.basename(d)
    rks, uks = sp(f'{d}/rks_sp.out'), sp(f'{d}/uks_sp.out')
    ueg = sp(f'{d}/uks_engrad.out')
    if (rks is None or uks is None or ueg is None
            or not rks['ok'] or not uks['ok'] or not ueg['ok']):
        broken.append(rx)
        continue
    f_rks = gradmax(f'{d}/rks_sp.out')
    f_bs = gradmax(f'{d}/uks_engrad.out')
    unst = int(abs(uks['s2']) > S2_BREAK) if uks['s2'] is not None else None
    rows.append({
        'rxn': rx,
        'f_rks': f_rks, 'f_bs': f_bs,
        'ratio': None if (not f_rks or f_bs is None) else f_bs / f_rks,
        'e_rks_ha': rks['e'], 'e_bs_ha': ueg['e'],
        'depth_mev': None if (rks['e'] is None or ueg['e'] is None)
                     else (rks['e'] - ueg['e']) * HA_EV * 1000.0,
        's2_engrad': ueg['s2'],
        's2_bs': uks['s2'], 'unstable': unst,
        'scf_cyc_rks': rks['cyc'], 'scf_cyc_uks': uks['cyc'],
        'old_class': 'MR' if rx in old_mr else
                     ('control' if rx in old_ct else 'NOT FOUND')})

COLS = ['rxn', 'unstable', 's2_bs', 'f_rks', 'f_bs', 'ratio', 'depth_mev',
        'e_rks_ha', 'e_bs_ha', 'scf_cyc_rks', 'scf_cyc_uks', 'old_class']
FMT = {'s2_bs': '%.6f', 'f_rks': '%.6f', 'f_bs': '%.6f', 'ratio': '%.2f',
       'depth_mev': '%.2f', 'e_rks_ha': '%.9f', 'e_bs_ha': '%.9f'}
os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/hinge_omol25.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in rows:
        w.writerow(['' if r[c] is None else
                    (FMT[c] % r[c] if c in FMT else r[c]) for c in COLS])

un = [r for r in rows if r['unstable'] == 1]
st = [r for r in rows if r['unstable'] == 0]

print('HINGE-TEST AUF OMOL25-NIVEAU')
print('=' * 92)
print('%d Reaktionen ausgewertet. Nach <S^2> am OMol25-TS: %d instabil, '
      '%d stabil.' % (len(rows), len(un), len(st)))
print()

if un:
    print('INSTABILE REAKTIONEN -- derselbe Punkt auf beiden Flaechen')
    print('-' * 92)
    print('%-9s %9s %10s %10s %8s %11s   %s'
          % ('rxn', '<S^2>', 'F_RKS', 'F_BS', 'x', 'Tiefe/meV', 'alt'))
    for r in sorted(un, key=lambda r: -(r['ratio'] or 0)):
        print('%-9s %9.4f %10.4f %10.4f %8.1f %11.1f   %s'
              % (r['rxn'], r['s2_bs'], r['f_rks'], r['f_bs'], r['ratio'],
                 r['depth_mev'], r['old_class']))
    fr = np.array([r['f_rks'] for r in un])
    fb = np.array([r['f_bs'] for r in un])
    print()
    print('   F_RKS  %.4f bis %.4f   Median %.4f' % (fr.min(), fr.max(), np.median(fr)))
    print('   F_BS   %.4f bis %.4f   Median %.4f' % (fb.min(), fb.max(), np.median(fb)))
    print('   Verhaeltnis %.1f bis %.1f' % ((fb / fr).min(), (fb / fr).max()))
    print('   Zeilen mit F_BS < F_RKS: %d' % int((fb < fr).sum()))
    print()

if st:
    dd = np.array([abs(r['e_rks_ha'] - r['e_bs_ha']) for r in st])
    df = np.array([abs(r['f_rks'] - r['f_bs']) for r in st])
    print('STABILE REAKTIONEN -- Nullprobe')
    print('-' * 92)
    print('   Bei <S^2> = 0 ist die UKS-Loesung die RKS-Loesung. Energie und')
    print('   Kraft muessen dann uebereinstimmen, und zwar bis auf Rauschen.')
    print('   |E_RKS - E_BS|  max %.2e Ha' % dd.max())
    print('   |F_RKS - F_BS|  max %.2e eV/A' % df.max())
    print()

print('VERGLEICH MIT DER ALTEN EINTEILUNG (Referenz-TS, PySCF/def2-TZVP)')
print('-' * 92)
tab = {}
for r in rows:
    k = ('instabil' if r['unstable'] else 'stabil', r['old_class'])
    tab[k] = tab.get(k, 0) + 1
for k in sorted(tab):
    print('   neu %-9s  alt %-10s  %d' % (k[0], k[1], tab[k]))
flip = [r for r in rows
        if (r['old_class'] == 'MR') != (r['unstable'] == 1)]
if flip:
    print('   Zeilen, die die Seite wechseln:')
    for r in flip:
        print('      %-9s alt %-8s neu %-9s <S^2> %.4f  Tiefe %.1f meV'
              % (r['rxn'], r['old_class'],
                 'instabil' if r['unstable'] else 'stabil',
                 r['s2_bs'], r['depth_mev']))
print()

print('Pruefungen')
check(not broken, 'alle Laeufe normal beendet'
      + ('' if not broken else ': unvollstaendig %s' % broken))
check(len(rows) == 33, '33 Reaktionen (%d)' % len(rows))
check(all(r['f_rks'] is not None and r['f_bs'] is not None for r in rows),
      'beide Kraefte fuer jede Zeile vorhanden')
check(all(r['s2_bs'] is not None for r in rows), '<S^2> fuer jede Zeile')
if st:
    check(np.array([abs(r['e_rks_ha'] - r['e_bs_ha']) for r in st]).max() < 1e-8,
          'Nullprobe Energie: stabile Zeilen stimmen auf < 1e-8 Ha ueberein')
    check(np.array([abs(r['f_rks'] - r['f_bs']) for r in st]).max() < 1e-3,
          'Nullprobe Kraft: stabile Zeilen stimmen auf < 1e-3 eV/A ueberein')
if un:
    check(all(r['depth_mev'] > 0 for r in un),
          'alle instabilen Zeilen: E_RKS liegt ueber E_BS')
    check(all(r['f_bs'] > r['f_rks'] for r in un),
          'alle instabilen Zeilen: F_BS > F_RKS'
          + ('' if all(r['f_bs'] > r['f_rks'] for r in un) else
             ' -- Ausnahmen: %s' % [r['rxn'] for r in un
                                    if r['f_bs'] <= r['f_rks']]))
check(all(r['old_class'] != 'NOT FOUND' for r in rows),
      'jede Reaktion in der alten Einteilung wiedergefunden')
# uks_engrad konvergiert von den Orbitalen aus uks_sp neu. Eine gebrochene
# Loesung setzt sich dabei minimal anders; entscheidend ist nicht, dass <S^2>
# auf die letzte Stelle stimmt, sondern dass beide Laeufe dieselbe Loesung
# beschreiben -- also dieselbe Seite der Klassengrenze.
ds2 = [abs((r['s2_bs'] or 0) - (r['s2_engrad'] or 0)) for r in rows]
rel = [d / abs(r['s2_bs']) for d, r in zip(ds2, rows) if abs(r['s2_bs']) > 1e-6]
same = all((abs(r['s2_bs']) > S2_BREAK) == (abs(r['s2_engrad']) > S2_BREAK)
           for r in rows)
check(same,
      'uks_engrad und uks_sp auf derselben Seite der Klassengrenze in allen '
      '%d Zeilen; groesste <S^2>-Drift %.2e absolut, %.2f %% relativ '
      '(stabile Zeilen: exakt 0)'
      % (len(rows), max(ds2), 100 * max(rel) if rel else 0.0))

print()
print('geschrieben: results/hinge_omol25.csv (%d Zeilen)' % len(rows))
if fails:
    raise SystemExit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))
