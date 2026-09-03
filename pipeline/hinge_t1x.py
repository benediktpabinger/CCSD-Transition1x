"""Der Hinge-Test an den LABEL-Geometrien: derselbe Punkt, zwei Flaechen.

Bricht ab, wenn eine Pruefung fehlschlaegt.

GEOMETRIE
    ~/t1x_ts/<rxn>.xyz -- der Uebergangszustand aus Transition1x selbst,
    Gruppe 'transition_state', Niveau wB97x/6-31G(d). Das ist die Struktur,
    auf der die Modelle trainiert sind. Kein eigener NEB, keine
    Nachoptimierung. Extrahiert von pipeline/extract_t1x_ts.py; die
    Kommentarzeile jeder xyz traegt Energie und Restkraft des Datensatzes auf
    seinem eigenen Niveau.

DIE DREI LAEUFE JE REAKTION    Slurm-Job 10773547, ~/orca_hinge_t1x/<rxn>/
    rks_sp.out      RKS + EnGrad, keine Stabilitaetsanalyse -> E_RKS, F_RKS
    uks_sp.out      UKS + STABPerform -> <S^2>, waehlt die Flaeche
    uks_engrad.out  EnGrad auf den Orbitalen von uks_sp (MORead) -> E_BS, F_BS
    Alle drei auf OMol25-Niveau: wB97M-V/def2-TZVPD, def2/J, RIJCOSX,
    TightSCF, DEFGRID3, Thresh 1e-12, TCut 1e-13, ORCA 5.0.4.
    E_BS aus dem EnGrad-Lauf, nicht aus uks_sp -- Begruendung in
    pipeline/hinge_omol25.py und docs/methods_hinge.md.

WAS HIER ANDERS IST ALS AN DEN NEB-GEOMETRIEN
    Der Label-TS ist ein stationaerer Punkt auf wB97x/6-31G(d), nicht auf
    wB97M-V/def2-TZVPD. F_RKS traegt hier also zwei Anteile: den Niveauwechsel
    und, bei den instabilen Zeilen, nichts weiter -- denn der Flaechenwechsel
    steckt allein in F_BS - F_RKS. Die STABILEN Zeilen messen den
    Niveauwechsel-Anteil direkt, weil dort beide Flaechen zusammenfallen. Sie
    sind damit die Bezugsgroesse und keine blosse Kontrolle.

    Zusaetzlich steht mit f_ref (aus dem H5) die Restkraft am selben Punkt auf
    dem EIGENEN Niveau des Datensatzes daneben. Der Weg
        f_ref  ->  f_rks  ->  f_bs
    trennt: wie gut ist der Label-Sattel bei sich selbst, wieviel kostet der
    Niveauwechsel, wieviel kostet der Flaechenwechsel.

results/hinge_t1x.csv
"""
import csv
import glob
import os
import re

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
EVA = 51.42208
HA_EV = 27.211386245988
S2_BREAK = 0.05

E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')
CYC_RE = re.compile(r'SCF CONVERGED AFTER\s+(\d+)\s+CYCLES')
REF_RE = re.compile(r'maxcomp_ref_eV_A=([\d.]+)')

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


def refforce(rx):
    p = f'{H}/t1x_ts/{rx}.xyz'
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        fh.readline()
        m = REF_RE.search(fh.readline())
    return float(m.group(1)) if m else None


rows, broken = [], []
for d in sorted(glob.glob(f'{H}/orca_hinge_t1x/rxn*')):
    rx = os.path.basename(d)
    rks, uks, ueg = (sp(f'{d}/rks_sp.out'), sp(f'{d}/uks_sp.out'),
                     sp(f'{d}/uks_engrad.out'))
    if any(x is None or not x['ok'] for x in (rks, uks, ueg)):
        broken.append(rx)
        continue
    f_rks, f_bs = gradmax(f'{d}/rks_sp.out'), gradmax(f'{d}/uks_engrad.out')
    rows.append({
        'rxn': rx,
        'unstable': int(abs(uks['s2']) > S2_BREAK),
        's2_bs': uks['s2'], 's2_engrad': ueg['s2'],
        'f_ref': refforce(rx), 'f_rks': f_rks, 'f_bs': f_bs,
        'ratio': None if not f_rks else f_bs / f_rks,
        'depth_mev': (rks['e'] - ueg['e']) * HA_EV * 1000.0,
        'e_rks_ha': rks['e'], 'e_bs_ha': ueg['e'],
        'scf_cyc_rks': rks['cyc'], 'scf_cyc_uks': uks['cyc']})

COLS = ['rxn', 'unstable', 's2_bs', 'f_ref', 'f_rks', 'f_bs', 'ratio',
        'depth_mev', 'e_rks_ha', 'e_bs_ha', 'scf_cyc_rks', 'scf_cyc_uks']
FMT = {'s2_bs': '%.6f', 'f_ref': '%.4f', 'f_rks': '%.6f', 'f_bs': '%.6f',
       'ratio': '%.2f', 'depth_mev': '%.2f', 'e_rks_ha': '%.9f',
       'e_bs_ha': '%.9f'}
os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/hinge_t1x.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in rows:
        w.writerow(['' if r[c] is None else
                    (FMT[c] % r[c] if c in FMT else r[c]) for c in COLS])

un = [r for r in rows if r['unstable'] == 1]
st = [r for r in rows if r['unstable'] == 0]

print('HINGE-TEST AN DEN TRANSITION1X-LABEL-GEOMETRIEN')
print('=' * 96)
print('%d Reaktionen. Nach <S^2> am Label-TS: %d instabil, %d stabil.'
      % (len(rows), len(un), len(st)))
print()

print('DER WEG f_ref -> f_rks -> f_bs   [eV/A, groesste Kartesische Komponente]')
print('-' * 96)
print('   f_ref  Restkraft am Label-TS auf wB97x/6-31G(d), aus dem Datensatz')
print('   f_rks  dieselbe Struktur, restringiert auf wB97M-V/def2-TZVPD')
print('   f_bs   dieselbe Struktur, Grundzustandsflaeche, gleiches Niveau')
print()
for lab, S in (('stabil', st), ('instabil', un)):
    if not S:
        continue
    a = np.array([r['f_ref'] for r in S])
    b = np.array([r['f_rks'] for r in S])
    c = np.array([r['f_bs'] for r in S])
    print('   %-9s n=%2d   f_ref %.4f   f_rks %.4f   f_bs %.4f   (Mediane)'
          % (lab, len(S), np.median(a), np.median(b), np.median(c)))
if st:
    b = np.array([r['f_rks'] for r in st])
    print()
    print('   Die stabilen Zeilen messen den Niveauwechsel allein: dort fallen')
    print('   beide Flaechen zusammen, f_rks = f_bs. Median %.4f, Spanne '
          '%.4f bis %.4f eV/A.' % (np.median(b), b.min(), b.max()))
print()

if un:
    print('INSTABILE REAKTIONEN')
    print('-' * 96)
    print('%-9s %9s %9s %9s %9s %8s %11s'
          % ('rxn', '<S^2>', 'f_ref', 'f_rks', 'f_bs', 'x', 'Tiefe/meV'))
    for r in sorted(un, key=lambda r: -(r['ratio'] or 0)):
        print('%-9s %9.4f %9.4f %9.4f %9.4f %8.1f %11.1f'
              % (r['rxn'], r['s2_bs'], r['f_ref'], r['f_rks'], r['f_bs'],
                 r['ratio'], r['depth_mev']))
    fr = np.array([r['f_rks'] for r in un])
    fb = np.array([r['f_bs'] for r in un])
    print()
    print('   f_rks %.4f bis %.4f   f_bs %.4f bis %.4f   Verhaeltnis %.1f bis %.1f'
          % (fr.min(), fr.max(), fb.min(), fb.max(),
             (fb / fr).min(), (fb / fr).max()))
    print('   Zeilen mit f_bs < f_rks: %d' % int((fb < fr).sum()))
    print()

if st:
    dd = np.array([abs(r['e_rks_ha'] - r['e_bs_ha']) for r in st])
    df = np.array([abs(r['f_rks'] - r['f_bs']) for r in st])
    print('NULLPROBE an den stabilen Zeilen')
    print('-' * 96)
    print('   |E_RKS - E_BS|  max %.2e Ha' % dd.max())
    print('   |F_RKS - F_BS|  max %.2e eV/A' % df.max())
    print()

fr_all = np.array([r['f_ref'] for r in rows if r['f_ref'] is not None])
worst = sorted(rows, key=lambda r: -(r['f_ref'] or 0))[:3]
print('LABEL-TS AUF DEM EIGENEN NIVEAU')
print('-' * 96)
print('   f_ref  min %.4f  Median %.4f  max %.4f eV/A'
      % (fr_all.min(), np.median(fr_all), fr_all.max()))
print('   die drei groessten:')
for r in worst:
    print('      %-9s f_ref %.4f   f_rks %.4f   %s'
          % (r['rxn'], r['f_ref'], r['f_rks'],
             'instabil' if r['unstable'] else 'stabil'))
print()

print('Pruefungen')
check(not broken, 'alle Laeufe normal beendet'
      + ('' if not broken else ': unvollstaendig %s' % broken))
check(len(rows) == 45, '45 Reaktionen (%d)' % len(rows))
check(all(r['f_ref'] is not None for r in rows),
      'Referenzkraft aus dem H5 fuer jede Zeile')
check(all(r['f_rks'] is not None and r['f_bs'] is not None for r in rows),
      'beide Kraefte fuer jede Zeile')
if st:
    check(np.array([abs(r['e_rks_ha'] - r['e_bs_ha']) for r in st]).max() < 1e-8,
          'Nullprobe Energie: stabile Zeilen auf < 1e-8 Ha')
    check(np.array([abs(r['f_rks'] - r['f_bs']) for r in st]).max() < 1e-3,
          'Nullprobe Kraft: stabile Zeilen auf < 1e-3 eV/A')
if un:
    check(all(r['depth_mev'] > 0 for r in un),
          'alle instabilen Zeilen: E_RKS liegt ueber E_BS')
    bad = [r['rxn'] for r in un if r['f_bs'] <= r['f_rks']]
    check(not bad, 'alle instabilen Zeilen: f_bs > f_rks'
          + ('' if not bad else ' -- Ausnahmen: %s' % bad))
same = all((abs(r['s2_bs']) > S2_BREAK) == (abs(r['s2_engrad']) > S2_BREAK)
           for r in rows)
check(same, 'uks_engrad und uks_sp auf derselben Seite der Klassengrenze')

print()
print('geschrieben: results/hinge_t1x.csv (%d Zeilen)' % len(rows))
if fails:
    raise SystemExit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))
