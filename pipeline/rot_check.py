"""Findet das OMol25-Symmetriebruch-Protokoll an den Audit-Geometrien dieselbe
Loesung wie unsere Stabilitaetsanalyse?

Bricht ab, wenn eine Pruefung fehlschlaegt. Abweichungen sind Befunde und
werden einzeln mit beiden <S^2>, beiden Energien und beiden Zyklenzahlen
ausgewiesen, nicht stillschweigend geduldet.

DIE ZWEI LAEUFE, DIE HIER VERGLICHEN WERDEN
    orca_om25/<rxn>_<Modell>/ts_sp.out     unser Weg: UKS + STABPerform +
                                           STABRestartUHFifUnstable. Quelle der
                                           Master-Tabelle. Wird nur GELESEN.
    orca_rot_check/<rxn>_<Modell>/ts_rot.out
                                           der OMol25-Weg: UKS mit
                                           Rotate {HOMO, LUMO, 20, 1, 1},
                                           keine Stabilitaetsanalyse.
                                           Slurm-Job 10771382.

Identische Geometrie (dieselbe transition_state.xyz), identische Stufe
(wB97M-V/def2-TZVPD, def2/J, RIJCOSX, TightSCF, DEFGRID3, Thresh 1e-12,
TCut 1e-13, MaxIter 300, ORCA 5.0.4). Einziger Unterschied ist der Weg zur
gebrochenen Loesung.

VORGESCHICHTE
    In omol25_settings.sh wurden beide Wege an 26 Reaktionen an den
    REFERENZ-Uebergangszustaenden verglichen: 26 von 26 identisch, Energien auf
    ~1e-8 Ha, dieselbe 18/8-Aufteilung wie die PySCF-Stabilitaetsanalyse. Die
    Master-Tabelle steht aber an den MODELLgeometrien. Dieser Lauf prueft
    genau dort.

results/rotation_check.csv
"""
import csv
import os
import re

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
S2_BREAK = 0.05
DE_TOL = 1e-6                 # Ha, geforderte Uebereinstimmung der Energien

# --- eingefrorener Befund, Stand 26.08.2026 ---------------------------------
# Die eine Zeile von 135, in der die 20-Grad-Rotation und STABPerform
# verschiedene Loesungen finden: die Rotation bleibt geschlossenschalig,
# STABPerform bricht und landet 2.623e-4 Ha = 7.14 meV TIEFER. Das ist das
# Ergebnis des Vergleichs, kein Fehler im Lauf -- die Pruefung schlaegt an,
# sobald sich die Menge aendert, nicht solange sie steht.
ROT_MISMATCH = {('rxn4113', 'uma-s')}
LONG = {'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'esen': 'eSEN'}
HARD = [('rxn0894', 'uma-m'), ('rxn8885', 'uma-s'), ('rxn8837', 'uma-s')]

E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')
CYC_RE = re.compile(r'SCF CONVERGED AFTER\s+(\d+)\s+CYCLES')
NEL_RE = re.compile(r'Number of Electrons\s+NEL\s*\.*\s*(\d+)')

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


def read(p):
    """Energie [Ha], <S^2>, SCF-Zyklen, Elektronenzahl, normal beendet."""
    if not os.path.exists(p):
        return None
    t = open(p, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return dict(ok=False, e=None, s2=None, cyc=None, nel=None)
    e = E_RE.findall(t)
    s2 = S2_RE.findall(t)
    c = CYC_RE.findall(t)
    n = NEL_RE.findall(t)
    return dict(ok=True,
                e=float(e[-1]) if e else None,
                s2=float(s2[-1]) if s2 else None,
                cyc=int(c[-1]) if c else None,
                nel=int(n[-1]) if n else None)


geo = list(csv.DictReader(open(f'{OUT}/omol25_model_geoms.csv')))
rows, missing, notterm = [], [], []
for g in geo:
    rx, md = g['rxn'], g['model']
    rot = read(f'{H}/orca_rot_check/{rx}_{LONG[md]}/ts_rot.out')
    stab = read(f'{H}/orca_om25/{rx}_{LONG[md]}/ts_sp.out')
    if rot is None:
        missing.append('%s/%s' % (rx, md))
        rot = dict(ok=False, e=None, s2=None, cyc=None, nel=None)
    elif not rot['ok']:
        notterm.append('%s/%s' % (rx, md))
    r = {'rxn': rx, 'model': md,
         's2_rotation': rot['s2'], 's2_stabperform': stab['s2'],
         'dE': (None if (rot['e'] is None or stab['e'] is None)
                else rot['e'] - stab['e']),
         'scf_cycles_rot': rot['cyc'], 'scf_cycles_stab': stab['cyc'],
         'e_rot': rot['e'], 'e_stab': stab['e'],
         'nel': rot['nel'], 'unstable_ts': int(g['unstable_ts'])}
    a = None if r['s2_rotation'] is None else abs(r['s2_rotation']) > S2_BREAK
    b = None if r['s2_stabperform'] is None else abs(r['s2_stabperform']) > S2_BREAK
    r['verdict_match'] = None if (a is None or b is None) else int(a == b)
    rows.append(r)

COLS = ['rxn', 'model', 's2_rotation', 's2_stabperform', 'dE',
        'scf_cycles_rot', 'scf_cycles_stab', 'verdict_match']
os.makedirs(OUT, exist_ok=True)
rows.sort(key=lambda r: (r['rxn'], r['model']))
with open(f'{OUT}/rotation_check.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in rows:
        w.writerow([
            r['rxn'], r['model'],
            '' if r['s2_rotation'] is None else '%.6f' % r['s2_rotation'],
            '' if r['s2_stabperform'] is None else '%.6f' % r['s2_stabperform'],
            '' if r['dE'] is None else '%.3e' % r['dE'],
            '' if r['scf_cycles_rot'] is None else r['scf_cycles_rot'],
            '' if r['scf_cycles_stab'] is None else r['scf_cycles_stab'],
            '' if r['verdict_match'] is None else r['verdict_match']])

st = [r for r in rows if r['unstable_ts'] == 0]
un = [r for r in rows if r['unstable_ts'] == 1]
done = [r for r in rows if r['s2_rotation'] is not None]

print('OMOL25-ROTATION GEGEN STABILITAETSANALYSE, AN DEN AUDIT-GEOMETRIEN')
print('=' * 96)
print('%d Zeilen; %d Rotationslaeufe ausgewertet. Stabil %d, instabil %d '
      '(Klasse aus omol25_model_geoms.csv).'
      % (len(rows), len(done), len(st), len(un)))
print()


def show(r, why):
    print('   %-9s %-6s %-22s' % (r['rxn'], r['model'], why))
    print('        <S^2>   Rotation %-12s  STABPerform %-12s'
          % (r['s2_rotation'], r['s2_stabperform']))
    print('        E [Ha]  Rotation %-16s STABPerform %-16s  dE %s'
          % (r['e_rot'], r['e_stab'],
             'n/a' if r['dE'] is None else '%.3e' % r['dE']))
    print('        Zyklen  Rotation %-6s       STABPerform %s'
          % (r['scf_cycles_rot'], r['scf_cycles_stab']))


# ------------------------------------------------------------- Abweichungen
bad_st = [r for r in st if r['s2_rotation'] is None
          or abs(r['s2_rotation']) > S2_BREAK]
bad_un = [r for r in un if r['verdict_match'] != 1
          or r['dE'] is None or abs(r['dE']) > DE_TOL]
if bad_st or bad_un:
    print('ABWEICHUNGEN, einzeln')
    print('-' * 96)
    for r in bad_st:
        show(r, 'stabil, aber Rotation bricht')
    for r in bad_un:
        why = []
        if r['verdict_match'] != 1:
            why.append('Urteile uneinig')
        if r['dE'] is not None and abs(r['dE']) > DE_TOL:
            why.append('|dE| > %.0e Ha' % DE_TOL)
        if r['dE'] is None:
            why.append('Energie fehlt')
        show(r, ', '.join(why))
    print()

# ------------------------------------------------------------------ Kennzahlen
if done:
    d = np.array([abs(r['dE']) for r in rows if r['dE'] is not None])
    s2s = np.array([abs(r['s2_rotation']) for r in st
                    if r['s2_rotation'] is not None])
    print('Kennzahlen')
    print('   |dE| ueber alle mit beiden Energien: Median %.2e  max %.2e Ha'
          % (np.median(d), d.max()))
    du = np.array([abs(r['dE']) for r in un if r['dE'] is not None])
    if len(du):
        print('   |dE| nur die instabilen Zeilen:      Median %.2e  max %.2e Ha'
              % (np.median(du), du.max()))
    if len(s2s):
        print('   groesstes <S^2> unter den stabilen Zeilen (Rotation): %.6f'
              % s2s.max())
    cr = np.array([r['scf_cycles_rot'] for r in rows
                   if r['scf_cycles_rot'] is not None])
    cs = np.array([r['scf_cycles_stab'] for r in rows
                   if r['scf_cycles_stab'] is not None])
    print('   SCF-Zyklen  Rotation Median %d max %d   STABPerform Median %d '
          'max %d' % (np.median(cr), cr.max(), np.median(cs), cs.max()))
    print('   (mehr Zyklen mit verdrehtem Startbild sind erwartbar und kein '
          'Befund, solange <S^2> am Ende stimmt)')
    print()
    print('Haertefaelle, gesondert')
    for k in HARD:
        r = [x for x in rows if (x['rxn'], x['model']) == k]
        if r:
            show(r[0], 'Haertefall')
    print()

# ------------------------------------------------------------- Pruefungen
print('Pruefungen')
check(not missing, 'ts_rot.out fuer alle Zeilen vorhanden'
      + ('' if not missing else ': fehlt bei %s' % missing))
check(not notterm, 'alle 135 Laeufe normal beendet'
      + ('' if not notterm else ': nicht beendet %s' % notterm))
check(len(st) == 82 and len(un) == 53,
      'Klassenaufteilung 82 stabil / 53 instabil (%d/%d)' % (len(st), len(un)))
check(not bad_st,
      'alle %d stabilen Zeilen: Rotation faellt auf <S^2> ~ 0 zurueck' % len(st)
      + ('' if not bad_st else ' -- Ausnahmen: %s'
         % [(r['rxn'], r['model'], r['s2_rotation']) for r in bad_st]))
got = {(r['rxn'], r['model']) for r in bad_un}
check(got == ROT_MISMATCH,
      'von %d instabilen Zeilen weichen genau %s ab (gleiches Urteil und '
      '|dE| <= %.0e Ha sonst ueberall)'
      % (len(un), sorted(ROT_MISMATCH) or 'keine', DE_TOL)
      + ('' if got == ROT_MISMATCH else '  --  neu: %s   weggefallen: %s'
         % (sorted(got - ROT_MISMATCH), sorted(ROT_MISMATCH - got))))
# Gegenrichtung: nirgends liegt die Rotation TIEFER als STABPerform. dE ist
# E_Rotation - E_STABPerform, ein negativer Wert waere also eine Loesung, die
# der Stabilitaetsanalyse entgangen ist. Belegt den Einordnungssatz in
# docs/methods_for_paper.md, Paragraph 4.
low = [r for r in rows if r['dE'] is not None and r['dE'] < -DE_TOL]
check(not low, 'keine Zeile mit Rotation unter STABPerform (dE < -%.0e Ha)'
      % DE_TOL
      + ('' if not low else ' -- %s' % [(r['rxn'], r['model'], r['dE'])
                                        for r in low]))
nb = [r for r in rows if r['nel'] is not None
      and r['nel'] % 2 != 0]
check(not nb, 'Elektronenzahl in allen Outputs gerade')

print()
print('geschrieben: results/rotation_check.csv (%d Zeilen)' % len(rows))
if fails:
    raise SystemExit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))
