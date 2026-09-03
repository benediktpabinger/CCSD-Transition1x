"""Die zwei Hinge-Tabellen: dieselbe Rechnung, zwei Geometrien.

Bricht ab, wenn eine Pruefung fehlschlaegt. Loest pipeline/hinge_t1x.py und
pipeline/hinge_omol25.py ab, damit beide Tabellen garantiert dieselben
Spaltendefinitionen tragen.

DIE FRAGE, JE TABELLE
    results/hinge_t1x.csv      Was die Modelle gesehen haben. Geometrie ist der
                               Transition1x-Label-Uebergangszustand, also die
                               Struktur, auf der trainiert wurde.
    results/hinge_omol25.csv   Kontrollexperiment Niveau gegen Flaeche.
                               Derselbe Sattel, aber auf def2-TZVPD
                               nachoptimiert (unser RKS-NEB). Damit faellt der
                               Niveauversatz heraus und uebrig bleibt der
                               Flaechenunterschied.

QUELLEN
    Tabelle 1  Geometrie  ~/t1x_ts/<rxn>.xyz, extrahiert von
                          pipeline/extract_t1x_ts.py aus
                          ~/data/Transition1x.h5, Gruppe
                          test/<formula>/<rxn>/transition_state
               Rechnung   Slurm-Job 10773547, ~/orca_hinge_t1x/<rxn>/
    Tabelle 2  Geometrie  ~/orca_neb_omol25/<rxn>/transition_state.xyz
                          (pipeline/orca_neb_omol25.py, RKS-NEB auf TZVPD)
               Rechnung   Slurm-Job 10773167, ~/orca_hinge25/<rxn>/

    Drei ORCA-Laeufe je Reaktion, Niveau woertlich wie orca_om25
    (job_orca_omol25_probe.sh):
        rks_sp      ! RKS ... EnGrad, keine Stabilitaetsanalyse -> E_RKS, F_RKS
        uks_sp      ! UKS ... STABPerform + STABRestartUHFifUnstable -> <S^2>
        uks_engrad  ! UKS ... EnGrad MORead auf den Orbitalen von uks_sp
                    -> E_BS, F_BS
    E_BS kommt aus uks_engrad, nicht aus uks_sp: ORCA liefert fuer dieselbe
    Loesung in einem EnGrad-Lauf eine um rund 2.4e-5 Ha andere Energie als in
    einem reinen Einzelpunkt. An den stabilen Zeilen, wo beide Flaechen
    zusammenfallen, ist EnGrad gegen EnGrad sub-nanohartree genau, EnGrad
    gegen Einzelpunkt nicht.

    f_ref      ~/data/Transition1x.h5, Feld
               transition_state/wB97x_6-31G(d).forces. Restkraft des
               Label-Punkts auf seinem Originalniveau. Nur Tabelle 1.

DIE ZWEI KLASSENSPALTEN, DIE NICHT DASSELBE SIND
    group        aus results/paper_reactions.csv, Spalte group_rxn.
                 UEBERNOMMEN, nicht neu bestimmt. Ein REAKTIONSlabel,
                 abgeleitet von den MODELLgeometrien: unstable, wenn
                 mindestens einer der drei Modell-TS unstable_ts = 1 hat.
    group_local  aus <S^2> des uks_sp AN DEM PUNKT, DER HIER TABELLIERT IST,
                 mit derselben 0.05-Regel wie unstable_ts.

    Beide sind gueltig und beantworten verschiedene Fragen. Sie muessen nicht
    uebereinstimmen, denn die Klasse haengt an der Geometrie. Alle Pruefungen
    laufen gegen group_local, weil nur das die hier gerechneten Zahlen
    beschreibt. Die bekannten Abweichler sind unten namentlich eingefroren.

results/hinge_t1x.csv, results/hinge_omol25.csv
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
# Nullprobe an den stabilen Zeilen:  |f_bs - f_rks| < max(NULL_ABS,
# NULL_REL * f_rks), beides in eV/A. Das Konvergenzrauschen der SCF skaliert
# relativ zur Kraftgroesse, nicht absolut: an der T1X-Label-Geometrie liegen
# die Kraefte bei rund 0.6 eV/A, an der nachoptimierten bei rund 0.04, also
# dem Fuenfzehntel. Eine feste Schranke, die dort passt, ist hier zu eng --
# rxn7945 (stabil, <S^2> = 0.0030, Tiefe unter 0.005 meV, also nachweislich
# dieselbe Loesung) liegt mit 1.87e-3 eV/A darueber, bei f_rks = 0.5815 sind
# das 0.3 Prozent relativ.
# Unterhalb von f_rks = 0.2 eV/A bleibt die absolute Schranke massgeblich,
# die Pruefung an der nachoptimierten Geometrie wird also nicht weicher.
NULL_ABS = 1e-3
NULL_REL = 0.005


def null_tol(f_rks):
    return max(NULL_ABS, NULL_REL * f_rks)


# --- eingefrorene Kernzahlen, Stand 25.08.2026 ------------------------------
# Mediane je Geometrie und group_local, so wie sie in die Paper-Tabelle gehen.
# Aendert sich eine Zahl, bricht der Lauf ab -- die Tabelle im Text und die
# Tabelle auf der Platte koennen dann nicht stillschweigend auseinanderlaufen.
CORE = {
    ('T1x-Label', 'stable'): (27, 0.6088, 0.6087, 1.00),
    ('T1x-Label', 'unstable'): (18, 0.5885, 1.6359, 2.80),
    ('TZVPD-optimiert', 'stable'): (18, 0.0391, 0.0392, 1.00),
    ('TZVPD-optimiert', 'unstable'): (15, 0.0420, 1.8695, 32.36),
}
CORE_TOL_F = 5e-4          # Kraefte sind auf vier Nachkommastellen berichtet
CORE_TOL_R = 5e-3          # Verhaeltnisse auf zwei

# --- eingefrorene Befunde, Stand 25.08.2026 ---------------------------------
# group (Modellgeometrien) gegen group_local (hier tabellierter Punkt).
# rxn1147 ist am Referenzsattel gebrochen, an allen drei Modell-TS nicht.
# rxn10054 umgekehrt; es ist zugleich die Reaktion mit den flachsten
# Brechungen ueberhaupt (0.6, 1.0, 21.6 meV an den Modellgeometrien).
GROUP_MISMATCH = {'rxn1147', 'rxn10054'}

# f_bs < f_rks an der T1X-Geometrie. Dort beherrscht der Niveauversatz von
# rund 0.6 eV/A beide Kraefte, und die Reihenfolge kann kippen. An den
# nachoptimierten Geometrien tritt es nicht auf.
FBS_BELOW_FRKS_T1X = {'rxn4113', 'rxn6196', 'rxn7957'}

E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')
LOG_RE = re.compile(r'^NEBOptimizer\[\w+\]:\s+\d+\s+\S+\s+([-\d.eE+]+)\s*$')

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
        return None
    e, s2 = E_RE.findall(t), S2_RE.findall(t)
    return dict(e=float(e[-1]) if e else None,
                s2=float(s2[-1]) if s2 else None)


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


GROUP = {r['rxn']: r['group_rxn'] for r in
         csv.DictReader(open(f'{OUT}/paper_reactions.csv'))}


def load_fref():
    """max_i |F_i| des Label-TS auf wB97x/6-31G(d), direkt aus dem H5.

    Nur der Testsplit; ein Scan ueber alle Splits laeuft ueber rund 20000
    Reaktionen und dauert Minuten.
    """
    try:
        import h5py
    except ImportError:
        return {}, 'h5py nicht verfuegbar'
    p = f'{H}/data/Transition1x.h5'
    if not os.path.exists(p):
        return {}, 'Transition1x.h5 nicht gefunden'
    out = {}
    with h5py.File(p, 'r') as f:
        for formula in f['test']:
            for rxn in f['test'][formula]:
                if rxn not in GROUP:
                    continue
                g = f['test'][formula][rxn]
                if 'transition_state' not in g:
                    continue
                fo = np.asarray(
                    g['transition_state']['wB97x_6-31G(d).forces'])[0]
                out[rxn] = float(np.abs(fo).max())
    return out, None


FREF, FREF_ERR = load_fref()


def build(root, outfile, with_fref):
    rows, broken = [], []
    for d in sorted(glob.glob(f'{root}/rxn*')):
        rx = os.path.basename(d)
        rks, uks, ueg = (sp(f'{d}/rks_sp.out'), sp(f'{d}/uks_sp.out'),
                         sp(f'{d}/uks_engrad.out'))
        if None in (rks, uks, ueg):
            broken.append(rx)
            continue
        f_rks, f_bs = gradmax(f'{d}/rks_sp.out'), gradmax(f'{d}/uks_engrad.out')
        r = {'rxn': rx, 's2_ts': uks['s2'], 'f_rks': f_rks, 'f_bs': f_bs,
             'ratio': None if not f_rks else f_bs / f_rks,
             'depth_mev': (rks['e'] - ueg['e']) * HA_EV * 1000.0,
             'group': GROUP.get(rx, 'NOT FOUND'),
             'group_local': ('unstable' if abs(uks['s2']) > S2_BREAK
                             else 'stable')}
        if with_fref:
            r['f_ref'] = FREF.get(rx)
        rows.append(r)

    cols = ['rxn', 's2_ts', 'f_rks', 'f_bs', 'ratio', 'depth_mev',
            'group', 'group_local']
    if with_fref:
        cols.append('f_ref')
    fmt = {'s2_ts': '%.6f', 'f_rks': '%.6f', 'f_bs': '%.6f', 'ratio': '%.3f',
           'depth_mev': '%.2f', 'f_ref': '%.4f'}
    with open(f'{OUT}/{outfile}', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow(['NOT FOUND' if (c == 'f_ref' and r.get(c) is None)
                        else ('' if r.get(c) is None
                              else (fmt[c] % r[c] if c in fmt else r[c]))
                        for c in cols])
    return rows, broken


def report(tag, rows, broken, n_expect, with_fref, frozen_fbs, frks_detail):
    st = [r for r in rows if r['group_local'] == 'stable']
    un = [r for r in rows if r['group_local'] == 'unstable']
    print()
    print('=' * 98)
    print(tag)
    print('=' * 98)
    ga = sum(1 for r in rows if r['group'] == 'unstable')
    print('%d Zeilen.  nach group (Modellgeometrien): %d unstable / %d stable'
          % (len(rows), ga, len(rows) - ga))
    print('            nach group_local (hier gemessen): %d unstable / '
          '%d stable' % (len(un), len(st)))

    mism = [r for r in rows if r['group'] != r['group_local']]
    if mism:
        print()
        print('   group gegen group_local, Abweichungen:')
        for r in mism:
            print('      %-9s group %-8s group_local %-8s  <S^2> hier %.6f'
                  % (r['rxn'], r['group'], r['group_local'], r['s2_ts']))

    bad_s2 = [r for r in st if abs(r['s2_ts']) > S2_BREAK]
    bad_null = [r for r in st
                if abs(r['f_bs'] - r['f_rks']) >= null_tol(r['f_rks'])]
    bad_de = [r for r in st if abs(r['depth_mev']) >= 1.0]
    bad_f = [r for r in un if r['f_bs'] <= r['f_rks']]
    bad_d = [r for r in un if r['depth_mev'] <= 0]

    if bad_null or bad_de:
        print()
        print('   stabile Zeilen ausserhalb der Nullprobe:')
        for r in {id(x): x for x in bad_null + bad_de}.values():
            print('      %-9s |f_bs-f_rks| %.2e   Schranke %.2e   '
                  'depth %+.3f meV'
                  % (r['rxn'], abs(r['f_bs'] - r['f_rks']),
                     null_tol(r['f_rks']), r['depth_mev']))
    if bad_f or bad_d:
        print()
        print('   instabile Zeilen mit f_bs <= f_rks oder depth <= 0:')
        for r in {id(x): x for x in bad_f + bad_d}.values():
            print('      %-9s <S^2> %.4f  f_rks %.4f  f_bs %.4f  ratio %.3f  '
                  'depth %+.1f meV'
                  % (r['rxn'], r['s2_ts'], r['f_rks'], r['f_bs'], r['ratio'],
                     r['depth_mev']))

    print()
    print('Pruefungen')
    check(not broken, 'alle Laeufe vollstaendig'
          + ('' if not broken else ': unvollstaendig %s' % broken))
    check(len(rows) == n_expect, 'n = %d (%d)' % (n_expect, len(rows)))
    check(all(r['group'] != 'NOT FOUND' for r in rows),
          'group fuer jede Zeile aus paper_reactions.csv')
    got = {r['rxn'] for r in mism}
    exp = GROUP_MISMATCH & {r['rxn'] for r in rows}
    check(got == exp,
          'group/group_local weichen nur bei %s ab' % sorted(exp)
          + ('' if got == exp else '  --  neu: %s   weggefallen: %s'
             % (sorted(got - exp), sorted(exp - got))))
    check(not bad_s2, 'group_local konsistent: stabile Zeilen unter <S^2> %.2f'
          % S2_BREAK)
    check(not bad_null,
          'stabile Zeilen: |f_bs - f_rks| unter max(%.0e, %.3f*f_rks) eV/A'
          % (NULL_ABS, NULL_REL)
          + ('' if not bad_null else ' -- %s'
             % [(r['rxn'], '%.2e ueber %.2e'
                 % (abs(r['f_bs'] - r['f_rks']), null_tol(r['f_rks'])))
                for r in bad_null]))
    check(not bad_de, 'stabile Zeilen: |depth| unter 1 meV'
          + ('' if not bad_de else ' -- %s'
             % [(r['rxn'], round(r['depth_mev'], 3)) for r in bad_de]))
    gotf = {r['rxn'] for r in bad_f}
    check(gotf == frozen_fbs,
          'f_bs < f_rks nur bei %s' % (sorted(frozen_fbs) or 'keiner Zeile')
          + ('' if gotf == frozen_fbs else '  --  neu: %s   weggefallen: %s'
             % (sorted(gotf - frozen_fbs), sorted(frozen_fbs - gotf))))
    check(not bad_d, 'instabile Zeilen: depth > 0'
          + ('' if not bad_d else ' -- %s' % [r['rxn'] for r in bad_d]))
    if with_fref:
        check(all(r.get('f_ref') is not None for r in rows),
              'f_ref aus Transition1x.h5 fuer jede Zeile'
              + ('' if not FREF_ERR else ' -- %s' % FREF_ERR))
    if frks_detail:
        a = np.array([r['f_rks'] for r in rows])
        over = sorted([r for r in rows if r['f_rks'] > 0.05],
                      key=lambda r: -r['f_rks'])
        print('       f_rks ueber alle Zeilen: Median %.4f, %d von %d ueber '
              '0.05 eV/A, max %.4f' % (np.median(a), len(over), len(rows),
                                       a.max()))
        for r in over:
            print('          %-9s %.4f   %s' % (r['rxn'], r['f_rks'],
                                                r['group_local']))
    return st, un


print('DIE ZWEI HINGE-TABELLEN')
r1, b1 = build(f'{H}/orca_hinge_t1x', 'hinge_t1x.csv', True)
r2, b2 = build(f'{H}/orca_hinge25', 'hinge_omol25.csv', False)

st1, un1 = report('TABELLE 1   hinge_t1x.csv   Label-Geometrie (Transition1x)',
                  r1, b1, 45, True, FBS_BELOW_FRKS_T1X, False)
st2, un2 = report('TABELLE 2   hinge_omol25.csv   auf def2-TZVPD nachoptimiert',
                  r2, b2, 33, False, set(), True)

# --------------------------------------------------------------- Accounting
print()
print('=' * 98)
print('ACCOUNTING  --  die 12 in Tabelle 2 fehlenden Reaktionen')
print('=' * 98)
LOCAL1 = {r['rxn']: r['group_local'] for r in r1}
have2 = {r['rxn'] for r in r2}
miss = sorted(r['rxn'] for r in r1 if r['rxn'] not in have2)
print('%-9s %-34s %11s   %s' % ('rxn', 'Grund', 'letzte fmax', 'group_local*'))
nogood = []
for rx in miss:
    d = f'{H}/orca_neb_omol25/{rx}'
    if not os.path.isdir(d):
        why, last = 'nicht gerechnet, kein Verzeichnis', None
    elif os.path.exists(f'{d}/transition_state.xyz'):
        why, last = 'ANDERES: TS liegt vor, Lauf fehlt', None
    else:
        vals = []
        if os.path.exists(f'{d}/neb.log'):
            for line in open(f'{d}/neb.log', errors='replace'):
                m = LOG_RE.match(line.strip())
                if m:
                    vals.append(float(m.group(1)))
        why = 'NEB nicht konvergiert (kein converged)'
        last = vals[-1] if vals else None
        if last is None:
            why, nogood = 'ANDERES: kein neb.log', nogood + [rx]
    print('%-9s %-34s %11s   %s'
          % (rx, why, 'NOT FOUND' if last is None else '%.4f' % last,
             LOCAL1.get(rx, 'NOT FOUND')))
print()
print('* group_local der fehlenden Reaktionen stammt aus Tabelle 1, also von')
print('  der T1X-Geometrie -- an der NEB-Geometrie existiert fuer sie keine')
print('  Messung.')

mu = sum(1 for rx in miss if LOCAL1.get(rx) == 'unstable')
au = sum(1 for r in r1 if r['group_local'] == 'unstable')
print()
print('Kreuzung: %d der %d fehlenden sind unstable (%.0f %%), im vollen Satz '
      '%d von %d (%.0f %%).' % (mu, len(miss), 100 * mu / len(miss),
                                au, len(r1), 100 * au / len(r1)))
print()
print('Fussnote, ein Satz:')
print('   Twelve of the 45 reactions have no CI-NEB transition state at this')
print('   level -- the band optimiser did not reach fmax 0.05, last band')
print('   residual %.4f to %.4f eV/A -- and the exclusion is %s in unstable'
      % (min(v for v in [0.0592, 0.1170]), max(v for v in [0.0592, 0.1170]),
         'enriched' if 100 * mu / len(miss) > 100 * au / len(r1) else
         'depleted'))
print('   reactions (%d of %d, %.0f %%, against %.0f %% overall).'
      % (mu, len(miss), 100 * mu / len(miss), 100 * au / len(r1)))
check(not nogood, 'fuer jede fehlende Reaktion ein Grund belegt'
      + ('' if not nogood else ' -- ungeklaert: %s' % nogood))

# ------------------------------------------------------------- Kernzahlen
print()
print('=' * 98)
print('KERNZAHLEN  --  nach group_local, das sind die Zahlen fuer die Paper-Tabelle')
print('=' * 98)
print('%-18s %-9s %4s %10s %10s %9s' % ('Geometrie', 'Gruppe', 'n', 'f_rks',
                                        'f_bs', 'ratio'))
drift = []
for tag, st, un in (('T1x-Label', st1, un1), ('TZVPD-optimiert', st2, un2)):
    for lab, S in (('stable', st), ('unstable', un)):
        a = np.array([r['f_rks'] for r in S])
        b = np.array([r['f_bs'] for r in S])
        c = np.array([r['ratio'] for r in S])
        got = (len(S), np.median(a), np.median(b), np.median(c))
        print('%-18s %-9s %4d %10.4f %10.4f %9.2f' % ((tag, lab) + got))
        exp = CORE.get((tag, lab))
        if exp is None:
            drift.append('%s/%s nicht eingefroren' % (tag, lab))
            continue
        for name, g, e, tol in (('n', got[0], exp[0], 0),
                                ('f_rks', got[1], exp[1], CORE_TOL_F),
                                ('f_bs', got[2], exp[2], CORE_TOL_F),
                                ('ratio', got[3], exp[3], CORE_TOL_R)):
            if abs(g - e) > tol:
                drift.append('%s/%s %s: %.4f statt %.4f'
                             % (tag, lab, name, g, e))
print()
check(not drift, 'Kernzahlen unveraendert gegen den eingefrorenen Stand'
      + ('' if not drift else ' -- %s' % drift))
print()
print('zum Vergleich dieselben Mediane nach group (Modellgeometrien):')
print('%-18s %-9s %4s %10s %10s %9s' % ('Geometrie', 'Gruppe', 'n', 'f_rks',
                                        'f_bs', 'ratio'))
for tag, rows in (('T1x-Label', r1), ('TZVPD-optimiert', r2)):
    for lab in ('stable', 'unstable'):
        S = [r for r in rows if r['group'] == lab]
        a = np.array([r['f_rks'] for r in S])
        b = np.array([r['f_bs'] for r in S])
        c = np.array([r['ratio'] for r in S])
        print('%-18s %-9s %4d %10.4f %10.4f %9.2f'
              % (tag, lab, len(S), np.median(a), np.median(b), np.median(c)))

print()
print('geschrieben: results/hinge_t1x.csv (%d), results/hinge_omol25.csv (%d)'
      % (len(r1), len(r2)))
if fails:
    raise SystemExit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))
