"""Der Versionstest: ORCA 5.0.4 (unsere Einzelpunkte) gegen ORCA 6.0.0 (OMol25).

FRAGE
    docs/chapter_omol25.tex sagt, unsere Einzelpunkte laufen auf den
    OMol25-Einstellungen -- wB97M-V/def2-TZVPD, def2/J, RIJCOSX, TightSCF,
    DEFGRID3, Thresh 1e-12, TCut 1e-13 -- aber in ORCA 5.0.4, waehrend OMol25
    ORCA 6.0.0 benutzt hat. Der Versionsunterschied war nicht geprueft. Hier
    wird er geprueft.

WARUM GERADE DIESE 45 STRUKTUREN
    OMol25 hat Transition1x mit neu gerechnet (Levine et al. 2025, Abschnitt
    2.5.1: "The Transition-1X dataset is a database of reactive trajectories
    and was recomputed in the UKS formalism"). Die Kerngeometrien blieben
    unveraendert. Fuer unsere 45 Uebergangszustaende existiert damit an
    IDENTISCHER Geometrie eine 5.0.4-Energie von uns und ein 6.0.0-Label von
    OMol25. Das ist der einzige Punkt, an dem die beiden Versionen ohne Umweg
    vergleichbar sind.

DIE ZWEI GRUPPEN
    Der Vergleich zerfaellt in zwei Faelle, die verschiedene Fragen
    beantworten; results/hinge_t1x.csv trennt sie ueber <S^2>:

    geschlossenschalig  <S^2> = 0 in unserem uks_sp.out. Beide Flaechen fallen
        zusammen, es gibt nur eine Loesung. Was uebrig bleibt, ist der
        Versionsunterschied allein. Erwartung: weit unter 1 meV.

    symmetriegebrochen  <S^2> > 0. Hier gibt es zwei Loesungen, und die
        Differenz sagt, auf welcher das OMol25-Label sitzt. Ein Unterschied in
        der Groesse der Bruchtiefe E_RKS - E_BS heisst: OMol25 ist auf der
        restringierten Loesung geblieben. Ein kleiner Unterschied heisst:
        beide haben die gebrochene gefunden.

UNSERE SEITE          ~/orca_hinge_t1x/<rxn>/   (pipeline/hinge_t1x.py)
    rks_sp.out      RKS + EnGrad, ohne Stabilitaetsanalyse -> E_RKS, F_RKS
    uks_sp.out      UKS + STABPerform                      -> <S^2>
    uks_engrad.out  EnGrad auf den Orbitalen von uks_sp    -> E_BS, F_BS
    E_BS kommt aus dem EnGrad-Lauf, nicht aus uks_sp: die beiden unterscheiden
    sich in ORCA 5.0.4 um rund 2.4e-5 Ha (pipeline/hinge_t1x.py).

OMOL25-SEITE          ~/omol25_hits_<split>.jsonl, aus pipeline/omol25_scan_t1x.py
    Energie in eV, Kraefte in eV/A, dazu s_squared, unrestricted, spin, charge,
    source, data_id.

EINHEITEN
    ORCA gibt FINAL SINGLE POINT ENERGY in Ha und den CARTESIAN GRADIENT in
    Eh/Bohr. Kraft = -Gradient. 1 Eh/Bohr = 51.42208 eV/A, 1 Ha = 27.2113862 eV.

Bricht ab, wenn eine Pruefung fehlschlaegt. results/omol25_label_compare.csv
"""
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
EVA = 51.42208                 # Eh/Bohr -> eV/A
HA_EV = 27.211386245988        # Ha -> eV
S2_BREAK = 0.05                # Schwelle fuer "symmetriegebrochen"
RMSD_MATCH = 1e-3              # A, Zuordnung Geometrie <-> OMol25-Eintrag
DE_CLOSED_MAX = 5.0            # meV, Einheiten-Plausibilitaet

E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


def orca(path):
    """Energie (Ha), <S^2> und Kraefte (eV/A) aus einem ORCA-Lauf."""
    if not os.path.exists(path):
        return None
    t = open(path, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return None
    e = E_RE.findall(t)
    s2 = S2_RE.findall(t)
    F = None
    i = t.find('CARTESIAN GRADIENT')
    if i >= 0:
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
            F = -np.array(G) * EVA          # Gradient -> Kraft, Eh/Bohr -> eV/A
    return dict(e=float(e[-1]) if e else None,
                s2=float(s2[-1]) if s2 else None, F=F)


def kabsch_rmsd(A, B):
    """RMSD nach optimaler Verschiebung und Drehung."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    V, S, W = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(V @ W))
    R = V @ np.diag([1.0, 1.0, d]) @ W
    return float(np.sqrt((((A @ R) - B) ** 2).sum(1).mean()))


# ---------------------------------------------------------------- Eingaben
tpath = f'{H}/t1x_ts_45.json'
if not os.path.exists(tpath):
    sys.exit('ABBRUCH: %s fehlt' % tpath)
targets = json.load(open(tpath))

hitfiles = sorted(glob.glob(f'{H}/omol25_hits_*.jsonl'))
if not hitfiles:
    sys.exit('ABBRUCH: keine omol25_hits_*.jsonl (pipeline/omol25_scan_t1x.py)')
hits = []
for p in hitfiles:
    split = os.path.basename(p)[len('omol25_hits_'):-len('.jsonl')]
    for line in open(p):
        r = json.loads(line)
        r['split'] = split
        hits.append(r)

print('DER VERSIONSTEST   ORCA 5.0.4 (wir) gegen ORCA 6.0.0 (OMol25)')
print('=' * 100)
print('%d Label-Geometrien, %d OMol25-Eintraege mit passender Summenformel'
      % (len(targets), len(hits)))
print('   Splits: %s' % ', '.join(sorted({h['split'] for h in hits})))
print()

# ---------------------------------------------------------------- Zuordnung
# Zuerst nach dem Reaktionsnamen in source vorsortieren. Ohne das waeren es
# 45 mal alle Treffer, und die Ausrichtung nach Kabsch fuer jedes Paar.
RXN_RE = re.compile(r't1x_(rxn\d+)_')
byrxn = collections.defaultdict(list)
for h in hits:
    m = RXN_RE.search(h.get('source') or '')
    byrxn[m.group(1) if m else None].append(h)
print('   %d OMol25-Eintraege tragen einen unserer Reaktionsnamen, verteilt '
      'auf %d Reaktionen' % (sum(len(byrxn[r]) for r in targets if r in byrxn),
                             sum(1 for r in targets if r in byrxn)))
print()

best = {}
for rx, t in targets.items():
    Zt = np.array(t['numbers'])
    Rt = np.array(t['positions'])
    cand = byrxn.get(rx) or [h for h in hits if len(h['numbers']) == len(Zt)]
    how = 'name' if byrxn.get(rx) else 'formel'
    for h in cand:
        Zh = np.array(h['numbers'])
        if Zh.shape != Zt.shape or not (Zh == Zt).all():
            continue
        Rh = np.array(h['positions']).reshape(-1, 3)
        d = float(np.sqrt(((Rh - Rt) ** 2).sum(1).mean()))
        # Die Ausrichtung nur, wenn die Reihenfolge nicht ohnehin schon passt.
        a = d if d < RMSD_MATCH else kabsch_rmsd(Rh, Rt)
        cur = best.get(rx)
        if cur is None or min(d, a) < min(cur['rmsd_direct'], cur['rmsd_align']):
            best[rx] = dict(hit=h, rmsd_direct=d, rmsd_align=a, how=how,
                            n_cand=len(cand))

matched = {rx: b for rx, b in best.items()
           if min(b['rmsd_direct'], b['rmsd_align']) < RMSD_MATCH}
print('ZUORDNUNG')
print('-' * 100)
print('   %d von %d Reaktionen in OMol25 gefunden (RMSD < %g A)'
      % (len(matched), len(targets), RMSD_MATCH))
unmatched = sorted(set(targets) - set(matched))
if unmatched:
    print('   nicht gefunden: %s' % ', '.join(unmatched))
    for rx in unmatched:
        b = best.get(rx)
        if b is None:
            print('      %-9s kein Eintrag mit gleicher Elementfolge im Scan'
                  % rx)
        else:
            print('      %-9s bester Kandidat RMSD %.4f A (ausgerichtet %.4f)'
                  % (rx, b['rmsd_direct'], b['rmsd_align']))
print()
if not matched:
    sys.exit('ABBRUCH: keine einzige Zuordnung -- nichts zu vergleichen')

# ---------------------------------------------------------------- Vergleich
rows = []
missing = []
for rx in sorted(matched):
    b = matched[rx]
    h = b['hit']
    d = f'{H}/orca_hinge_t1x/{rx}'
    rks, uks, ueg = (orca(f'{d}/rks_sp.out'), orca(f'{d}/uks_sp.out'),
                     orca(f'{d}/uks_engrad.out'))
    if any(x is None for x in (rks, uks, ueg)):
        missing.append(rx)
        continue
    n = len(targets[rx]['numbers'])
    Fo = np.array(h['forces']).reshape(-1, 3)
    if (Fo.shape != (n, 3) or rks['F'] is None or ueg['F'] is None
            or rks['F'].shape != (n, 3) or ueg['F'].shape != (n, 3)):
        missing.append(rx + ' (Kraftform)')
        continue

    bs = abs(uks['s2']) > S2_BREAK
    e_rks = rks['e'] * HA_EV
    e_bs = ueg['e'] * HA_EV
    e_om = h['energy']
    rows.append(dict(
        rxn=rx, split=h['split'], data_id=h['data_id'], source=h['source'],
        match_by=b['how'], n_cand=b['n_cand'],
        rmsd_A=min(b['rmsd_direct'], b['rmsd_align']),
        rmsd_direct_A=b['rmsd_direct'], n_atoms=n,
        group='bs' if bs else 'closed',
        s2_ours=uks['s2'], s2_omol=h['s_squared'],
        omol_unrestricted=int(bool(h['unrestricted'])),
        omol_charge=h['charge'], omol_spin=h['spin'],
        e_rks_ha=rks['e'], e_bs_ha=ueg['e'], e_omol_ev=e_om,
        d_rks_mev=(e_rks - e_om) * 1000.0,
        d_bs_mev=(e_bs - e_om) * 1000.0,
        depth_mev=(rks['e'] - ueg['e']) * HA_EV * 1000.0,
        dF_rks_max=float(np.abs(rks['F'] - Fo).max()),
        dF_bs_max=float(np.abs(ueg['F'] - Fo).max()),
        fmax_rks=float(np.abs(rks['F']).max()),
        fmax_bs=float(np.abs(ueg['F']).max()),
        fmax_omol=float(np.abs(Fo).max())))

# Auf welcher Loesung sitzt das OMol25-Label? Die naeher liegende gewinnt --
# aber nur, wenn sie ueberhaupt nahe ist. Liegen beide weit weg, hat ORCA 6.0.0
# eine dritte Loesung gefunden, und das waere selbst ein Ergebnis.
for r in rows:
    if r['group'] == 'closed':
        r['omol_surface'] = 'einzige'
    elif min(abs(r['d_bs_mev']), abs(r['d_rks_mev'])) > DE_CLOSED_MAX:
        r['omol_surface'] = 'weder'
    else:
        r['omol_surface'] = ('gebrochen'
                             if abs(r['d_bs_mev']) < abs(r['d_rks_mev'])
                             else 'restringiert')

COLS = ['rxn', 'split', 'data_id', 'group', 'n_atoms', 'match_by', 'n_cand',
        'rmsd_A',
        'rmsd_direct_A', 's2_ours', 's2_omol', 'omol_unrestricted',
        'omol_charge', 'omol_spin', 'e_rks_ha', 'e_bs_ha', 'e_omol_ev',
        'd_rks_mev', 'd_bs_mev', 'depth_mev', 'dF_rks_max', 'dF_bs_max',
        'fmax_rks', 'fmax_bs', 'fmax_omol', 'omol_surface', 'source']
FMT = {'rmsd_A': '%.6f', 'rmsd_direct_A': '%.6f', 's2_ours': '%.6f',
       's2_omol': '%.6f', 'e_rks_ha': '%.9f', 'e_bs_ha': '%.9f',
       'e_omol_ev': '%.9f', 'd_rks_mev': '%.4f', 'd_bs_mev': '%.4f',
       'depth_mev': '%.2f', 'dF_rks_max': '%.6f', 'dF_bs_max': '%.6f',
       'fmax_rks': '%.6f', 'fmax_bs': '%.6f', 'fmax_omol': '%.6f'}
os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/omol25_label_compare.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in sorted(rows, key=lambda r: (r['group'], r['rxn'])):
        w.writerow([FMT[c] % r[c] if c in FMT else r[c] for c in COLS])

cl = [r for r in rows if r['group'] == 'closed']
bs = [r for r in rows if r['group'] == 'bs']


def stat(S, key):
    a = np.abs(np.array([r[key] for r in S]))
    return float(np.median(a)), float(a.max())


print('GESCHLOSSENSCHALIGE ZEILEN -- der reine Versionstest')
print('-' * 100)
if cl:
    m, x = stat(cl, 'd_rks_mev')
    print('   n = %d' % len(cl))
    print('   |dE|  Median %.4f meV   groesster %.4f meV' % (m, x))
    print('   groesste Kraftkomponente |F_5.0.4 - F_6.0.0|  %.6f eV/A'
          % max(r['dF_rks_max'] for r in cl))
    print()
    print('   %-9s %14s %14s %12s' % ('rxn', 'dE/meV', 'dF/eV A^-1', 'RMSD/A'))
    for r in sorted(cl, key=lambda r: -abs(r['d_rks_mev']))[:8]:
        print('   %-9s %14.4f %14.6f %12.2e'
              % (r['rxn'], r['d_rks_mev'], r['dF_rks_max'], r['rmsd_A']))
else:
    print('   keine')
print()

print('SYMMETRIEGEBROCHENE ZEILEN -- auf welcher Loesung sitzt das Label')
print('-' * 100)
if bs:
    nres = sum(r['omol_surface'] == 'restringiert' for r in bs)
    nbrk = sum(r['omol_surface'] == 'gebrochen' for r in bs)
    nnei = sum(r['omol_surface'] == 'weder' for r in bs)
    print('   n = %d   davon %d auf der restringierten, %d auf der gebrochenen'
          ' Loesung, %d auf keiner von beiden' % (len(bs), nres, nbrk, nnei))
    m, x = stat(bs, 'd_bs_mev')
    print('   gegen unsere gebrochene Loesung    |dE| Median %9.3f meV  max %9.1f meV'
          % (m, x))
    m, x = stat(bs, 'd_rks_mev')
    print('   gegen unsere restringierte Loesung |dE| Median %9.3f meV  max %9.1f meV'
          % (m, x))
    print('   groesste Kraftkomponente gegen die gebrochene Loesung %.6f eV/A'
          % max(r['dF_bs_max'] for r in bs))
    print()
    print('   %-9s %9s %9s %11s %11s %11s %8s %s'
          % ('rxn', '<S^2>we', '<S^2>om', 'dE_bs/meV', 'dE_rks/meV',
             'Tiefe/meV', 'unrestr', 'Label sitzt auf'))
    for r in sorted(bs, key=lambda r: -r['depth_mev']):
        print('   %-9s %9.4f %9.4f %11.2f %11.2f %11.2f %8d %s'
              % (r['rxn'], r['s2_ours'], r['s2_omol'], r['d_bs_mev'],
                 r['d_rks_mev'], r['depth_mev'], r['omol_unrestricted'],
                 r['omol_surface']))
else:
    print('   keine')
print()

print('WIDERSPRUCH ZWISCHEN OMOL25-METADATEN UND UNSERER EINSTUFUNG')
print('-' * 100)
contra = [r for r in rows
          if (abs(r['s2_omol']) > S2_BREAK) != (r['group'] == 'bs')]
if contra:
    for r in contra:
        print('   %-9s wir <S^2> %.4f (%s), OMol25 s_squared %.4f, '
              'unrestricted %d'
              % (r['rxn'], r['s2_ours'], r['group'], r['s2_omol'],
                 r['omol_unrestricted']))
else:
    print('   keiner')
print()

print('Pruefungen')
check(not missing, 'alle zugeordneten Reaktionen mit vollstaendigen Laeufen'
      + ('' if not missing else ': fehlt %s' % missing))
check(bool(rows), 'mindestens eine Zeile')
if cl:
    med = float(np.median(np.abs([r['d_rks_mev'] for r in cl])))
    check(med < DE_CLOSED_MAX,
          'Einheiten-Plausibilitaet: Median |dE| der geschlossenschaligen '
          'Zeilen %.4f meV < %g meV' % (med, DE_CLOSED_MAX))
check(all(r['rmsd_A'] < RMSD_MATCH for r in rows),
      'jede Zeile RMSD < %g A' % RMSD_MATCH)
check(all(r['omol_charge'] == 0 and r['omol_spin'] == 1 for r in rows),
      'OMol25 fuehrt jede Struktur als neutrales Singulett')
check(all(r['n_atoms'] == len(targets[r['rxn']]['numbers']) for r in rows),
      'Atomzahl stimmt in jeder Zeile')
print()
print('results/omol25_label_compare.csv   %d Zeilen' % len(rows))
if fails:
    sys.exit('ABBRUCH: %d Pruefung(en) fehlgeschlagen:\n  ' % len(fails)
             + '\n  '.join(fails))
