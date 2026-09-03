"""Auf welcher Flaeche sitzt das MLIP bei den drei Energie-Ausreissern?

Frage: liegt die MLIP-Barriere naeher an der restringierten Loesung (RKS) oder
an der gebrochenen (BS), beides am identischen Modell-TS gemessen?

Datenlage, alles am selben Punkt
    E_MLIP(TS), E_MLIP(R)   extxyz der Modell-NEBs, Feld energy=        [eV]
    E_BS(TS), E(R)          orca_om25/<rxn>_<Modell>/{ts_sp,r_sp}.out   TZVPD
                            ts_sp ist UKS mit STABPerform und dort auf die
                            gebrochene Loesung konvergiert, <S^2> ~ 1.0;
                            r_sp hat <S^2> = 0 und ist damit zugleich die
                            restringierte Loesung am Edukt
    E_RKS(TS)               orca_rks_sheet/<rxn>_<Modell>/ts_rks.out    TZVPD
                            eigens gerechnet, Job 10767516, Einstellungen
                            wortgleich zu ts_sp bis auf RKS statt UKS und ohne
                            Stabilitaetsanalyse
    Nebenrechnung           stab_pipeline (PySCF wB97M-V/def2-TZVP, grids 3)
                            liefert dieselben beiden Flaechen auf TZVP; sie
                            steht als Naeherung daneben, weil sie den Fall
                            rxn0894/UMA-M nur halb abdeckt

results/sheet_check_outliers.txt
"""
import csv
import io
import json
import os

RES = 'results'
MISS = 'fehlt'
HA_EV = 27.211386245988
LONG = {'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'esen': 'eSEN'}
SLUG = {v: k for k, v in LONG.items()}

ROWS = [('rxn0894', 'esen'), ('rxn0894', 'uma-m'), ('rxn8837', 'uma-s')]
CTRL = [('rxn7060', 'esen')]

# stab_pipeline, PySCF wB97M-V/def2-TZVP grids 3, an der Modellgeometrie.
# Woertlich aus stab_pipeline/<rxn>/result.json abgelesen. Hartree.
STAB = {
    ('rxn0894', 'esen'):  dict(e_rks=-322.2329902964, e_bs=-322.3796919426,
                               ext_stable=False),
    ('rxn0894', 'uma-m'): dict(e_rks=-322.2212308107, e_bs=None,
                               ext_stable=None),
    ('rxn8837', 'uma-s'): dict(e_rks=-322.9642715092, e_bs=-323.0919564513,
                               ext_stable=False),
    ('rxn7060', 'esen'):  dict(e_rks=-323.2539543344, e_bs=None,
                               ext_stable=True),
}
# Referenz-Edukt, ORCA orca_endpoint/<rxn>_reactant/rks.out. Hartree.
E0_REF = {'rxn0894': -322.542417281839, 'rxn8837': -323.423754074566,
          'rxn7060': -323.482272880759}

C = {(r['rxn'], r['model']): r
     for r in csv.DictReader(open(os.path.join(RES, 'omol25_compare.csv')))}

# E_RKS(TS) auf TZVPD, sofern der Lauf schon eingesammelt ist
RKSD = {}
p_rks = os.path.join(RES, 'rks_sheet_tzvpd.json')
if os.path.exists(p_rks):
    for k, v in json.load(open(p_rks)).items():
        rx, md = k.split('/')
        RKSD[(rx, SLUG.get(md, md))] = v      # {'barr': eV, 'cycles': int}


def num(r, k):
    return float(r[k]) if (r is not None and r[k] != '') else None


out = io.StringIO()
W = out.write

W('AUF WELCHEM SHEET SITZT DAS MLIP BEI DEN DREI ENERGIE-AUSREISSERN?\n')
W('=' * 112 + '\n')
W('Alle Barrieren in eV, vom Edukt zum vom Modell selbst vorhergesagten '
  'Uebergangszustand.\n')
W('RKS und BS unterscheiden sich nur am Uebergangszustand: das Edukt ist in '
  'allen vier Zeilen\n')
W('geschlossenschalig (<S^2> = 0 in r_sp.out), der Nullpunkt bevorzugt also '
  'keine der beiden Flaechen.\n\n')

# ---------------------------------------------------------------------- 1
W('1   HERKUNFT VON E_RKS\n')
W('-' * 112 + '\n')
W('   a) orca_freq/<rxn>_<Modell>/bs_sp.out, TZVP, Vor-Restart-Energie\n')
W('      NICHTS GEFUNDEN. Alle vier Outputs enthalten genau einen '
  'SCF-Block, "Stability Analysis indicates\n')
W('      a stable HF/KS wave function" und genau eine FINAL SINGLE POINT '
  'ENERGY. STABRestartUHFifUnstable hat\n')
W('      nirgends gegriffen, weil der SCF direkt auf der gebrochenen Loesung '
  'landete -- es gibt dort keine\n')
W('      restringierte Energie zum Ablesen.\n')
W('   b) stab_pipeline/<rxn>/result.json\n')
W('      GEFUNDEN, aber auf anderem Niveau: PySCF wB97M-V/def2-TZVP, grids 3, '
  'conv_tol 1e-10. Fuer\n')
W('      rxn0894/UMA-M steht dort e_rks, aber ext_stable = None und '
  'bs.e_uks = None: die Stabilitaets-\n')
W('      analyse ist an dieser Geometrie nicht durchgelaufen. Deshalb fehlt '
  'die Zeile auch in which_sheet.txt.\n')
W('   c) RKS-Singlepoint auf TZVPD\n')
if RKSD:
    W('      NEU GERECHNET. ! RKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF '
      'DEFGRID3, Thresh 1e-12, TCut 1e-13,\n')
    W('      sonst wortgleich zu ts_sp; keine Stabilitaetsanalyse, die '
      'restringierte Loesung ist hier gewollt.\n')
    W('      pipeline/job_rks_sheet.sh, Slurm-Job 10767516, vier Aufgaben, '
      'Ergebnisse in orca_rks_sheet/.\n')
else:
    W('      NOCH NICHT EINGESAMMELT. Job laeuft; '
      'results/rks_sheet_tzvpd.json fehlt noch.\n')
W('\n')

# ---------------------------------------------------------------------- 2
W('2   BARRIEREN AM IDENTISCHEN MODELL-TS\n')
W('-' * 112 + '\n')
W('2a  TZVPD durchgaengig, ORCA 5.0.4, wB97M-V/def2-TZVPD DEFGRID3 '
  'Thresh 1e-12.\n')
W('    Nullpunkt fuer alle drei Barrieren ist das Modell-Edukt desselben '
  'Laufs -- kein Niveau- und kein\n')
W('    Strukturwechsel innerhalb einer Zeile.\n\n')
W('%-9s %-7s %9s %9s %9s %10s %11s %9s   %s\n'
  % ('rxn', 'Modell', 'E_MLIP', 'E_RKS', 'E_BS', '|MLIP-RKS|', '|MLIP-BS|',
     '<S^2> TS', 'naeher an'))
W('%-9s %-7s %9s %9s %9s %10s %11s %9s\n'
  % ('', '', '[eV]', '[eV]', '[eV]', '[eV]', '[eV]', ''))
verd_d = {}
for rx, md in ROWS + CTRL:
    r = C.get((rx, md))
    bm, bb, s2 = num(r, 'barr_model'), num(r, 'barr_tzvpd'), num(r, 's2_ts_tzvpd')
    tag = '   <- Kontrolle' if (rx, md) in CTRL else ''
    e = RKSD.get((rx, md))
    if e is None:
        verd_d[(rx, md)] = MISS
        W('%-9s %-7s %9.4f %9s %9.4f %10s %11.4f %9.4f   %s%s\n'
          % (rx, LONG[md], bm, MISS, bb, MISS, abs(bm - bb), s2, MISS, tag))
        continue
    br = e['barr']
    dr, db = abs(bm - br), abs(bm - bb)
    gap = abs(br - bb)
    v = ('Flaechen fallen zusammen (%.3f eV)' % gap if gap < 0.05
         else ('RKS' if dr < db else 'BS'))
    verd_d[(rx, md)] = v
    W('%-9s %-7s %9.4f %9.4f %9.4f %10.4f %11.4f %9.4f   %s%s\n'
      % (rx, LONG[md], bm, br, bb, dr, db, s2, v, tag))
W('\n')

W('2b  TZVP-NAEHERUNG als Gegenprobe. Zwei Unterschiede zu 2a:\n')
W('    - E_RKS und E_BS aus PySCF/def2-TZVP, der Nullpunkt E0 aus ORCA. '
  'Ein Codewechsel innerhalb einer\n')
W('      Differenz; er verschiebt beide DFT-Barrieren um denselben Betrag und '
  'laesst den Vergleich RKS gegen BS\n')
W('      unberuehrt, nicht aber den Abstand zur MLIP-Barriere.\n')
W('    - Nullpunkt ist das REFERENZ-Edukt, nicht das Modell-Edukt. Laut '
  'which_sheet.txt liegen die beiden\n')
W('      Strukturen im Median 0.0005 A und maximal 0.0207 A auseinander.\n\n')
W('%-9s %-7s %9s %9s %9s %10s %11s   %s\n'
  % ('rxn', 'Modell', 'E_MLIP', 'E_RKS', 'E_BS', '|MLIP-RKS|', '|MLIP-BS|',
     'naeher an'))
W('%-9s %-7s %9s %9s %9s %10s %11s\n'
  % ('', '', '[eV]', '[eV]', '[eV]', '[eV]', '[eV]'))
verd_p = {}
for rx, md in ROWS + CTRL:
    st, e0 = STAB.get((rx, md)), E0_REF.get(rx)
    bm = num(C.get((rx, md)), 'barr_model')
    tag = '   <- Kontrolle' if (rx, md) in CTRL else ''
    if st is None or e0 is None or st['e_rks'] is None:
        verd_p[(rx, md)] = MISS
        W('%-9s %-7s %9s %9s %9s %10s %11s   %s%s\n'
          % (rx, LONG[md], MISS, MISS, MISS, MISS, MISS, MISS, tag))
        continue
    br = (st['e_rks'] - e0) * HA_EV
    if st['e_bs'] is not None:
        bb, bbs = (st['e_bs'] - e0) * HA_EV, '%9.4f'
    elif st['ext_stable'] is True:
        bb, bbs = br, '%9.4f'          # keine gebrochene Loesung an dem Punkt
    else:
        bb, bbs = None, '%9s'
    if bb is None:
        verd_p[(rx, md)] = MISS
        W('%-9s %-7s %9.4f %9.4f %9s %10.4f %11s   %s%s\n'
          % (rx, LONG[md], bm, br, MISS, abs(bm - br), MISS,
             'BS-Seite fehlt', tag))
        continue
    dr, db, gap = abs(bm - br), abs(bm - bb), abs(br - bb)
    v = ('Flaechen fallen zusammen (%.3f eV)' % gap if gap < 0.05
         else ('RKS' if dr < db else 'BS'))
    verd_p[(rx, md)] = v
    W('%-9s %-7s %9.4f %9.4f %9.4f %10.4f %11.4f   %s%s\n'
      % (rx, LONG[md], bm, br, bb, dr, db, v, tag))
W('\n')

# ---------------------------------------------------------------------- 3
W('3   GEGENPROBE: DER EINE RKS-FOLGER IM 40er-FLAECHENTEST\n')
W('-' * 112 + '\n')
WS = {}
for line in io.open('which_sheet.txt', encoding='utf-8'):
    f = line.split()
    if len(f) >= 8 and f[0].startswith('rxn') and f[1] in SLUG:
        WS[(f[0], SLUG[f[1]])] = dict(b_model=float(f[2]), b_rks=float(f[3]),
                                      b_bs=float(f[4]), d_rks=float(f[5]),
                                      d_bs=float(f[6]),
                                      verdict=' '.join(f[7:]))
W('   Quelle: which_sheet.txt, erzeugt von '
  'pipeline/which_sheet_did_models_learn.py.\n')
W('   Bilanz dort: %s.\n'
  % ', '.join('%s %d' % (t, sum(1 for w in WS.values() if w['verdict'] == t))
              for t in ('follows BS', 'follows RKS', 'sheets coincide here')))
rks_rows = [k for k, w in WS.items() if w['verdict'] == 'follows RKS']
for k in rks_rows:
    w = WS[k]
    W('   Der einzige RKS-Folger:  %s / %s   Modell %.2f   RKS %.2f   '
      'BS %.2f   |m-RKS| %.2f   |m-BS| %.2f\n'
      % (k[0], LONG[k[1]], w['b_model'], w['b_rks'], w['b_bs'], w['d_rks'],
         w['d_bs']))
inter = [k for k in rks_rows if k in ROWS]
W('   Ist das eine der drei?  %s\n'
  % ('ja, %s' % ', '.join('%s/%s' % (a, LONG[b]) for a, b in inter)
     if inter else 'nein'))
for a, b in ROWS:
    if (a, b) in inter:
        continue
    w = WS.get((a, b))
    W('   %s / %-6s steht dort als: %s\n'
      % (a, LONG[b], w['verdict'] if w else
         'nicht enthalten (stab_pipeline lieferte keine BS-Seite)'))
W('\n')

# ---------------------------------------------------------------------- 4
W('4   KONTROLLE rxn7060 / eSEN  (grosses Kraftresiduum, kleines '
  'Energieresiduum)\n')
W('-' * 112 + '\n')
r = C[('rxn7060', 'esen')]
W('   TZVPD:  <S^2> am TS = %.4f. Der UKS-Lauf mit STABPerform ist auf die '
  'geschlossenschalige Loesung\n' % num(r, 's2_ts_tzvpd'))
W('           konvergiert und meldet sie als stabil; eine davon verschiedene '
  'gebrochene Loesung gibt es an\n')
W('           diesem Punkt nicht.\n')
W('   TZVP:   stab_pipeline gibt ext_stable = True und keine BS-Loesung; '
  'which_sheet.txt fuehrt die Zeile\n')
W('           als "sheets coincide here" (RKS %.2f gegen BS %.2f).\n'
  % (WS[('rxn7060', 'esen')]['b_rks'], WS[('rxn7060', 'esen')]['b_bs']))
W('   Zahlen: Energieresiduum %+.4f eV, Kraftresiduum %.4f eV/A.\n'
  % (num(r, 'barr_model') - num(r, 'barr_tzvpd'), num(r, 'maxcomp_err')))
W('\n   Gemeldete Abweichung von der Erwartung: die Zeile ist nicht '
  '"klar BS-nah", sondern liegt gar nicht in der\n')
W('   instabilen Gruppe. Bei <S^2> = 0 fallen die beiden Flaechen zusammen, '
  'die Frage nach dem Sheet stellt sich\n')
W('   dort nicht. Ihr grosses Kraftresiduum bleibt damit ohne '
  'Sheet-Erklaerung.\n\n')

# ---------------------------------------------------------------------- 5
W('5   ANTWORT\n')
W('-' * 112 + '\n')
have = [verd_d[k] for k in ROWS if verd_d[k] != MISS]
uni = set(v for v in have if v in ('RKS', 'BS'))
if not have:
    sat = 'Die Energie-Ausreisser sitzen: noch nicht entscheidbar, ' \
          'E_RKS auf TZVPD fehlt.'
elif len(have) < len(ROWS):
    sat = 'Die Energie-Ausreisser sitzen uneinheitlich.'
elif uni == {'RKS'}:
    sat = 'Die Energie-Ausreisser sitzen auf RKS.'
elif len(uni) > 1:
    sat = 'Die Energie-Ausreisser sitzen uneinheitlich.'
else:
    sat = 'Die Energie-Ausreisser sitzen auf keiner der beiden.'
W(sat + '\n\n')
W('Zeile fuer Zeile:\n')
W('   %-9s %-6s   TZVPD %-34s   TZVP %s\n'
  % ('rxn', 'Modell', '', ''))
for k in ROWS:
    W('   %-9s %-6s   %-40s   %s\n'
      % (k[0], LONG[k[1]], verd_d[k], verd_p[k]))
W('   %-9s %-6s   %-40s   %s   <- Kontrolle\n'
  % (CTRL[0][0], LONG[CTRL[0][1]], verd_d[CTRL[0]], verd_p[CTRL[0]]))

W('\nAbstand zur naeheren der beiden Flaechen, damit "naeher an" nicht '
  'als "auf" gelesen wird:\n')
for k in ROWS + CTRL:
    r, e = C.get(k), RKSD.get(k)
    if e is None:
        W('   %-9s %-6s   %s\n' % (k[0], LONG[k[1]], MISS))
        continue
    bm, bb, br = num(r, 'barr_model'), num(r, 'barr_tzvpd'), e['barr']
    W('   %-9s %-6s   %.3f eV von %-3s   die beiden Flaechen liegen %.3f eV '
      'auseinander\n'
      % (k[0], LONG[k[1]], min(abs(bm - br), abs(bm - bb)),
         'RKS,' if abs(bm - br) < abs(bm - bb) else 'BS,', abs(br - bb)))

p = os.path.join(RES, 'sheet_check_outliers.txt')
io.open(p, 'w', encoding='utf-8').write(out.getvalue())
print(out.getvalue())
print('geschrieben:', p)
