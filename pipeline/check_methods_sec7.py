# -*- coding: utf-8 -*-
"""Prueft Paragraph 7 von docs/methods_for_paper.md gegen die Hinge-Tabellen.

Bricht ab, wenn eine Zahl im Fliesstext nicht mehr zu results/hinge_t1x.csv
oder results/hinge_omol25.csv passt. Gegenstueck zur CORE-Sperre in
pipeline/hinge_tables.py: dort ist eingefroren, was das Skript ausgibt, hier,
was der Text behauptet. Beides zusammen schliesst die Luecke, durch die
Bildunterschriften und Methodenteile still veralten.

Geprueft werden: die acht Kernzahlen-Zeilen, die Zeilenzahlen 45 und 33, die
zwei group/group_local-Wechsler samt <S^2>, die drei Kipp-Zeilen samt
Verhaeltnissen und f_rks-Spanne, die 13 Zeilen ueber 0.05 eV/A einzeln, die 12
fehlenden Reaktionen einzeln, die Kreuzung 9 stabil / 3 instabil, der
Fussnotensatz, die rxn7945-Begruendung der gemischten Schranke und die
Nachfolgerzahl der obsoleten TZVP-Groesse in Paragraph 6, die Leiter
f_ref -> f_rks -> f_bs samt eingefrorenem f_ref-Median. Zusaetzlich,
dass keine Markdown-Tabelle in Paragraph 7 eine abweichende Spaltenzahl hat
und dass der Text nicht mehr auf das abgeloeste docs/methods_hinge.md verweist.

Lauf: python pipeline/check_methods_sec7.py   (aus dem Repo-Wurzelverzeichnis)
"""
import csv
import io
import re
import statistics as st

md = io.open('docs/methods_for_paper.md', encoding='utf-8').read()
sec = md[md.index('## 7. The hinge measurements'):]
flat = ' '.join(sec.split())
t1 = list(csv.DictReader(io.open('results/hinge_t1x.csv', encoding='utf-8')))
t2 = list(csv.DictReader(io.open('results/hinge_omol25.csv', encoding='utf-8')))
bad = []
med = lambda S, c: st.median([float(r[c]) for r in S])


def want(txt, why):
    txt = ' '.join(txt.split()).replace('-', '−')
    if txt not in flat.replace('-', '−'):
        bad.append('%s -- fehlt: %r' % (why, txt))


# --- Markdown-Tabellen: gleiche Spaltenzahl je Block ------------------------
prev, width = None, None
for i, l in enumerate(sec.split('\n')):
    if not l.startswith('|'):
        continue
    n = len(re.split(r'(?<!\\)\|', l.strip())) - 2
    if prev is None or i != prev + 1:
        width = n
    elif n != width:
        bad.append('Tabellenzeile %d: %d Zellen statt %d -- %s'
                   % (i, n, width, l[:60]))
    prev = i

# --- Kernzahlen -------------------------------------------------------------
for tag, rows_ in (('T1x label', t1), ('def2-TZVPD re-optimised', t2)):
    for lab in ('stable', 'unstable'):
        S = [r for r in rows_ if r['group_local'] == lab]
        want('| %s | %s | %d | %.4f | %.4f | %.2f |'
             % (tag, lab, len(S), med(S, 'f_rks'), med(S, 'f_bs'),
                med(S, 'ratio')), 'Kernzahl %s/%s' % (tag, lab))

# --- Zeilenzahlen -----------------------------------------------------------
want('(%d rows) measures what the training labels' % len(t1), 'n Tabelle 1')
want('(%d rows) isolates the surface effect' % len(t2), 'n Tabelle 2')
want('label transition states, %d of 45' % len(t1), 'Geometrien Tabelle 1')
want('re-optimised at ωB97M-V/def2-TZVPD, %d of 45' % len(t2),
     'Geometrien Tabelle 2')

# --- die Leiter f_ref -> f_rks -> f_bs, Unterabschnitt "Reading the T1x table"
fref = sorted(float(r['f_ref']) for r in t1)
want('median of **%.4f eV Å⁻¹** across the %d rows' % (st.median(fref), len(t1)),
     'f_ref-Median (eingefroren)')
want('%d of them below 0.05' % sum(1 for v in fref if v < 0.05),
     'f_ref unter 0.05')
top = max(t1, key=lambda r: float(r['f_ref']))
want('%s at %.4f' % (top['rxn'], float(top['f_ref'])), 'f_ref-Ausreisser')
S = {lab: [r for r in t1 if r['group_local'] == lab]
     for lab in ('stable', 'unstable')}
want('`f_rks` medians %.4f (stable, n = %d) and\n%.4f (unstable, n = %d)'
     % (med(S['stable'], 'f_rks'), len(S['stable']),
        med(S['unstable'], 'f_rks'), len(S['unstable'])), 'Leiter f_rks')
want('a factor of about 40 over `f_ref`', 'Leiter-Faktor')
want('(`f_bs` %.4f, ratio **%.2f**)'
     % (med(S['unstable'], 'f_bs'), med(S['unstable'], 'ratio')), 'Leiter f_bs')
U2 = [r for r in t2 if r['group_local'] == 'unstable']
want('it barely moves, %.4f to %.4f'
     % (med(S['unstable'], 'f_bs'), med(U2, 'f_bs')), 'Zaehler bewegt sich kaum')
want('by a factor of fourteen, %.4f to %.4f'
     % (med(S['unstable'], 'f_rks'), med(U2, 'f_rks')), 'Nenner faellt')
if not 13.5 <= med(S['unstable'], 'f_rks') / med(U2, 'f_rks') < 14.5:
    bad.append('Nennerfaktor %.2f, im Text steht vierzehn'
               % (med(S['unstable'], 'f_rks') / med(U2, 'f_rks')))
if not 35 <= med(S['stable'], 'f_rks') / st.median(fref) < 45:
    bad.append('f_rks/f_ref = %.1f, im Text steht etwa 40'
               % (med(S['stable'], 'f_rks') / st.median(fref)))

# --- die zwei Wechsler ------------------------------------------------------
sw = {r['rxn']: r for r in t1 if r['group'] != r['group_local']}
sw2 = {r['rxn']: r for r in t2 if r['group'] != r['group_local']}
if sorted(sw) != ['rxn10054', 'rxn1147'] or sorted(sw2) != sorted(sw):
    bad.append('Wechslermenge %s / %s' % (sorted(sw), sorted(sw2)))
for rx in sorted(sw):
    fmt = lambda v: ('%.4f' % v) if abs(v) > 1e-6 else '%.6f' % v
    want('| %s | %s | %s | %s / %s |'
         % (rx, sw[rx]['group'], sw[rx]['group_local'],
            fmt(float(sw[rx]['s2_ts'])), fmt(float(sw2[rx]['s2_ts']))),
         'Wechsler %s' % rx)

# --- die drei Kipp-Zeilen ---------------------------------------------------
tilt = sorted((r for r in t1 if r['group_local'] == 'unstable'
               and float(r['f_bs']) < float(r['f_rks'])),
              key=lambda r: r['rxn'])
if [r['rxn'] for r in tilt] != ['rxn4113', 'rxn6196', 'rxn7957']:
    bad.append('Kippmenge %s' % [r['rxn'] for r in tilt])
else:
    want('rxn4113, rxn6196 and rxn7957 — ratios %.3f, %.3f and %.3f'
         % tuple(float(r['ratio']) for r in tilt), 'Kipp-Verhaeltnisse')
    want('%.2f to %.2f eV Å⁻¹ for these three'
         % (min(float(r['f_rks']) for r in tilt),
            max(float(r['f_rks']) for r in tilt)), 'Kipp-f_rks-Spanne')
if [r for r in t2 if r['group_local'] == 'unstable'
        and float(r['f_bs']) < float(r['f_rks'])]:
    bad.append('Kippzeile in Tabelle 2 vorhanden')

# --- die 13 ueber 0.05 ------------------------------------------------------
over = sorted([r for r in t2 if float(r['f_rks']) > 0.05],
              key=lambda r: -float(r['f_rks']))
if len(over) != 13:
    bad.append('%d Zeilen ueber 0.05, nicht 13' % len(over))
want('Thirteen of the %d rows in Table 2' % len(t2), 'Anzahl 13')
want('median %.4f over all %d, maximum %.4f (%s)'
     % (med(t2, 'f_rks'), len(t2), float(over[0]['f_rks']), over[0]['rxn']),
     'Median/Max f_rks Tabelle 2')
for r in over[1:]:
    want('%s %.4f' % (r['rxn'], float(r['f_rks'])), 'Liste %s' % r['rxn'])

# --- die 12 fehlenden -------------------------------------------------------
miss = sorted({r['rxn'] for r in t1} - {r['rxn'] for r in t2})
if len(miss) != 12:
    bad.append('%d fehlende, nicht 12' % len(miss))
loc = {r['rxn']: r['group_local'] for r in t1}
mu = sum(1 for rx in miss if loc[rx] == 'unstable')
au = sum(1 for r in t1 if r['group_local'] == 'unstable')
want('**%d stable, %d unstable** — %.0f %% unstable against %.0f %% over the '
     'full %d' % (len(miss) - mu, mu, 100 * mu / len(miss),
                  100 * au / len(t1), len(t1)), 'Kreuzung stabil/instabil')
want('(%d of %d, %.0f %%, against %.0f %% overall).'
     % (mu, len(miss), 100 * mu / len(miss), 100 * au / len(t1)), 'Fussnote')
for rx in miss:
    want('| %s | 0.' % rx, 'fehlende Reaktion %s' % rx)

# --- rxn7945 ----------------------------------------------------------------
r = [x for x in t1 if x['rxn'] == 'rxn7945'][0]
want('⟨S²⟩ = %.4f' % float(r['s2_ts']), 'rxn7945 <S^2>')
want('%.2f·10⁻³ eV Å⁻¹'
     % (abs(float(r['f_bs']) - float(r['f_rks'])) * 1e3), 'rxn7945 Delta')
want('`f_rks` = %.4f is 0.3 %% relative' % float(r['f_rks']), 'rxn7945 f_rks')

# --- 1.8695 als Nachfolger der obsoleten 1.697 ------------------------------
mb = med([r for r in t2 if r['group_local'] == 'unstable'], 'f_bs')
if ('**%.4f eV Å⁻¹**' % mb) not in md:
    bad.append('Nachfolgerzahl %.4f fehlt in Paragraph 6' % mb)

# --- kein Verweis mehr auf das abgeloeste Dokument -------------------------
if 'methods_hinge.md' in sec:
    bad.append('Paragraph 7 verweist noch auf das abgeloeste methods_hinge.md')

print('VERIFY Paragraph 7 gegen die CSVs')
for b in bad:
    print('  FEHL ' + b)
print('  %d Beanstandungen' % len(bad))
if bad:
    raise SystemExit(1)
