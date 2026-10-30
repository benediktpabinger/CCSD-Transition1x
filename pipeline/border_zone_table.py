"""Sammler der Grenzzonen-Pruefung (docs/paper_decisions.md, D13).

Fuer die 213 Reaktionen aus results/border_zone_selection.csv (OMol25-Label
mit 0.001 < <S^2> <= 0.1 am gespeicherten Transition1x-TS) liest er die drei
ORCA-Laeufe aus ~/orca_border_zone/<rxn>/ (pipeline/job_border_zone.sh):
    rks_sp.out      RKS + EnGrad                  -> E_RKS, F_RKS
    uks_sp.out      UKS + STABPerform             -> lambda_min, <S^2>
    uks_engrad.out  EnGrad auf den uks_sp-Orbitalen -> E_BS, F_BS
und stellt sie dem OMol25-Label gegenueber.

Ausgabe results/border_zone.csv, eine Zeile je Reaktion:
    rxn, split, formula, n_atoms
    s2_omol, e_omol_ev, fmax_omol     das OMol25-Label (ORCA 6, 20-Grad-Rotation)
    lambda_min                       kleinster Eigenwert der Stabilitaetsanalyse
                                     der restringierten Loesung, au; < 0 heisst
                                     instabil, ORCA startet dann neu
    orca_unstable                    1 wenn ORCA "UNSTABLE" meldet
    uks_direct_bs                    1 wenn die UKS-Rechnung schon vor der
                                     Stabilitaetsanalyse auf der broken-symmetry-
                                     Loesung lag (s2_sp > 0.05 ohne Neustart);
                                     lambda_min gilt dann fuer diese Loesung
    s2_ours                          <S^2> der Loesung, auf der uks_engrad endet:
                                     derselbe konvergierte Zustand, aus dem E_BS
                                     und F_BS stammen (wie E_BS in hinge_t1x.py)
    s2_sp                            <S^2> am Ende von uks_sp, vor dem MORead-
                                     Neustart im EnGrad-Lauf. In der Grenzzone
                                     kann der Neustart die Loesung noch ein
                                     Stueck weiter relaxieren (rxn0030: 0.0508
                                     -> 0.0383, Energie 0.6 meV ueber RKS ->
                                     0.4 meV darunter); sp_engrad_differ zaehlt
                                     das, statt abzubrechen
    group_ours                       bs wenn s2_ours > 0.05, sonst closed
    depth_mev                        E_RKS - E_BS, meV (0 bei closed-shell)
    f_rks, f_bs                      groesste Kraftkomponente, eV/A, beide Flaechen
    e_rks_ha, e_bs_ha
    d_omol_rks_mev, d_omol_bs_mev    OMol25-Energie minus E_RKS bzw. E_BS, meV:
                                     sagt, auf welcher Loesung das Label sitzt
    scf_cyc_rks, scf_cyc_uks

Pruefungen
    alle drei Laeufe normal beendet (sonst nur Bericht, keine CSV)
    E_RKS >= E_BS - 1e-6 Ha (die UKS-Loesung liegt nie ueber der RKS). Die
        Toleranz ist 1e-6 statt 1e-7 Ha: bei marginalen Instabilitaeten
        (lambda_min ~ -3e-4 au) endet der Neustart bis zu 6.5e-7 Ha = 0.018 meV
        UEBER der RKS-Loesung (6 Reaktionen, bis 88 SCF-Zyklen) -- die
        "tiefere" Loesung ist dort tiefer um weniger als das SCF-Rauschen,
        die Tiefe ist praktisch null
    lambda_min > 0  =>  |E_RKS - E_BS| < 1e-8 Ha und |F_RKS - F_BS| < 5e-3 eV/A
        (Kraft-Toleranz 5e-3 statt 1e-3 wie in hinge_t1x.py: rxn3725 mit
        lambda_min = +0.0007 au, einer sehr weichen Mode, gibt 0.0016 eV/A)
    (uks_sp und uks_engrad auf verschiedenen Seiten von 0.05 ist in der
     Grenzzone erwartbar und wird nur gezaehlt, Spalte sp_engrad_differ)
    Anker: rxn7937 und rxn7945 liegen auch in ~/orca_hinge_t1x (Thesis, Job
        10773547); s2 und Energien muessen auf 1e-6 gleich sein

    python pipeline/border_zone_table.py [--partial]
        --partial: Bericht ueber die fertigen Reaktionen, keine CSV
"""
import csv
import os
import re
import sys

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# BZ_SET waehlt den Satz: Auswahl-CSV, Laufverzeichnis, Ausgabe-CSV, Anker.
#   border, zero   ORCA 5.0.4 auf xeon24el8 (2026-10-28/29), Anker = Thesis-Laeufe
#   s2pos, zero6   ORCA 6.0.1 auf el9 (D16): alle 1278 Reaktionen mit
#                  OMol25-s2 > 1e-3 (pipeline/mk_stab_tasks.py) und dieselbe
#                  200er-Stichprobe noch einmal; Anker = Gate-Laeufe
#                  ~/orca6_hinge_t1x (gleiche Version, gleiche Geometrie)
SETS = {'border': ('border_zone_selection.csv', 'orca_border_zone', 'border_zone.csv', 'orca_hinge_t1x'),
        'zero': ('zero_sample_selection.csv', 'orca_zero_sample', 'zero_sample.csv', 'orca_hinge_t1x'),
        's2pos': ('s2pos_selection.csv', 'orca6_s2pos', 'stab_s2pos_orca6.csv', 'orca6_hinge_t1x'),
        'zero6': ('zero_sample_selection.csv', 'orca6_zero_sample', 'zero_sample_orca6.csv', 'orca6_hinge_t1x'),
        'cshigh': ('cshigh_selection.csv', 'orca6_cshigh', 'stab_cshigh_orca6.csv', 'orca6_hinge_t1x'),
        'twins': ('twins_selection.csv', 'orca6_twins', 'stab_twins_orca6.csv', 'orca6_hinge_t1x'),
        'base': ('base_selection.csv', 'orca6_base', 'stab_base_orca6.csv', 'orca6_hinge_t1x')}
SEL_CSV, RUN_DIR, OUT_CSV, ANCHOR_DIR = SETS[os.environ.get('BZ_SET', 'border')]
RUNS = f'{H}/{RUN_DIR}'
ANCHOR_RUNS = f'{H}/{ANCHOR_DIR}'
EVA = 51.42208
HA_EV = 27.211386245988
S2_BREAK = 0.05
PARTIAL = '--partial' in sys.argv

E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')
CYC_RE = re.compile(r'SCF CONVERGED AFTER\s+(\d+)\s+CYCLES')
# kleinster Stabilitaets-Eigenwert der ERSTEN Analyse (restringierte Loesung):
# ORCA 5.0.4 druckt "Root  Eigenvalue (au)" / "0  -0.0310", ORCA 6.0.1
# "The eigenvalues of the stability matrix:" / "E( 0) = -0.03114230 Eh"
LAM_RE = re.compile(r'Root\s+Eigenvalue \(au\)\s*\n\s*0\s+([-\d.]+)'
                    r'|eigenvalues of the stability matrix:\s*\n\s*E\(\s*0\)\s*=\s*([-\d.]+)')
UNSTABLE_RE = re.compile(r'indicates an UNSTABLE|indicates that the wavefunction is unstable')

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
    e, s2, c, lam = (E_RE.findall(t), S2_RE.findall(t), CYC_RE.findall(t),
                     LAM_RE.findall(t))
    lam0 = next((a or b for a, b in lam), None)
    return dict(e=float(e[-1]) if e else None,
                s2=float(s2[-1]) if s2 else None,
                cyc=int(c[-1]) if c else None,
                lam=float(lam0) if lam0 is not None else None,
                unstable=int(bool(UNSTABLE_RE.search(t))))


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


def collect(d):
    rks, uks, ueg = (sp(f'{d}/rks_sp.out'), sp(f'{d}/uks_sp.out'),
                     sp(f'{d}/uks_engrad.out'))
    if any(x is None for x in (rks, uks, ueg)):
        return None
    f_bs = gradmax(f'{d}/uks_engrad.out')
    f_rks = gradmax(f'{d}/rks_sp.out')
    if f_bs is None or f_rks is None:
        return None
    return dict(lambda_min=uks['lam'], orca_unstable=uks['unstable'],
                s2_ours=ueg['s2'], s2_sp=uks['s2'],
                sp_engrad_differ=int((abs(uks['s2']) > S2_BREAK)
                                     != (abs(ueg['s2']) > S2_BREAK)),
                depth_mev=(rks['e'] - ueg['e']) * HA_EV * 1000.0,
                f_rks=f_rks, f_bs=f_bs, e_rks_ha=rks['e'], e_bs_ha=ueg['e'],
                scf_cyc_rks=rks['cyc'], scf_cyc_uks=uks['cyc'])


sel = list(csv.DictReader(open(os.path.join(REPO, 'results', SEL_CSV))))
rows, missing = [], []
for s in sel:
    r = collect(f"{RUNS}/{s['rxn']}")
    if r is None:
        missing.append(s['rxn'])
        continue
    row = dict(rxn=s['rxn'], split=s['split'], formula=s['formula'],
               n_atoms=int(s['n_atoms']), s2_omol=float(s['s2_omol']),
               e_omol_ev=float(s['e_omol_ev']), fmax_omol=float(s['fmax_omol']))
    row.update(r)
    row['group_ours'] = 'bs' if abs(r['s2_ours']) > S2_BREAK else 'closed'
    row['d_omol_rks_mev'] = (row['e_omol_ev'] - r['e_rks_ha'] * HA_EV) * 1000.0
    row['d_omol_bs_mev'] = (row['e_omol_ev'] - r['e_bs_ha'] * HA_EV) * 1000.0
    rows.append(row)

print('GRENZZONE DER <S^2>-SCHWELLE: %d von %d Reaktionen fertig'
      % (len(rows), len(sel)))
if not rows:
    sys.exit('noch keine fertige Reaktion')

bs = [r for r in rows if r['group_ours'] == 'bs']
cs = [r for r in rows if r['group_ours'] == 'closed']
print()
print('Unsere Stabilitaetsanalyse an den %d fertigen:' % len(rows))
print('   broken-symmetry (s2_ours > %.2f): %3d' % (S2_BREAK, len(bs)))
print('   closed-shell                   : %3d   davon ORCA "UNSTABLE" %d, '
      'lambda_min < 0 %d, s2_ours > 0.001 %d'
      % (len(cs), sum(r['orca_unstable'] for r in cs),
         sum((r['lambda_min'] or 0) < 0 for r in cs),
         sum(abs(r['s2_ours']) > 1e-3 for r in cs)))
dif = [r['rxn'] for r in rows if r['sp_engrad_differ']]
print('   uks_sp und uks_engrad auf verschiedenen Seiten von %.2f: %d%s'
      % (S2_BREAK, len(dif), (' (%s)' % ' '.join(dif[:10])) if dif else ''))
print()
print('Nach OMol25-<S^2> gestaffelt (Zeilen: OMol25-Bin; Spalten: unser Urteil)')
edges = [1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 7e-2, 1e-1]
print('   %-14s %4s %4s %7s %9s %9s' % ('s2_omol', 'bs', 'cs', 'n', 'med s2', 'med depth'))
for a, b in zip(edges, edges[1:]):
    S = [r for r in rows if a < r['s2_omol'] <= b]
    if not S:
        continue
    print('   %5.3f..%-5.3f  %4d %4d %7d %9.4f %9.2f'
          % (a, b, sum(r['group_ours'] == 'bs' for r in S),
             sum(r['group_ours'] == 'closed' for r in S), len(S),
             np.median([r['s2_ours'] for r in S]),
             np.median([r['depth_mev'] for r in S])))
if bs:
    d = np.array([r['depth_mev'] for r in bs])
    s = np.array([r['s2_ours'] for r in bs])
    print()
    print('   broken-symmetry: depth %.1f..%.1f meV (Median %.1f), s2_ours %.3f..%.3f'
          % (d.min(), d.max(), np.median(d), s.min(), s.max()))
print()
print('Auf welcher Loesung sitzt das OMol25-Label? |E_omol - E| in meV')
for lab, S in (('bs', bs), ('closed', cs)):
    if not S:
        continue
    a = np.abs([r['d_omol_rks_mev'] for r in S])
    b = np.abs([r['d_omol_bs_mev'] for r in S])
    print('   %-7s n=%3d   gegen E_RKS: Median %.3f max %.3f   gegen E_BS: Median %.3f max %.3f'
          % (lab, len(S), np.median(a), a.max(), np.median(b), b.max()))
print()
print('Restkraft am gespeicherten TS, Median eV/A:  f_rks %.3f   f_bs %.3f   fmax_omol %.3f'
      % (np.median([r['f_rks'] for r in rows]), np.median([r['f_bs'] for r in rows]),
         np.median([r['fmax_omol'] for r in rows])))

print()
print('Pruefungen')
check(not missing, 'alle %d Reaktionen fertig' % len(sel)
      + ('' if not missing else ' -- fehlen %d: %s%s' % (
          len(missing), ' '.join(missing[:8]), ' ...' if len(missing) > 8 else '')))
bad = [r['rxn'] for r in rows if r['e_rks_ha'] < r['e_bs_ha'] - 1e-6]
check(not bad, 'E_RKS >= E_BS - 1e-6 Ha ueberall' + ('' if not bad else ': %s' % bad))
neg = [r for r in rows if r['depth_mev'] < 0]
print('       (UKS marginal ueber RKS, |dE| < 1e-6 Ha: %d Zeilen, tiefste %.4f meV)'
      % (len(neg), min((r['depth_mev'] for r in neg), default=0)))
# lambda_min gehoert zur ERSTEN Stabilitaetsanalyse, also zu der Loesung, auf
# die die UKS-Rechnung aus dem Startguess zuerst konvergiert ist. Meist ist
# das die restringierte (s2 = 0); bei einigen tiefen Faellen landet die SCF
# aber direkt auf der broken-symmetry-Loesung (uks_direct_bs = 1, z. B.
# rxn2531: s2 0.68, lambda +0.056, Tiefe 261 meV) -- dann ist lambda_min > 0
# die Stabilitaet DIESER Loesung, und E_BS != E_RKS ist richtig.
for r in rows:
    r['uks_direct_bs'] = int(abs(r['s2_sp']) > S2_BREAK and not r['orca_unstable'])
nd = sum(r['uks_direct_bs'] for r in rows)
print('       (UKS direkt auf der broken-symmetry-Loesung konvergiert, ohne Neustart: %d Zeilen)' % nd)
st = [r for r in rows if r['lambda_min'] is not None and r['lambda_min'] > 0
      and abs(r['s2_ours']) <= S2_BREAK and not r['uks_direct_bs']]
if st:
    # 1e-6 Ha (0.03 meV): zwei getrennte SCF-Laeufe (RKS, UKS ohne Neustart)
    # auf derselben Loesung stimmen in ORCA 6.0.1 (LEANSCF) auf ~1e-7 Ha
    check(max(abs(r['e_rks_ha'] - r['e_bs_ha']) for r in st) < 1e-6,
          'lambda_min > 0 auf der closed-shell-Loesung: |E_RKS - E_BS| < 1e-6 Ha (%d Zeilen, max %.1e)'
          % (len(st), max(abs(r['e_rks_ha'] - r['e_bs_ha']) for r in st)))
    check(max(abs(r['f_rks'] - r['f_bs']) for r in st) < 5e-3,
          'lambda_min > 0 auf der closed-shell-Loesung: |F_RKS - F_BS| < 5e-3 eV/A (max %.4f)'
          % max(abs(r['f_rks'] - r['f_bs']) for r in st))
check(all(r['lambda_min'] is not None for r in rows), 'lambda_min in jeder Zeile gelesen')
anch = []
# Anker: alle Reaktionen des Satzes, die auch im Ankerverzeichnis liegen
# (border/zero: rxn7937, rxn7945 in den Thesis-Laeufen; s2pos: die 17
# broken-symmetry-Reaktionen der 45 in den Gate-Laeufen)
for rx in sorted(r['rxn'] for r in rows if os.path.isdir(f'{ANCHOR_RUNS}/{r["rxn"]}')):
    a = collect(f'{ANCHOR_RUNS}/{rx}')
    b = next((r for r in rows if r['rxn'] == rx), None)
    if a is None or b is None:
        continue
    anch.append(rx)
    ok = (abs(a['s2_ours'] - b['s2_ours']) < 1e-6
          and abs(a['e_rks_ha'] - b['e_rks_ha']) < 1e-6
          and abs(a['e_bs_ha'] - b['e_bs_ha']) < 1e-6)
    check(ok, 'Anker %s reproduziert orca_hinge_t1x (s2 %.6f/%.6f, dE_bs %.1e Ha)'
          % (rx, b['s2_ours'], a['s2_ours'], abs(a['e_bs_ha'] - b['e_bs_ha'])))

if PARTIAL or missing:
    print('\nkeine CSV geschrieben (%s)' % ('--partial' if PARTIAL else 'unvollstaendig'))
    sys.exit(0 if PARTIAL else 1)
if fails:
    sys.exit('ABBRUCH: %d Pruefung(en) fehlgeschlagen:\n  ' % len(fails)
             + '\n  '.join(fails))

COLS = ['rxn', 'split', 'formula', 'n_atoms', 's2_omol', 'e_omol_ev', 'fmax_omol',
        'lambda_min', 'orca_unstable', 'uks_direct_bs', 's2_ours', 's2_sp', 'sp_engrad_differ',
        'group_ours', 'depth_mev',
        'f_rks', 'f_bs', 'e_rks_ha', 'e_bs_ha', 'd_omol_rks_mev', 'd_omol_bs_mev',
        'scf_cyc_rks', 'scf_cyc_uks']
FMT = {'s2_omol': '%.6f', 'e_omol_ev': '%.9f', 'fmax_omol': '%.6f',
       'lambda_min': '%.6f', 's2_ours': '%.6f', 's2_sp': '%.6f', 'depth_mev': '%.3f',
       'f_rks': '%.6f', 'f_bs': '%.6f', 'e_rks_ha': '%.9f', 'e_bs_ha': '%.9f',
       'd_omol_rks_mev': '%.3f', 'd_omol_bs_mev': '%.3f'}
out = os.path.join(REPO, 'results', OUT_CSV)
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in sorted(rows, key=lambda r: r['rxn']):
        w.writerow([('' if r.get(c) is None else
                     (FMT[c] % r[c] if c in FMT else r[c])) for c in COLS])
print('\n%s   %d Zeilen' % (os.path.relpath(out, REPO), len(rows)))
