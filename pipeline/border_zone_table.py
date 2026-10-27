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
    s2_ours                          <S^2> der Loesung, auf der uks_sp endet
    group_ours                       bs wenn s2_ours > 0.05, sonst closed
    depth_mev                        E_RKS - E_BS, meV (0 bei closed-shell)
    f_rks, f_bs                      groesste Kraftkomponente, eV/A, beide Flaechen
    e_rks_ha, e_bs_ha
    d_omol_rks_mev, d_omol_bs_mev    OMol25-Energie minus E_RKS bzw. E_BS, meV:
                                     sagt, auf welcher Loesung das Label sitzt
    scf_cyc_rks, scf_cyc_uks

Pruefungen
    alle drei Laeufe normal beendet (sonst nur Bericht, keine CSV)
    uks_engrad und uks_sp auf derselben Seite von 0.05
    E_RKS >= E_BS - 1e-7 Ha (die UKS-Loesung liegt nie ueber der RKS)
    lambda_min > 0  =>  |E_RKS - E_BS| < 1e-8 Ha und |F_RKS - F_BS| < 1e-3 eV/A
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
RUNS = f'{H}/orca_border_zone'
ANCHOR_RUNS = f'{H}/orca_hinge_t1x'
EVA = 51.42208
HA_EV = 27.211386245988
S2_BREAK = 0.05
PARTIAL = '--partial' in sys.argv

E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')
CYC_RE = re.compile(r'SCF CONVERGED AFTER\s+(\d+)\s+CYCLES')
LAM_RE = re.compile(r'Root\s+Eigenvalue \(au\)\s*\n\s*0\s+([-\d.]+)')

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
    return dict(e=float(e[-1]) if e else None,
                s2=float(s2[-1]) if s2 else None,
                cyc=int(c[-1]) if c else None,
                lam=float(lam[0]) if lam else None,
                unstable=int('indicates an UNSTABLE' in t))


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
                s2_ours=uks['s2'], s2_engrad=ueg['s2'],
                depth_mev=(rks['e'] - ueg['e']) * HA_EV * 1000.0,
                f_rks=f_rks, f_bs=f_bs, e_rks_ha=rks['e'], e_bs_ha=ueg['e'],
                scf_cyc_rks=rks['cyc'], scf_cyc_uks=uks['cyc'])


sel = list(csv.DictReader(open(os.path.join(REPO, 'results',
                                            'border_zone_selection.csv'))))
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
check(all((abs(r['s2_ours']) > S2_BREAK) == (abs(r['s2_engrad']) > S2_BREAK)
          for r in rows), 'uks_engrad und uks_sp auf derselben Seite von %.2f' % S2_BREAK)
bad = [r['rxn'] for r in rows if r['e_rks_ha'] < r['e_bs_ha'] - 1e-7]
check(not bad, 'E_RKS >= E_BS ueberall' + ('' if not bad else ': %s' % bad))
st = [r for r in rows if r['lambda_min'] is not None and r['lambda_min'] > 0]
if st:
    check(max(abs(r['e_rks_ha'] - r['e_bs_ha']) for r in st) < 1e-8,
          'lambda_min > 0: |E_RKS - E_BS| < 1e-8 Ha (%d Zeilen)' % len(st))
    check(max(abs(r['f_rks'] - r['f_bs']) for r in st) < 1e-3,
          'lambda_min > 0: |F_RKS - F_BS| < 1e-3 eV/A')
check(all(r['lambda_min'] is not None for r in rows), 'lambda_min in jeder Zeile gelesen')
anch = []
for rx in ('rxn7937', 'rxn7945'):
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
        'lambda_min', 'orca_unstable', 's2_ours', 'group_ours', 'depth_mev',
        'f_rks', 'f_bs', 'e_rks_ha', 'e_bs_ha', 'd_omol_rks_mev', 'd_omol_bs_mev',
        'scf_cyc_rks', 'scf_cyc_uks']
FMT = {'s2_omol': '%.6f', 'e_omol_ev': '%.9f', 'fmax_omol': '%.6f',
       'lambda_min': '%.6f', 's2_ours': '%.6f', 'depth_mev': '%.3f',
       'f_rks': '%.6f', 'f_bs': '%.6f', 'e_rks_ha': '%.9f', 'e_bs_ha': '%.9f',
       'd_omol_rks_mev': '%.3f', 'd_omol_bs_mev': '%.3f'}
out = os.path.join(REPO, 'results', 'border_zone.csv')
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in sorted(rows, key=lambda r: r['rxn']):
        w.writerow([('' if r.get(c) is None else
                     (FMT[c] % r[c] if c in FMT else r[c])) for c in COLS])
print('\n%s   %d Zeilen' % (os.path.relpath(out, REPO), len(rows)))
