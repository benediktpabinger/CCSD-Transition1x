"""Eine Tabelle: das OMol25-Label am gespeicherten Uebergangszustand JEDER
Transition1x-Reaktion. Bricht ab, wenn eine Pruefung fehlschlaegt, statt
eine falsche Datei zu schreiben (Disziplin von results/README.md).

Eingaben (aus pipeline/job_omol25_ts_scan.sh, Job 11011680)
    ~/omol25_ts_all.jsonl         Treffer des Scanners, >= 1 Zeile je gefundener
                                  Reaktion (OMol25 fuehrt manche Rahmen doppelt);
                                  20 MB, nicht in git
    results/omol25_census/omol25_census_ts_all.json
                                  je Reaktion n_frames und best_rmsd_A
    ~/t1x_ts_all.json             Split, Formel, Energie und groesste
                                  Kraftkomponente auf wB97x/6-31G(d) (H5)

Ausgabe  results/omol25_ts_all.csv, eine Zeile je Reaktion im H5 (10073),
auch die nicht gefundenen (found = 0; die bleiben nach docs/paper_decisions.md
D10 ausserhalb des Pools und stehen nur zur Dokumentation in der Tabelle):

    rxn, split, formula, n_atoms, found, n_hits, n_frames, rmsd_A, best_rmsd_A
    dup_conflict  1 wenn OMol25 denselben Rahmen mehrfach fuehrt und die
                  Wiederholungen sich in der Klasse (s2 > 0.05) widersprechen;
                  dann zaehlt die energetisch tiefere Loesung (D11)
    s2            <S^2> des OMol25-Labels (UKS, 20-Grad-Rotation, ORCA 6.0.0)
    group_omol    bs wenn s2 > 0.05, sonst closed; Vorauswahl, nicht Urteil --
                  das Urteil faellt unsere eigene Stabilitaetsanalyse
    e_omol_ev     OMol25-Energie, eV
    fmax_omol     groesste Kraftkomponente des OMol25-Labels, eV/A, ueber alle
                  3N Komponenten -- dieselbe Konvention wie f_dft_max in
                  omol25_model_geoms.csv. Das ist die Restkraft auf der
                  unrestricted Flaeche am Label-TS, Stufe 3 der Leiter.
    e_ref, fmax_ref   dasselbe auf wB97x/6-31G(d) aus dem H5, Stufe 1
    unrestricted, charge, spin, n_scf_steps, source

Pruefungen
    jeder Treffer rmsd < 1e-5 A
    doppelte Treffer einer Reaktion derselben Klasse stimmen in s2 (1e-4) und
        Energie (1e-4 eV); Klassenwidersprueche sind erlaubt, werden gezaehlt
        und in dup_conflict vermerkt, und die tiefere Loesung muss die
        broken-symmetry sein (sonst ABBRUCH: dann waere die Regel D11 falsch)
    die 44 Reaktionen aus results/omol25_label_compare.csv: s2 und Energie
        identisch (1e-6) -- der Anker an unsere eigene DFT
    jede Zeile neutrales Singulett, unrestricted, Atomzahl wie im H5
    Zeilenzahl = Reaktionen im H5

    python pipeline/omol25_ts_table.py
"""
import collections
import csv
import json
import os
import sys

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
S2_BREAK = 0.05
RMSD_MAX = 1e-5

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


ts = json.load(open(f'{H}/t1x_ts_all.json'))
census = json.load(open(os.path.join(REPO, 'results', 'omol25_census',
                                     'omol25_census_ts_all.json')))
hits = collections.defaultdict(list)
for line in open(f'{H}/omol25_ts_all.jsonl'):
    r = json.loads(line)
    hits[r['rxn']].append(r)

def is_bs(h):
    return abs(h['s_squared']) > S2_BREAK


rows = []
dup_bad = []        # Wiederholungen derselben Klasse, die nicht uebereinstimmen
dup_conflict = []   # Wiederholungen, die sich in der Klasse widersprechen
dup_wrong_way = []  # Widerspruch, bei dem die tiefere Loesung closed-shell waere
for rx, t in ts.items():
    hs = hits.get(rx, [])
    c = census.get(rx, {})
    row = dict(rxn=rx, split=t['split'], formula=t['formula'], n_atoms=t['n'],
               found=int(bool(hs)), n_hits=len(hs),
               n_frames=c.get('n_frames', 0), best_rmsd_A=c.get('best_rmsd_A'),
               e_ref=t['e_ref'], fmax_ref=t['fmax_ref'], dup_conflict=0)
    if hs:
        # D11: unter Wiederholungen desselben Rahmens zaehlt die energetisch
        # tiefere Loesung (das ist, was das OMol25-Protokoll anstrebt und was
        # unsere Stabilitaetsanalyse liefern wuerde).
        h = min(hs, key=lambda h: h['energy'])
        F = np.array(h['forces']).reshape(-1, 3)
        row.update(rmsd_A=h['rmsd_A'], s2=h['s_squared'],
                   group_omol='bs' if is_bs(h) else 'closed',
                   e_omol_ev=h['energy'], fmax_omol=float(np.abs(F).max()),
                   unrestricted=int(bool(h['unrestricted'])),
                   charge=h['charge'], spin=h['spin'],
                   n_scf_steps=h['n_scf_steps'], source=h['source'],
                   n_atoms_omol=len(h['numbers']))
        for o in hs:
            if o is h:
                continue
            if is_bs(o) != is_bs(h):
                row['dup_conflict'] = 1
                if not is_bs(h):
                    dup_wrong_way.append(rx)
            elif (abs(o['s_squared'] - h['s_squared']) > 1e-4
                    or abs(o['energy'] - h['energy']) > 1e-4):
                dup_bad.append(rx)
        if row['dup_conflict']:
            dup_conflict.append(rx)
    rows.append(row)

found = [r for r in rows if r['found']]
print('%d Reaktionen im H5, %d gefunden, %d nicht' % (len(rows), len(found),
                                                       len(rows) - len(found)))
for s in ('train', 'val', 'test'):
    S = [r for r in rows if r['split'] == s]
    print('   %-6s %6d  gefunden %6d  broken-symmetry %5d'
          % (s, len(S), sum(r['found'] for r in S),
             sum(r.get('group_omol') == 'bs' for r in S)))
print('broken-symmetry (s2 > %.2f): %d von %d gefundenen'
      % (S2_BREAK, sum(r['group_omol'] == 'bs' for r in found), len(found)))
miss = [r for r in rows if not r['found']]
if miss:
    b = [r['best_rmsd_A'] for r in miss if r['best_rmsd_A'] is not None]
    print('nicht gefunden: %d, davon %d ohne jeden Rahmen in OMol25; bester RMSD '
          'der uebrigen %s' % (len(miss), sum(r['n_frames'] == 0 for r in miss),
                              ('%.1e..%.1e A' % (min(b), max(b))) if b else '-'))
multi = [r for r in found if r['n_hits'] > 1]
print('mehrfach gefuehrte Rahmen: %d Reaktionen (2..%d Wiederholungen), '
      'Klassenwiderspruch bei %d: %s'
      % (len(multi), max((r['n_hits'] for r in multi), default=1),
         len(dup_conflict), ', '.join(dup_conflict) or '-'))

print()
print('Pruefungen')
check(all(r['rmsd_A'] < RMSD_MAX for r in found), 'jeder Treffer rmsd < %g A' % RMSD_MAX)
check(not dup_bad, 'Wiederholungen derselben Klasse stimmen in s2 und Energie ueberein'
      + ('' if not dup_bad else ': %s' % dup_bad))
check(not dup_wrong_way, 'bei Klassenwiderspruch ist die tiefere Loesung broken-symmetry'
      + ('' if not dup_wrong_way else ': %s' % dup_wrong_way))
ref = list(csv.DictReader(open(os.path.join(REPO, 'results',
                                            'omol25_label_compare.csv'))))
byrx = {r['rxn']: r for r in found}
anchor_bad = []
for v in ref:
    r = byrx.get(v['rxn'])
    if (r is None or abs(r['s2'] - float(v['s2_omol'])) > 1e-6
            or abs(r['e_omol_ev'] - float(v['e_omol_ev'])) > 1e-6):
        anchor_bad.append(v['rxn'])
check(not anchor_bad, 'Anker: die %d Reaktionen aus omol25_label_compare.csv '
      'identisch in s2 und Energie' % len(ref)
      + ('' if not anchor_bad else ': %s' % anchor_bad))
check(all(r['charge'] == 0 and r['spin'] == 1 for r in found),
      'jede gefundene Zeile neutrales Singulett')
check(all(r['unrestricted'] == 1 for r in found), 'jede gefundene Zeile unrestricted')
check(all(r['n_atoms_omol'] == r['n_atoms'] for r in found), 'Atomzahl wie im H5')
check(len(rows) == len(ts), 'eine Zeile je Reaktion im H5')

if fails:
    sys.exit('ABBRUCH: %d Pruefung(en) fehlgeschlagen:\n  ' % len(fails)
             + '\n  '.join(fails))

COLS = ['rxn', 'split', 'formula', 'n_atoms', 'found', 'n_hits', 'n_frames',
        'dup_conflict', 'rmsd_A', 'best_rmsd_A', 's2', 'group_omol',
        'e_omol_ev', 'fmax_omol',
        'e_ref', 'fmax_ref', 'unrestricted', 'charge', 'spin', 'n_scf_steps',
        'source']
FMT = {'rmsd_A': '%.2e', 'best_rmsd_A': '%.2e', 's2': '%.6f',
       'e_omol_ev': '%.9f', 'fmax_omol': '%.6f', 'e_ref': '%.9f',
       'fmax_ref': '%.6f'}
out = os.path.join(REPO, 'results', 'omol25_ts_all.csv')
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in sorted(rows, key=lambda r: r['rxn']):
        w.writerow([('' if r.get(c) is None else
                     (FMT[c] % r[c] if c in FMT else r[c])) for c in COLS])
print()
print('%s   %d Zeilen' % (os.path.relpath(out, REPO), len(rows)))
