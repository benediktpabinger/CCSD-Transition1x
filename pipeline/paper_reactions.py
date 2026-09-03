"""Die 45 Reaktionen: Auswahlmerkmale und Klasse, eine Zeile je Reaktion.

Bricht ab, wenn eine Pruefung fehlschlaegt.

QUELLEN, alle im Repo bzw. auf dem Cluster nachvollziehbar
    nfod        ~/fod_ranking.json, Feld 'results', je Eintrag
                {rxn, nfod, nfod_check, n_atoms, n_elec, energy_Ha}.
                279 Reaktionen. Wie diese 279 entstanden sind, steht nicht in
                der Datei -- fuer die Auswahl der 45 ist nur ihre Rangfolge
                nach nfod noetig.
    stratum     nicht von Hand, sondern aus derselben Vorschrift wie die
                urspruengliche Auswahl, woertlich uebernommen aus
                pipeline/which_sheet_did_models_learn.py (Zeilen 'sel = set(...)')
                und bestaetigt vom Kommentar in pipeline/job_neb_omol25_45.sh:
                    high    res[i]      for i in range(26)
                    spread  res[i - 1]  for i in [11, 40, 68, 97, 126, 154,
                                                  183, 212, 240, 269]
                    low     res[i]      for i in range(n - 10, n)
                res ist nach -nfod sortiert. rxn0896 hat Rang 11 und faellt in
                high und spread; es wird high zugeordnet, daher 45 statt 46.
    formula     ~/orca_neb_results/<rxn>/reactant.xyz, Summenformel des
                Referenz-Edukts. Gegen die drei Modell-Edukte geprueft.
    group_rxn   abgeleitet aus results/omol25_model_geoms.csv: unstable, wenn
                mindestens eine der drei Modellzeilen unstable_ts = 1 hat.
                Nicht von Hand gesetzt.

results/paper_reactions.csv
"""
import collections
import csv
import json
import os

H = '/home/energy/s242862'
OUT = f'{H}/results'
NF = f'{H}/fod_ranking.json'
MD = {'uma-s': 'uma_neb_results', 'uma-m': 'uma_m_neb_results',
      'esen': 'esen_neb_results'}
SPREAD_IDX = [11, 40, 68, 97, 126, 154, 183, 212, 240, 269]

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


def formula(p):
    if not os.path.exists(p):
        return None
    L = open(p, errors='replace').read().split('\n')
    n = int(L[0].split()[0])
    c = collections.Counter(l.split()[0] for l in L[2:2 + n])
    return ''.join('%s%d' % (e, c[e]) for e in sorted(c))


# ------------------------------------------------------- Auswahl nachbauen
res = sorted(json.load(open(NF))['results'], key=lambda r: -r['nfod'])
n = len(res)
nfod = {r['rxn']: r['nfod'] for r in res}
rank = {r['rxn']: i + 1 for i, r in enumerate(res)}

high = [res[i]['rxn'] for i in range(26)]
spread = [res[i - 1]['rxn'] for i in SPREAD_IDX]
low = [res[i]['rxn'] for i in range(n - 10, n)]

stratum = {}
for r in low:
    stratum[r] = 'low'
for r in spread:
    stratum[r] = 'spread'
for r in high:            # high gewinnt bei Ueberschneidung (rxn0896)
    stratum[r] = 'high'

# ------------------------------------------------------------- Klasse
geo = list(csv.DictReader(open(f'{OUT}/omol25_model_geoms.csv')))
byrxn = collections.defaultdict(list)
for r in geo:
    byrxn[r['rxn']].append(r)
group = {k: ('unstable' if any(x['unstable_ts'] == '1' for x in v) else 'stable')
         for k, v in byrxn.items()}

# --------------------------------------------------------------- Tabelle
rows = []
badform = []
for rx in sorted(stratum, key=lambda r: rank[r]):
    f = formula(f'{H}/orca_neb_results/{rx}/reactant.xyz')
    other = {formula(f'{H}/{d}/{rx}/reactant.xyz') for d in MD.values()}
    other.discard(None)
    if f is not None and other and other != {f}:
        badform.append('%s: Referenz %s, Modelle %s' % (rx, f, sorted(other)))
    rows.append({'rxn': rx, 'nfod': nfod[rx], 'stratum': stratum[rx],
                 'formula': f if f else 'NOT FOUND',
                 'group_rxn': group.get(rx, 'NOT FOUND')})

os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/paper_reactions.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['rxn', 'nfod', 'stratum', 'formula', 'group_rxn'])
    for r in rows:
        w.writerow([r['rxn'], '%.6f' % r['nfod'], r['stratum'], r['formula'],
                    r['group_rxn']])

# ------------------------------------------------------------ Pruefungen
print('DIE 45 REAKTIONEN')
print('=' * 74)
print('fod_ranking.json: %d Reaktionen, nfod %.3f bis %.3f'
      % (n, res[-1]['nfod'], res[0]['nfod']))
st = collections.Counter(r['stratum'] for r in rows)
gr = collections.Counter(r['group_rxn'] for r in rows)
fm = collections.Counter(r['formula'] for r in rows)
print()
print('Strata   %s' % dict(st))
print('Klassen  %s' % dict(gr))
print('Formeln  %s' % dict(fm))
print()
print('Pruefungen')
check(len(rows) == 45, 'n = 45 (%d)' % len(rows))
check(st['high'] == 26, 'Stratum high = 26 (%d)' % st['high'])
check(st['spread'] == 9, 'Stratum spread = 9 (%d)' % st['spread'])
check(st['low'] == 10, 'Stratum low = 10 (%d)' % st['low'])
check(gr['stable'] == 27, 'group_rxn stable = 27 (%d)' % gr['stable'])
check(gr['unstable'] == 18, 'group_rxn unstable = 18 (%d)' % gr['unstable'])
check(set(stratum) == set(byrxn),
      'dieselben 45 Reaktionen wie in omol25_model_geoms.csv')
check(all(len(v) == 3 for v in byrxn.values()),
      'je Reaktion genau drei Modellzeilen')
check(not any(r['formula'] == 'NOT FOUND' for r in rows),
      'Summenformel fuer alle Zeilen gefunden')
check(not badform, 'Referenz- und Modell-Edukt haben dieselbe Summenformel'
      + ('' if not badform else ': ' + '; '.join(badform)))
ov = set(high) & set(spread)
check(ov == {res[10]['rxn']},
      'genau eine Ueberschneidung high/spread: %s (Rang %d)'
      % (sorted(ov), rank[sorted(ov)[0]] if ov else -1))
check(max(rank[r] for r in high) == 26 and min(rank[r] for r in low) == n - 9,
      'high sind die Raenge 1-26, low die Raenge %d-%d' % (n - 9, n))

print()
print('geschrieben: results/paper_reactions.csv (%d Zeilen)' % len(rows))
if fails:
    raise SystemExit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))
