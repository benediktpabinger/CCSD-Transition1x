"""Die Zwillinge: zu jeder broken-symmetry-Reaktion mit N_FOD < 0.5 eine
closed-shell-Reaktion mit gleichem MR-Charakter (docs/paper_decisions.md D19,
Option A: Zwillinge plus Basislinie).

Oberhalb von N_FOD 0.5 werden ALLE closed-shell-Reaktionen gerechnet (Satz
cshigh); dort entstehen die Zwillinge bei der Auswertung aus den gerechneten.
Hier also nur die 365 broken-symmetry-Reaktionen unter 0.5
(results/s2pos_selection.csv), Pool: closed-shell (OMol25-s2 <= 1e-3, in
OMol25 gefunden) mit N_FOD < 0.5. Regel:
    1. im N_FOD-Fenster |dN_FOD| <= 0.05 (results/fod_ts_all.csv, am
       gespeicherten TS) die naechste closed-shell-Reaktion mit gleicher Zahl
       schwerer Atome (C, N, O), sonst die naechste im Fenster;
    2. ist das Fenster leer: ohne Partner (results/twins_unmatched_bs.csv;
       unter 0.5 ist der Pool dicht, erwartet: keine).
    Ohne Zuruecklegen, aufsteigend nach N_FOD.

Ausgaben wie mk_stab_tasks.py: ~/t1x_ts_twins/<rxn>.xyz,
~/t1x_ts_twins_tasks.txt, results/twins_selection.csv (eine Zeile je
closed-shell-Zwilling mit seinem broken-symmetry-Partner).

    python pipeline/mk_twins_tasks.py
"""
import bisect
import csv
import json
import os

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(REPO, 'results')
SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
XYZ_DIR = f'{H}/t1x_ts_twins'
TASKFILE = f'{H}/t1x_ts_twins_tasks.txt'
FOD_MAX = 0.5

omol = {r['rxn']: r for r in csv.DictReader(open(f'{RES}/omol25_ts_all.csv'))}
fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(f'{RES}/fod_ts_all.csv'))}
bs = [r['rxn'] for r in csv.DictReader(open(f'{RES}/s2pos_selection.csv')) if fod[r['rxn']] < FOD_MAX]
ts = json.load(open(f'{H}/t1x_ts_all.json'))


def n_heavy(rx):
    return sum(z > 1 for z in ts[rx]['numbers'])


pool = [rx for rx, r in omol.items()
        if r['found'] == '1' and float(r['s2']) <= 1e-3 and fod[rx] < FOD_MAX]
# Pool nach Schweratomzahl, je Groesse nach N_FOD sortiert
by_size = {}
for rx in pool:
    by_size.setdefault(n_heavy(rx), []).append((fod[rx], rx))
for k in by_size:
    by_size[k].sort()
all_sorted = sorted((fod[rx], rx) for rx in pool)
taken = set()


def take_nearest(lst, x):
    """naechstes freies Element zu x in der sortierten Liste (nfod, rxn):
    der erste freie Eintrag oberhalb und der erste unterhalb, davon der
    naehere."""
    i = bisect.bisect_left(lst, (x, ''))
    up = next(((f, r) for f, r in lst[i:] if r not in taken), None)
    down = next(((f, r) for f, r in reversed(lst[:i]) if r not in taken), None)
    cands = [c for c in (up, down) if c is not None]
    return min(cands, key=lambda c: abs(c[0] - x)) if cands else None


TOL = 0.05   # Paar nur, wenn |dN_FOD| <= TOL
pairs, unmatched = [], []
# Aufsteigend nach N_FOD: unten ist der closed-shell-Pool dicht, oben
# (N_FOD > 0.8) gibt es kaum closed-shell-Reaktionen -- die bleiben ohne
# Partner, das ist ein Befund, keine Luecke der Methode.
for rx_bs in sorted(bs, key=lambda r: fod[r]):
    x, nh = fod[rx_bs], n_heavy(rx_bs)
    lo = bisect.bisect_left(all_sorted, (x - TOL, ''))
    hi = bisect.bisect_right(all_sorted, (x + TOL, '~'))
    window = [(f, r) for f, r in all_sorted[lo:hi] if r not in taken]
    same = [c for c in window if n_heavy(c[1]) == nh]
    if same:
        cand, rule = min(same, key=lambda c: abs(c[0] - x)), 'size+nfod'
    elif window:
        cand, rule = min(window, key=lambda c: abs(c[0] - x)), 'nfod'
    else:
        unmatched.append(dict(rxn_bs=rx_bs, nfod_bs=x, n_heavy_bs=nh,
                              s2_omol=omol[rx_bs]['s2']))
        continue
    f, rx_cs = cand
    taken.add(rx_cs)
    pairs.append(dict(rxn=rx_cs, partner_bs=rx_bs, match_rule=rule,
                      nfod_cs=f, nfod_bs=x, n_heavy_cs=n_heavy(rx_cs), n_heavy_bs=nh,
                      split=omol[rx_cs]['split'], formula=omol[rx_cs]['formula'],
                      n_atoms=omol[rx_cs]['n_atoms'], s2_omol=omol[rx_cs]['s2'],
                      e_omol_ev=omol[rx_cs]['e_omol_ev'], fmax_omol=omol[rx_cs]['fmax_omol']))

assert len({p['rxn'] for p in pairs}) == len(pairs), 'Partner doppelt vergeben'
pairs.sort(key=lambda p: p['rxn'])
os.makedirs(XYZ_DIR, exist_ok=True)
with open(TASKFILE, 'w') as tf:
    for p in pairs:
        t = ts[p['rxn']]
        Z, R = t['numbers'], t['positions']
        nel = sum(Z)
        assert nel % 2 == 0
        xyz = f"{XYZ_DIR}/{p['rxn']}.xyz"
        with open(xyz, 'w') as fh:
            fh.write(f'{len(Z)}\n')
            fh.write(f"rxn={p['rxn']} split={p['split']} stored Transition1x TS "
                     f"(wB97x/6-31G(d)) s2_omol={float(p['s2_omol']):.6f} matched to {p['partner_bs']}\n")
            for z, (x, y, w) in zip(Z, R):
                fh.write(f'{SYM[z]:2s} {x:16.8f} {y:16.8f} {w:16.8f}\n')
        tf.write(f"{p['rxn']}:{xyz}:{nel}\n")
out = f'{RES}/twins_selection.csv'
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
    w.writeheader()
    w.writerows(pairs)

with open(f'{RES}/twins_unmatched_bs.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['rxn_bs', 'nfod_bs', 'n_heavy_bs', 's2_omol'])
    w.writeheader()
    w.writerows(sorted(unmatched, key=lambda u: u['rxn_bs']))

import numpy as np  # noqa: E402
d = np.array([abs(p['nfod_cs'] - p['nfod_bs']) for p in pairs])
print(f'{len(pairs)} Paare (|dN_FOD| <= {TOL}) aus einem Pool von {len(pool)} closed-shell-Reaktionen, '
      f'{len(unmatched)} broken-symmetry-Reaktionen ohne Partner')
print('   Regel size+nfod: %d, nfod im Fenster: %d'
      % (sum(p['match_rule'] == 'size+nfod' for p in pairs), sum(p['match_rule'] == 'nfod' for p in pairs)))
print('   |dN_FOD|: Median %.4f, 90 %% %.4f, max %.4f' % (np.median(d), np.percentile(d, 90), d.max()))
print('   N_FOD Median: gepaarte bs %.3f, cs %.3f; ohne Partner %.3f (min %.3f)'
      % (np.median([p['nfod_bs'] for p in pairs]), np.median([p['nfod_cs'] for p in pairs]),
         np.median([u['nfod_bs'] for u in unmatched]) if unmatched else float('nan'),
         min((u['nfod_bs'] for u in unmatched), default=float('nan'))))
print('   results/twins_unmatched_bs.csv')
print(f'{TASKFILE}\n{XYZ_DIR}/\n{os.path.relpath(out, REPO)}')
