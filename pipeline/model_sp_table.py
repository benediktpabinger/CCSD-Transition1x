"""Modell gegen DFT-Label an den gespeicherten Transition1x-Uebergangszustaenden.

Verbindet die Modell-Einzelpunkte (pipeline/model_sp_ts.py, JSONL in
~/model_sp_ts/) mit dem OMol25-Label derselben Geometrie
(~/omol25_ts_all.jsonl, bei Mehrfachtreffern die energetisch tiefste Zeile,
D11) und mit Gruppe, <S^2> und N_FOD aus results/.

Je (Reaktion, Modell, Spin) eine Zeile:
    rxn, model, spin, split, formula, n_atoms, group_omol, s2_omol, nfod
    e_model_ev, e_omol_ev, de_mev            Modell minus Label, meV (Totalenergie)
    fmax_model, fmax_omol                     groesste Kraftkomponente, eV/A
    f_mae                                     mittlerer |F_model - F_omol| ueber 3N
    f_maxdiff                                 groesste Komponentendifferenz
    f_cos                                     Kosinus zwischen den 3N-Vektoren

Zwei Modi:
    --tag spintest   (Dateien spintest_<model>_spin<k>.jsonl, 45 Anker):
        druckt je Modell und Spin die Mediane gegen das Label -- die
        Spin-Konvention, die zum Singulett-Label passt, gewinnt.
        Schreibt results/model_sp_spintest.csv.
    --tag all        (Dateien all_<model>_spin1.jsonl, alle Reaktionen):
        schreibt results/model_sp_ts_all.csv und druckt BS gegen CS.

Pruefungen: jede Modellzeile hat ein Label, Atomzahl stimmt, keine NaN,
im Modus all: 10009 Reaktionen je Modell.

    python pipeline/model_sp_table.py --tag spintest
    python pipeline/model_sp_table.py --tag all
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np

H = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='all')
a = ap.parse_args()

# OMol25-Label: tiefste Zeile je Reaktion (D11)
label = {}
for line in open(f'{H}/omol25_ts_all.jsonl'):
    d = json.loads(line)
    if d['rxn'] not in label or d['energy'] < label[d['rxn']]['energy']:
        label[d['rxn']] = d
meta = {r['rxn']: r for r in csv.DictReader(open(os.path.join(REPO, 'results', 'omol25_ts_all.csv')))}
fod = {r['rxn']: float(r['nfod']) for r in csv.DictReader(open(os.path.join(REPO, 'results', 'fod_ts_all.csv')))}

rows, nolabel, badn = [], [], []
files = sorted(glob.glob(f'{H}/model_sp_ts/{a.tag}_*_spin*.jsonl'))
if not files:
    sys.exit(f'keine Dateien {H}/model_sp_ts/{a.tag}_*.jsonl')
for p in files:
    m = re.match(r'.*/%s_(.+)_spin(\d+)\.jsonl' % a.tag, p)
    model, spin = m.group(1), int(m.group(2))
    for line in open(p):
        d = json.loads(line)
        rx = d['rxn']
        if rx not in label:
            # ohne Label nur die 64 nicht in OMol25 gefundenen (D10); alles
            # andere waere ein Fehler
            if meta.get(rx, {}).get('found') != '0':
                nolabel.append(rx)
            continue
        L = label[rx]
        Fm = np.array(d['forces_ev_a'])
        Fl = np.array(L['forces']).reshape(-1, 3)
        if Fm.shape != Fl.shape:
            badn.append(rx)
            continue
        o = meta[rx]
        rows.append(dict(
            rxn=rx, model=model, spin=spin, split=o['split'], formula=o['formula'],
            n_atoms=int(o['n_atoms']), group_omol=o['group_omol'], s2_omol=float(o['s2']),
            nfod=fod.get(rx), e_model_ev=d['energy_ev'], e_omol_ev=L['energy'],
            de_mev=(d['energy_ev'] - L['energy']) * 1e3,
            fmax_model=float(np.abs(Fm).max()), fmax_omol=float(np.abs(Fl).max()),
            f_mae=float(np.abs(Fm - Fl).mean()), f_maxdiff=float(np.abs(Fm - Fl).max()),
            f_cos=float(np.dot(Fm.ravel(), Fl.ravel()) /
                        (np.linalg.norm(Fm) * np.linalg.norm(Fl) + 1e-30))))

print('%d Zeilen aus %d Dateien (%s)' % (len(rows), len(files), a.tag))
models = sorted({r['model'] for r in rows})
spins = sorted({r['spin'] for r in rows})


def med(S, k):
    return np.median([r[k] for r in S]) if S else float('nan')


print()
print('Modell gegen OMol25-Label, Mediane:   dE [meV]  |dE| [meV]   f_mae [eV/A]  f_maxdiff  f_cos   n')
for mo in models:
    for sp in spins:
        S = [r for r in rows if r['model'] == mo and r['spin'] == sp]
        if not S:
            continue
        print('   %-10s spin %d   %+9.1f  %9.1f   %10.4f  %9.4f  %6.3f  %5d'
              % (mo, sp, med(S, 'de_mev'), np.median(np.abs([r['de_mev'] for r in S])),
                 med(S, 'f_mae'), med(S, 'f_maxdiff'), med(S, 'f_cos'), len(S)))
if a.tag == 'all' or len(models) * len(spins) > 1:
    print()
    print('Nach Gruppe (OMol25-Label), Mediane je Modell/Spin:')
    for mo in models:
        for sp in spins:
            for g in ('bs', 'closed'):
                S = [r for r in rows if r['model'] == mo and r['spin'] == sp and r['group_omol'] == g]
                if S:
                    print('   %-10s spin %d  %-7s n=%5d  |dE| %7.1f meV   f_mae %.4f   fmax_model %.3f  fmax_omol %.3f'
                          % (mo, sp, g, len(S), np.median(np.abs([r['de_mev'] for r in S])),
                             med(S, 'f_mae'), med(S, 'fmax_model'), med(S, 'fmax_omol')))

print()
print('Pruefungen')
check(not nolabel, 'jede Modellzeile hat ein OMol25-Label'
      + ('' if not nolabel else ': %d ohne (%s ...)' % (len(nolabel), ' '.join(sorted(set(nolabel))[:5]))))
check(not badn, 'Atomzahl Modell = Label' + ('' if not badn else ': %s' % badn[:5]))
check(all(np.isfinite([r['e_model_ev'], r['f_mae']]).all() for r in rows), 'keine NaN')
if a.tag == 'all':
    cnt = collections.Counter((r['model'], r['spin']) for r in rows)
    check(all(v == 10009 for v in cnt.values()), '10009 Reaktionen je Modell: %s' % dict(cnt))
if fails:
    sys.exit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))

COLS = ['rxn', 'model', 'spin', 'split', 'formula', 'n_atoms', 'group_omol', 's2_omol', 'nfod',
        'e_model_ev', 'e_omol_ev', 'de_mev', 'fmax_model', 'fmax_omol', 'f_mae', 'f_maxdiff', 'f_cos']
FMT = {'s2_omol': '%.6f', 'nfod': '%.4f', 'e_model_ev': '%.6f', 'e_omol_ev': '%.6f',
       'de_mev': '%.3f', 'fmax_model': '%.6f', 'fmax_omol': '%.6f', 'f_mae': '%.6f',
       'f_maxdiff': '%.6f', 'f_cos': '%.5f'}
out = os.path.join(REPO, 'results', 'model_sp_spintest.csv' if a.tag == 'spintest' else 'model_sp_ts_all.csv')
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in sorted(rows, key=lambda r: (r['model'], r['spin'], r['rxn'])):
        w.writerow([('' if r.get(c) is None else (FMT[c] % r[c] if c in FMT else r[c])) for c in COLS])
print('\n%s   %d Zeilen' % (os.path.relpath(out, REPO), len(rows)))
