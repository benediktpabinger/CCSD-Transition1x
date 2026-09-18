# -*- coding: utf-8 -*-
"""Prueft docs/chapter_omol25.tex gegen die validierten Tabellen.

Bricht ab, wenn eine Zahl im Kapiteltext nicht mehr zu
results/omol25_model_geoms.csv, results/neb_runs.csv,
results/barrier_spread.csv, results/paper_reactions.csv,
results/hinge_t1x.csv oder results/hinge_omol25.csv passt.
Gegenstueck zu check_methods_sec7.py fuer das Kapitel.

Gruppennamen im Text: closed-shell (<S^2> = 0), broken-symmetry (<S^2> > 0).

Lauf: python pipeline/check_results_section.py   (aus dem Repo-Wurzelverzeichnis)
"""
import csv
import io
import statistics as st

import numpy as np

M = list(csv.DictReader(io.open('results/omol25_model_geoms.csv', encoding='utf-8')))
N = {(r['rxn'], r['model']): r for r in
     csv.DictReader(io.open('results/neb_runs.csv', encoding='utf-8'))}
S = list(csv.DictReader(io.open('results/barrier_spread.csv', encoding='utf-8')))
P = list(csv.DictReader(io.open('results/paper_reactions.csv', encoding='utf-8')))
T1 = list(csv.DictReader(io.open('results/hinge_t1x.csv', encoding='utf-8')))
T2 = list(csv.DictReader(io.open('results/hinge_omol25.csv', encoding='utf-8')))
tex = io.open('docs/chapter_omol25.tex', encoding='utf-8').read()
flat = ' '.join(tex.split())
bad = []


def want(txt, why):
    if ' '.join(txt.split()) not in flat:
        bad.append('%s -- fehlt: %r' % (why, txt))


def forbid(txt, why):
    if txt in flat:
        bad.append('%s -- noch im Text: %r' % (why, txt))


med = lambda rows, c: st.median([float(r[c]) for r in rows])
stab = [r for r in M if r['unstable_ts'] == '0']
unst = [r for r in M if r['unstable_ts'] == '1']
MODELS = (('uma-s', 'UMA-S'), ('uma-m', 'UMA-M'), ('esen', 'eSEN'))
NST, NUN = 'closed-shell', 'broken-symmetry'

# ---------------------------------------------------- alte Gruppennamen
for old in ('RKS-stable', 'RKS stable', 'RKS-unstable', 'RKS unstable',
            'UKS stable', 'UKS-stable', 'fig:spread'):
    forbid(old, 'alter Name / toter Verweis')

# ---------------------------------------------------- Klassifikation
s2 = [abs(float(r['s2_ts'])) for r in M]
want('gives %d closed-shell and %d broken-symmetry structures, with no borderline case: the smallest non-zero value is $%.4f$'
     % (sum(1 for v in s2 if v == 0), sum(1 for v in s2 if v > 0),
        min(v for v in s2 if v > 0)), 'S2-Verteilung')
want('%d %s and %d %s structures' % (len(stab), NST, len(unst), NUN), 'Gruppen n')
# ---- MR-Kontrolle innerhalb der Straten (Discussion, Alternative explanations)
strat = {r['rxn']: r['stratum'] for r in P}
hcs = [r for r in M if strat[r['rxn']] == 'high' and r['unstable_ts'] == '0']
hbs = [r for r in M if strat[r['rxn']] == 'high' and r['unstable_ts'] == '1']
lcs = [r for r in M if strat[r['rxn']] == 'low' and r['unstable_ts'] == '0']
want('is %.3f~eV\\,\\AA$^{-1}$ in the low-MR tier and %.3f in the high-MR tier, and the force error %.3f against %.3f'
     % (med(lcs, 'f_dft_max'), med(hcs, 'f_dft_max'), med(lcs, 'f_err_mae'), med(hcs, 'f_err_mae')), 'MR-Kontrolle closed-shell')
want('%d of the %d model transition states are closed-shell and %d broken-symmetry; the low-MR tier holds %d'
     % (len(hcs), len(hcs) + len(hbs), len(hbs), len(lcs)), 'MR-Kontrolle n')
want('%.3f against %.3f in residual force and %.3f against %.3f in force error'
     % (med(hbs, 'f_dft_max'), med(hcs, 'f_dft_max'), med(hbs, 'f_err_mae'), med(hcs, 'f_err_mae')), 'MR-Kontrolle high')
t1h_bs = [T1r for T1r in T1 if strat[T1r['rxn']] == 'high' and T1r['group_local'] == 'unstable']
t1h_cs = [T1r for T1r in T1 if strat[T1r['rxn']] == 'high' and T1r['group_local'] == 'stable']
want('(%.2f against %.2f)' % (med(t1h_bs, 'f_bs'), med(t1h_cs, 'f_bs')), 'MR-Kontrolle Leiter')
grp = {r['rxn']: r['group_rxn'] for r in P}
want('which gives %d %s and %d %s reactions'
     % (sum(1 for g in grp.values() if g == 'unstable'), NUN,
        sum(1 for g in grp.values() if g == 'stable'), NST), 'Reaktionsgruppen')
strat = {}
for r in P:
    strat.setdefault(r['stratum'], []).append(r['group_rxn'] == 'unstable')
want('%d of the %d high-MR reactions are %s, %d of the %d mid-MR reactions, and none of the %d low-MR'
     % (sum(strat['high']), len(strat['high']), NUN,
        sum(strat['spread']), len(strat['spread']), len(strat['low'])),
     'Kreuztabelle Strata')
assert sum(strat['low']) == 0

# ---------------------------------------------------- Fig 1: Restkraefte
want('%.4f~eV\\,\\AA$^{-1}$ at the %s structures and %.4f at the %s ones, a factor of %.2f'
     % (med(stab, 'f_model_max'), NST, med(unst, 'f_model_max'), NUN,
        med(unst, 'f_model_max') / med(stab, 'f_model_max')), 'Modell gesamt')
want('%.4f~eV\\,\\AA$^{-1}$ at the %s structures and %.4f at the %s ones --- %.2f times larger'
     % (med(stab, 'f_dft_max'), NST, med(unst, 'f_dft_max'), NUN,
        med(unst, 'f_dft_max') / med(stab, 'f_dft_max')), 'DFT gesamt')
for key, lab in MODELS:
    s = [r for r in stab if r['model'] == key]
    u = [r for r in unst if r['model'] == key]
    want('%s %.3f against %.3f' % (lab, med(s, 'f_dft_max'), med(u, 'f_dft_max')),
         'DFT ' + lab)
    want('%s %.3f/%.3f, %.3f/%.3f' % (lab, med(s, 'f_model_max'), med(s, 'f_dft_max'),
                                      med(u, 'f_model_max'), med(u, 'f_dft_max')),
         'Caption Fig 1 ' + lab)
es, eu = ([r for r in g if r['model'] == 'esen'] for g in (stab, unst))
want('(%.3f against %.3f)' % (med(eu, 'f_model_max'), med(es, 'f_model_max')),
     'eSEN Modell niedriger')
ok = [r for r in M if N[(r['rxn'], r['model'])]['criterion_met'] == '1']
s2_ = [r for r in ok if r['unstable_ts'] == '0']
u2_ = [r for r in ok if r['unstable_ts'] == '1']
want('gives %.2f for the DFT separation and %.2f for the model separation'
     % (med(u2_, 'f_dft_max') / med(s2_, 'f_dft_max'),
        med(u2_, 'f_model_max') / med(s2_, 'f_model_max')), 'Robustheit 112')
want('unchanged for %s structures ($%.3f$~eV\\,\\AA$^{-1}$) and lowers it slightly for %s ones ($%.3f$ to $%.3f$)'
     % (NST, med(s2_, 'f_dft_max'), NUN, med(unst, 'f_dft_max'), med(u2_, 'f_dft_max')),
     'Fussnote 112')
want('moves from $%.2f$ to $%.2f$' % (med(unst, 'f_dft_max') / med(stab, 'f_dft_max'),
                                       med(u2_, 'f_dft_max') / med(s2_, 'f_dft_max')),
     'Fussnote Verhaeltnis')

# ---------------------------------------------------- Fig 2: Kraftfehler
parts = []
for key, lab in MODELS:
    s = [r for r in stab if r['model'] == key]
    u = [r for r in unst if r['model'] == key]
    parts.append((med(s, 'f_err_mae'), med(u, 'f_err_mae')))
want('broken-symmetry: %.4f against %.4f~eV\\,\\AA$^{-1}$ for UMA-S, %.4f against %.4f for UMA-M and %.4f against %.4f for eSEN, factors of %.1f, %.1f and %.1f'
     % (parts[0] + parts[1] + parts[2] + tuple(u / s for s, u in parts)), 'MAE je Modell')
want('median rises from %.4f to %.4f' % (med(stab, 'f_err_mae'), med(unst, 'f_err_mae')), 'MAE gepoolt')
ms, mu = med(stab, 'f_err_mae'), med(unst, 'f_err_mae')
a = sum(1 for r in unst if float(r['f_err_mae']) < ms)
b = sum(1 for r in stab if float(r['f_err_mae']) > mu)
lo = min(float(r['f_err_mae']) for r in unst)
hi = max(float(r['f_err_mae']) for r in stab)
ov = sum(1 for r in M if lo <= float(r['f_err_mae']) <= hi)
want('%d of the %d %s structures lie below the %s median' % (a, len(unst), NUN, NST),
     'Ueberlappung a')
want('%d of the %d %s structures above the %s one' % (b, len(stab), NST, NUN),
     'Ueberlappung b')
dep = [float(r['depth_ts_mev']) for r in unst]
want('spans %.1f to %.0f~meV' % (min(dep), max(dep)), 'Tiefenspanne')
from scipy.stats import spearmanr  # noqa: E402
rho = spearmanr(dep, [float(r['f_err_mae']) for r in unst]).correlation
want(r'Spearman $\rho = %.2f$ over the %d structures' % (rho, len(unst)), 'Spearman')

# ---------------------------------------------------- Fig 3A: Barrierenfehler
def errs(rows):
    return [abs(float(r['err_barr'])) * 1000 for r in rows if r['err_barr'] != '']
sm = [st.median(errs([r for r in stab if r['model'] == k])) for k, _ in MODELS]
um = [st.median(errs([r for r in unst if r['model'] == k])) for k, _ in MODELS]
want('%.1f, %.1f and %.1f~meV on the %s' % (tuple(sm) + (NST,)), 'Median closed-shell')
want('%.1f, %.1f and %.1f~meV on the %s structures' % (tuple(um) + (NUN,)),
     'Median broken-symmetry')
ratio = [st.mean(errs([r for r in unst if r['model'] == k])) /
         st.mean(errs([r for r in stab if r['model'] == k])) for k, _ in MODELS]
want('are %.0f, %.0f and %.0f times larger' % tuple(ratio), 'MAE-Verhaeltnis')
within = sum(1 for r in unst if r['err_barr'] != '' and abs(float(r['err_barr'])) < 0.043)
want('%d of the %d %s structures lie below 43~meV' % (within, len(unst), NUN),
     '49 von 53')
tops = {}
for key, lab in MODELS:
    u = [r for r in unst if r['model'] == key and r['err_barr'] != '']
    top = max(u, key=lambda r: abs(float(r['err_barr'])))
    tops[lab] = top['rxn']
    want('$%+.2f$~eV for %s' % (float(top['err_barr']), lab), 'Ausreisser-Wert ' + lab)
want('%s misses by' % tops['UMA-S'], 'Ausreisser UMA-S')
assert tops['UMA-M'] == tops['eSEN']
want('%s by' % tops['UMA-M'], 'Ausreisser UMA-M/eSEN')
want('the named reaction sets the mean of each %s group' % NUN, 'Caption Fig 3A')

# ---------------------------------------------------- Fig 3B: Spannweite
# Spannweite wie in der Figur aus den rohen Barrieren gerechnet: der
# closed-shell-Median ist 0.33499999 meV, in der CSV auf 0.335 gerundet;
# Text und Figur schreiben 0.33, die CSV-Rundung ergaebe 0.34.
import collections  # noqa: E402
by = collections.defaultdict(dict)
for r in M:
    by[r['rxn']][r['model']] = r
raw = {}
for k, v in by.items():
    b = [float(v[m]['barr_dft']) for m, _ in MODELS]
    raw[k] = (max(b) - min(b)) * 1000.0
for r in S:
    assert abs(raw[r['rxn']] - float(r['spread_mev'])) < 5e-4, r['rxn']
ss = [raw[r['rxn']] for r in S if r['group_rxn'] == 'stable']
su = [raw[r['rxn']] for r in S if r['group_rxn'] == 'unstable']
want('median spread is %.2f~meV at the %s reactions and %.2f~meV at the %s ones, a factor of %.0f; the means are %.0f and %.0f~meV'
     % (st.median(ss), NST, st.median(su), NUN, st.median(su) / st.median(ss),
        st.mean(ss), st.mean(su)), 'Spread Mediane/Mittel')
over_u = sorted(v for v in su if v > 43)
over_s = sorted((r['rxn'], float(r['spread_mev'])) for r in S
                if r['group_rxn'] == 'stable' and float(r['spread_mev']) > 43)
want('%s of the %d %s reactions exceed chemical accuracy, from %.0f~meV up to %.0f~meV, against %s of the %d %s ones, %s and %s'
     % ({5: 'Five'}[len(over_u)], len(su), NUN, over_u[0], over_u[-1],
        {2: 'two'}[len(over_s)], len(ss), NST,
        *[n for n, _ in sorted(over_s, key=lambda t: -t[1])]), 'ueber 43')
top3 = {r['rxn'] for r in sorted(S, key=lambda r: -float(r['spread_mev']))[:3]}
assert {tops['UMA-S'], tops['UMA-M']} <= top3, top3
want('%s and %s, are also among the three largest spreads' % (tops['UMA-S'], tops['UMA-M']),
     'Ausreisser in beiden Panels')
want('counts as %s if at least one of its three transition states' % NUN,
     'Caption Fig 3B')

# ---------------------------------------------------- Leiter
t1s = [r for r in T1 if r['group_local'] == 'stable']
t1u = [r for r in T1 if r['group_local'] == 'unstable']
# Die Faktoren sind, wie in hinge_tables.py (CORE) eingefroren, Mediane der
# Verhaeltnisse je Reaktion (Spalte ratio), nicht Verhaeltnisse der Mediane.
t2s = [r for r in T2 if r['group_local'] == 'stable']
t2u = [r for r in T2 if r['group_local'] == 'unstable']
want('restricted surface ($%.3f$ and $%.3f$)' % (med(t2s, 'f_rks'), med(t2u, 'f_rks')), 'Leiter nachopt RKS')
want('is still off, $%.2f$, a factor of $%.0f$'
     % (med(t2u, 'f_bs'), med(t2u, 'ratio')), 'Leiter nachopt BS')
want('%s (%d)     & $%.4f$ & $%.3f$ & $%.3f$' % (NST, len(t1s), med(t1s, 'f_ref'),
                                                 med(t1s, 'f_rks'), med(t1s, 'f_bs')),
     'Tabelle Zeile 1')
want('%s (%d)  & $%.4f$ & $%.3f$ & $%.3f$' % (NUN, len(t1u), med(t1u, 'f_ref'),
                                              med(t1u, 'f_rks'), med(t1u, 'f_bs')),
     'Tabelle Zeile 2')
want('(nine %s, three %s)' % (NST, NUN), 'Fussnote Nachoptimierung')
assert len(T2) == 33 and len(t2s) == 18 and len(t2u) == 15

# --- Versionstest ORCA 5.0.4 gegen die OMol25-Labels (ORCA 6.0.0) ----------
# results/omol25_label_compare.csv aus pipeline/omol25_label_compare.py.
V = list(csv.DictReader(io.open('results/omol25_label_compare.csv', encoding='utf-8')))
vc = [r for r in V if r['group'] == 'closed']
vb = [r for r in V if r['group'] == 'bs']
assert len(V) == 44 and len(vc) == 27 and len(vb) == 17, 'Versionstest: Zeilenzahl'
assert max(float(r['rmsd_A']) for r in V) <= 1e-6, 'Versionstest: Geometrie nicht identisch'
dE = [float(r['d_rks_mev']) for r in vc] + [float(r['d_bs_mev']) for r in vb]
assert max(abs(x) for x in dE) < 0.1, 'Versionstest: |dE| >= 0.1 meV'
assert -0.10 <= min(dE) and max(dE) <= -0.06, 'Versionstest: Offset ausserhalb -0.10..-0.06'
dF = [float(r['dF_rks_max']) for r in vc] + [float(r['dF_bs_max']) for r in vb]
assert max(dF) <= 0.02, 'Versionstest: dF > 0.02 eV/A'
assert all(r['omol_surface'] == 'gebrochen' and r['omol_unrestricted'] == '1' for r in vb), \
    'Versionstest: OMol25-Label nicht auf der gebrochenen Loesung'
assert max(abs(float(r['s2_ours']) - float(r['s2_omol'])) for r in vb) < 0.007, 'Versionstest: <S^2>'
dep = [float(r['depth_mev']) for r in vb]
assert round(min(dep)) == 16 and round(max(dep)) == 616, 'Versionstest: Bruchtiefe-Spanne'
want('44 of our 45 transition states are in the release', 'Versionstest Treffer')
want('agree to under $0.1$~meV in energy and $0.02$~eV', 'Versionstest dE/dF')
want('at all %d broken-symmetry geometries the OMol25 label sits on the' % len(vb), 'Versionstest BS')

# ---------------------------------------------------- NEB-Schritte
okN = [float(r['n_steps']) for r in N.values() if r['criterion_met'] == '1']
noN = [r for r in N.values() if r['criterion_met'] == '0']
lim = [r for r in noN if r['converged_marker'] == '1']
want('converged after a median of %d optimiser steps' % st.median(okN), 'NEB Schritte konvergiert')
okM = sum(1 for r in N.values() if r['converged_marker'] == '1')
want('reported success in %d of the %d searches' % (okM, len(N)), 'Optimierer-Erfolgsmarker')
want('%d at the step limit, after a median of %d steps, and %s aborted by the solver'
     % (len(lim), st.median([float(r['n_steps']) for r in lim]), {2: 'two'}[len(noN) - len(lim)]),
     'NEB Schritte nicht konvergiert')

# ---------------------------------------------------- Delta-Tiers (Kapitel 3) in der T1x-Stabilitaet
import json
D30 = json.load(io.open('results/delta_fixed_head/full_benchmark_results.json', encoding='utf-8'))['reactions']
tier = {r['rxn']: r['group'] for r in D30}
t1g = {r['rxn']: r['group'] for r in T1}
assert len(tier) == 30 and all(x in t1g for x in tier), 'Delta-Tiers: nicht alle 30 in hinge_t1x'
WORDS = {0: 'none', 1: 'one', 7: 'seven'}
bs = {t: sum(1 for x, g in tier.items() if g == t and t1g[x] == 'unstable') for t in ('high', 'mid', 'low')}
want('The 30 reactions of Chapter~\\ref{ch:delta} are all among the 45', 'Delta-Tiers Teilmenge')
want('the lowest solution is broken-symmetry for %s of the ten high-MR, %s of the ten mid-MR and %s of the ten low-MR reactions'
     % (WORDS[bs['high']], WORDS[bs['mid']], WORDS[bs['low']]), 'Delta-Tiers BS-Anteile')

print('VERIFY chapter_omol25.tex gegen die Tabellen')
for x in bad:
    print('  FEHL ' + x)
print('  %d Beanstandungen' % len(bad))
if bad:
    raise SystemExit(1)
