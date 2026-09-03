"""Sind die Energie-Extremfaelle dieselben wie die Kraft-Extremfaelle?

Alles aus results/omol25_compare.csv und den vorhandenen ORCA-Outputs unter
orca_om25/. Keine Neuberechnung. Fehlende Groessen werden als 'fehlt'
ausgewiesen, nicht ersetzt.

results/energy_outlier_check.txt
"""
import csv
import io
import os

import numpy as np

RES = 'results'
R = list(csv.DictReader(open(os.path.join(RES, 'omol25_compare.csv'))))
MISS = 'fehlt'
LONG = {'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'esen': 'eSEN'}

# Schritt 4 woertlich aus dem Cluster-Lauf (_probe_outliers.sh, orca_om25/).
# Nur abgelesen, nicht gedeutet.
SCF = {
    ('rxn0894', 'esen'):  dict(term=1, cyc_ts=92, cyc_r=15,
                               stab='stable HF/KS wave function',
                               restart=0, s2_r=0.000000),
    ('rxn0894', 'uma-m'): dict(term=1, cyc_ts=80, cyc_r=15,
                               stab='stable HF/KS wave function',
                               restart=0, s2_r=-0.000000),
    ('rxn8837', 'uma-s'): dict(term=1, cyc_ts=95, cyc_r=14,
                               stab='stable HF/KS wave function',
                               restart=0, s2_r=0.000000),
}


def num(r, k):
    return float(r[k]) if r[k] != '' else None


rows = []
for r in R:
    bm, bd = num(r, 'barr_model'), num(r, 'barr_tzvpd')
    rm, rd = num(r, 'rxne_model'), num(r, 'rxne_tzvpd')
    rows.append({
        'rxn': r['rxn'], 'model': r['model'],
        'e_fwd': None if None in (bm, bd) else bm - bd,
        'e_rev': None if None in (bm, bd, rm, rd) else (bm - rm) - (bd - rd),
        'F_model': num(r, 'F_model'), 'F_dft': num(r, 'F_tzvpd'),
        'f_res': num(r, 'maxcomp_err'), 'f_mae': num(r, 'mae_force'),
        's2': num(r, 's2_ts_tzvpd')})

hasE = [d for d in rows if d['e_fwd'] is not None]
hasF = [d for d in rows if d['f_res'] is not None]
ordE = sorted(hasE, key=lambda d: -abs(d['e_fwd']))
ordF = sorted(hasF, key=lambda d: -d['f_res'])
rankE = {(d['rxn'], d['model']): i + 1 for i, d in enumerate(ordE)}
rankF = {(d['rxn'], d['model']): i + 1 for i, d in enumerate(ordF)}

out = io.StringIO()
W = out.write

W('IDENTITAET DER DREI ENERGIE-AUSREISSER AUS DER BARRIEREN-RESIDUENANALYSE\n')
W('=' * 104 + '\n')
W('Quelle:  results/omol25_compare.csv  (%d Zeilen; %d mit Energieresiduum, '
  '%d mit Kraftresiduum)\n' % (len(rows), len(hasE), len(hasF)))
W('         orca_om25/<rxn>_<Modell>/{ts_sp,r_sp}.out  fuer Abschnitt 4\n')
W('Niveau:  wB97M-V/def2-TZVPD, def2/J, RIJCOSX, TightSCF, DEFGRID3, '
  'Thresh 1e-12, TCut 1e-13, STABPerform.\n')
W('Alle Groessen an derselben, unrelaxierten Modellgeometrie. '
  'Es wurde nichts neu gerechnet.\n')
W('\nDefinitionen\n')
W('   dE_fwd = [E_MLIP(TS)-E_MLIP(R)] - [E_DFT(TS)-E_DFT(R)]      '
  'mit Vorzeichen\n')
W('   dE_rev = [E_MLIP(TS)-E_MLIP(P)] - [E_DFT(TS)-E_DFT(P)]      '
  'mit Vorzeichen\n')
W('   F_res  = max_i |F_Modell,i - F_DFT,i|   ueber alle 3N kartesischen '
  'Komponenten\n')
W('   F_model, F_dft = max_i |F_i| des jeweiligen Kraftfeldes am selben '
  'Punkt\n\n')

# ------------------------------------------------------------------ 1 + 2
W('1 UND 2   DIE DREI ZEILEN MIT |dE_fwd| > 0.5 eV, DANEBEN IHR KRAFT-BEFUND\n')
W('-' * 104 + '\n')
big = sorted([d for d in hasE if abs(d['e_fwd']) > 0.5],
             key=lambda d: -abs(d['e_fwd']))
W('%-9s %-7s %9s %9s %9s %9s %9s %9s %8s %6s %6s\n'
  % ('rxn', 'Modell', 'dE_fwd', 'dE_rev', 'F_model', 'F_dft', 'F_res',
     'F_MAE', '<S^2>', 'RgF', 'RgE'))
W('%-9s %-7s %9s %9s %9s %9s %9s %9s %8s %6s %6s\n'
  % ('', '', '[eV]', '[eV]', '[eV/A]', '[eV/A]', '[eV/A]', '[eV/A]', 'TS',
     '/%d' % len(hasF), '/%d' % len(hasE)))


def f(v, spec='%9.4f'):
    return (spec % v) if v is not None else ('%9s' % MISS)


for d in big:
    k = (d['rxn'], d['model'])
    W('%-9s %-7s %s %s %s %s %s %s %8s %6s %6s\n'
      % (d['rxn'], LONG[d['model']],
         f(d['e_fwd'], '%+9.4f'), f(d['e_rev'], '%+9.4f'),
         f(d['F_model']), f(d['F_dft']), f(d['f_res']), f(d['f_mae']),
         ('%8.4f' % d['s2']) if d['s2'] is not None else ('%8s' % MISS),
         rankF.get(k, MISS), rankE.get(k, MISS)))
W('\nAlle drei liegen im Kraftresiduum unter den ersten sechs von %d, '
  'aber nicht unter den ersten drei.\n' % len(hasF))
W('\n')

# ---------------------------------------------------------------------- 3
W('3   TOP-5 NACH KRAFTRESIDUUM UND TOP-5 NACH ENERGIERESIDUUM\n')
W('-' * 104 + '\n')
W('%-46s   %s\n' % ('nach F_res = max_i |dF|', 'nach |dE_fwd|'))
W('%-46s   %s\n' % ('-' * 46, '-' * 46))
for i in range(5):
    a, b = ordF[i], ordE[i]
    ka, kb = (a['rxn'], a['model']), (b['rxn'], b['model'])
    W('%d. %-9s %-6s %7.4f eV/A  (RgE %3s)   %d. %-9s %-6s %+8.4f eV  '
      '(RgF %3s)\n'
      % (i + 1, a['rxn'], LONG[a['model']], a['f_res'], rankE.get(ka, MISS),
         i + 1, b['rxn'], LONG[b['model']], b['e_fwd'], rankF.get(kb, MISS)))
sF = {(d['rxn'], d['model']) for d in ordF[:5]}
sE = {(d['rxn'], d['model']) for d in ordE[:5]}
inter = sorted(sF & sE)
W('\nSchnittmenge der beiden Top-5:  %d von 5  ->  %s\n'
  % (len(inter), ', '.join('%s/%s' % (a, LONG[b]) for a, b in inter)))
W('Nur in der Kraftliste:          %s\n'
  % ', '.join('%s/%s' % (a, LONG[b]) for a, b in sorted(sF - sE)))
W('Nur in der Energieliste:        %s\n'
  % ', '.join('%s/%s' % (a, LONG[b]) for a, b in sorted(sE - sF)))

both = [d for d in rows if d['e_fwd'] is not None and d['f_res'] is not None]
ae = np.array([abs(d['e_fwd']) for d in both])
af = np.array([d['f_res'] for d in both])
rk = lambda x: np.argsort(np.argsort(x)).astype(float)
W('\nSpearman-Rangkorrelation |dE_fwd| gegen F_res ueber alle %d Zeilen mit '
  'beiden Werten:  rho = %+.3f\n' % (len(both), np.corrcoef(rk(ae), rk(af))[0, 1]))
W('\n')

# ---------------------------------------------------------------------- 4
W('4   DFT-SINGLEPOINTS DER DREI ZEILEN, ABGELESEN AUS DEN ORCA-OUTPUTS\n')
W('-' * 104 + '\n')
W('%-9s %-7s %6s %8s %8s %-38s %8s\n'
  % ('rxn', 'Modell', 'TERM', 'SCF TS', 'SCF R', 'Stabilitaetsmeldung (ts_sp)',
     'Restart'))
W('%-9s %-7s %6s %8s %8s %-38s %8s\n'
  % ('', '', 'normal', 'Zyklen', 'Zyklen', '', 'auf UHF'))
for d in big:
    k = (d['rxn'], d['model'])
    s = SCF.get(k)
    if s is None:
        W('%-9s %-7s %s\n' % (d['rxn'], LONG[d['model']], MISS))
        continue
    W('%-9s %-7s %6s %8d %8d %-38s %8s\n'
      % (d['rxn'], LONG[d['model']], 'ja' if s['term'] else 'nein',
         s['cyc_ts'], s['cyc_r'], s['stab'], 'nein' if not s['restart'] else 'ja'))
W('\nWeiter abgelesen, ohne Deutung:\n')
W('   - In allen drei Faellen genau ein SCF-Block je Lauf; kein zweiter '
  'Durchlauf, kein STABRestart ausgeloest.\n')
W('   - Keine Zeile enthaelt SCF NOT CONVERGED, CONVERGENCE FAILURE oder '
  'SERIOUS PROBLEM.\n')
W('   - Die Zeile "Restarting incremental Fock matrix formation" steht in '
  'allen Outputs; das ist der\n     Neuaufbau der inkrementellen Fock-Matrix, '
  'nicht der STABPerform-Restart.\n')
W('   - <S^2> am TS: %s.  Am Edukt: %s.\n'
  % (', '.join('%s/%s %.3f' % (d['rxn'], LONG[d['model']], d['s2'])
               for d in big),
     ', '.join('%s/%s %.3f' % (d['rxn'], LONG[d['model']],
                               SCF[(d['rxn'], d['model'])]['s2_r'])
               for d in big)))
W('   - Zyklenzahl am TS 80 bis 95, am Edukt 14 bis 15. Der Median ueber alle '
  'Laeufe wurde nicht erhoben,\n     die Zahlen stehen hier ohne Vergleich.\n')
W('\n')

# ---------------------------------------------------------------------- 5
W('5   ANTWORT\n')
W('-' * 104 + '\n')
W('Energie- und Kraft-Extremfaelle ueberlappen teilweise.\n\n')
W('Belege: die drei Energie-Ausreisser stehen im Kraftresiduum auf Rang %s '
  'von %d, also alle unter den\n'
  % (', '.join(str(rankF[(d['rxn'], d['model'])]) for d in big), len(hasF)))
W('ersten sechs, aber die Kraft-Raenge 2 und 3 (%s) sind im Energieresiduum '
  'nur Rang %s.\n'
  % (', '.join('%s/%s' % (d['rxn'], LONG[d['model']]) for d in ordF[1:3]),
     ' und '.join(str(rankE.get((d['rxn'], d['model']), MISS))
                  for d in ordF[1:3])))
W('Die Top-5-Listen teilen sich %d von 5 Eintraegen. Ueber alle %d Zeilen '
  'betraegt die Rangkorrelation %+.2f.\n'
  % (len(inter), len(both), np.corrcoef(rk(ae), rk(af))[0, 1]))

p = os.path.join(RES, 'energy_outlier_check.txt')
io.open(p, 'w', encoding='utf-8').write(out.getvalue())
print(out.getvalue())
print('geschrieben:', p)
