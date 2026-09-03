"""Eine Tabelle fuer alles, was auf OMol25-Niveau an den MODELLGEOMETRIEN
gemessen wurde. Nichts anderes kommt hinein.

Warum es diese Datei gibt: die Zahlen lagen bisher in omol25_compare.csv
(dort mit def2-TZVP-Spalten vermischt), in paper_rows_ext.csv (PySCF/def2-TZVP)
und in rks_sheet_tzvpd.json (vier Zeilen von Hand). Diese Tabelle fuehrt nur
die OMol25-Groessen und benennt fuer jede ihre Quelle.

NIVEAU, fuer jede einzelne Zahl in dieser Datei identisch
    ORCA 5.0.4
    ! wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
    %scf Thresh 1e-12 / TCut 1e-13 / MaxIter 300 end
    UKS-Laeufe zusaetzlich mit STABPerform true und
    STABRestartUHFifUnstable true; der RKS-Lauf bewusst ohne, weil dort die
    restringierte Loesung gewollt ist, auch wo sie nicht der Grundzustand ist.

GEOMETRIEN
    Immer die drei Strukturen, die das jeweilige MLIP selbst erzeugt hat:
    <modeldir>/<rxn>/{reactant,transition_state,product}.xyz, unrelaxiert.
    Keine DFT-Optimierung, keine Referenzgeometrie. Die OMol25-NEB unter
    orca_neb_omol25/ hat mit dieser Datei nichts zu tun.

QUELLEN
    orca_om25/<rxn>_<Modell>/ts_sp.out      UKS + STABPerform am Modell-TS
    orca_om25/<rxn>_<Modell>/r_sp.out       dasselbe am Modell-Edukt
    orca_om25/<rxn>_<Modell>/p_sp.out       dasselbe am Modell-Produkt
    orca_om25/<rxn>_<Modell>/ts_engrad.out  Gradient auf den Orbitalen von
                                            ts_sp (MORead), also auf der
                                            Flaeche, die STABPerform gewaehlt
                                            hat
    orca_rks_sheet/<rxn>_<Modell>/ts_rks.out
                                            RKS-Einzelpunkt am selben
                                            Modell-TS, Jobs 10767516 und
                                            10767531
    <modeldir>/<rxn>/*.xyz                  Energie und Kraefte des MLIP aus
                                            der Kommentarzeile bzw. den
                                            letzten drei Spalten

results/omol25_model_geoms.csv
"""
import csv
import glob
import os
import re

import numpy as np

H = '/home/energy/s242862'
OUT = f'{H}/results'
EVA = 51.42208                 # Eh/Bohr -> eV/A
HA_EV = 27.211386245988
S2_BREAK = 0.05                # <S^2> ist 0 oder > 0.058, dazwischen nichts
MD = {'UMA-S': 'uma_neb_results', 'UMA-M': 'uma_m_neb_results',
      'eSEN': 'esen_neb_results'}
SLUG = {'UMA-S': 'uma-s', 'UMA-M': 'uma-m', 'eSEN': 'esen'}

E_RE = re.compile(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)')
S2_RE = re.compile(r'Expectation value of <S\*\*2>\s*:\s*([-\d.]+)')
CYC_RE = re.compile(r'SCF CONVERGED AFTER\s+(\d+)\s+CYCLES')


def sp(p):
    """Energie [Ha], <S^2>, SCF-Zyklen aus einem Einzelpunkt-Output."""
    if not os.path.exists(p):
        return None, None, None
    t = open(p, errors='replace').read()
    if 'ORCA TERMINATED NORMALLY' not in t:
        return None, None, None
    e = E_RE.findall(t)
    if not e or float(e[-1]) == 0.0:
        return None, None, None
    s2 = S2_RE.findall(t)
    c = CYC_RE.findall(t)
    return (float(e[-1]), float(s2[-1]) if s2 else None,
            int(c[-1]) if c else None)


def gradient(p):
    """dE/dx [eV/A], ganzer Vektor. Die Kraft ist minus davon."""
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
    return np.array(G) * EVA if G else None


def model_energy(p):
    """ASE schreibt die Energie in eV in die Kommentarzeile."""
    if not os.path.exists(p):
        return None
    m = re.search(r'\benergy=(-?[\d.eE+]+)', open(p, errors='replace').read(4000))
    return float(m.group(1)) if m else None


def model_forces(p):
    """Kraftvektoren des MLIP: die letzten drei Spalten der extxyz."""
    if not os.path.exists(p):
        return None
    L = open(p, errors='replace').read().split('\n')
    n = int(L[0].split()[0])
    if 'forces' not in L[1]:
        return None
    F = []
    for line in L[2:2 + n]:
        f = line.split()
        if len(f) < 7:
            return None
        F.append([float(x) for x in f[4:7]])
    return np.array(F) if len(F) == n else None


rows, gaps = [], []
for d in sorted(glob.glob(f'{H}/orca_om25/rxn*')):
    rx, mm = os.path.basename(d).rsplit('_', 1)
    if mm not in MD:
        continue
    md = f'{H}/{MD[mm]}/{rx}'

    e_ts, s2_ts, cyc_ts = sp(f'{d}/ts_sp.out')
    e_r, s2_r, _ = sp(f'{d}/r_sp.out')
    e_p, s2_p, _ = sp(f'{d}/p_sp.out')
    e_rks, _, cyc_rks = sp(f'{H}/orca_rks_sheet/{rx}_{mm}/ts_rks.out')

    G = gradient(f'{d}/ts_engrad.out')
    Fm = model_forces(f'{md}/transition_state.xyz')
    em = {k: model_energy(f'{md}/{f}.xyz') for k, f in
          (('R', 'reactant'), ('TS', 'transition_state'), ('P', 'product'))}

    unst = None if s2_ts is None else int(abs(s2_ts) > S2_BREAK)

    # Brechungstiefe: E_RKS - E_BS am selben Punkt, in meV.
    # Bei den stabilen Zeilen wurde kein RKS-Lauf gestartet: dort hat
    # STABPerform bestaetigt, dass es keine zweite Loesung gibt, die Tiefe
    # ist 0 per Konstruktion. Die Herkunft steht in depth_src.
    if e_rks is not None and e_ts is not None:
        depth, dsrc = (e_rks - e_ts) * HA_EV * 1000.0, 'rks_sp'
    elif unst == 0:
        depth, dsrc = 0.0, 'stabperform_stable'
    else:
        depth, dsrc = None, ''
        if unst == 1:
            gaps.append('%s %s: instabil, aber kein ts_rks.out' % (rx, mm))

    r = {
        'rxn': rx, 'model': SLUG[mm],
        'e_r_uks_ha': e_r, 'e_ts_uks_ha': e_ts, 'e_p_uks_ha': e_p,
        'e_ts_rks_ha': e_rks,
        's2_r': s2_r, 's2_ts': s2_ts, 's2_p': s2_p,
        'unstable_ts': unst, 'depth_ts_mev': depth, 'depth_src': dsrc,
        'f_model_max': None if Fm is None else float(np.abs(Fm).max()),
        # groesste Kraftnorm je Atom -- das ist ASE fmax, das Abbruch-
        # kriterium der NEB. Steht hier, damit das Verhaeltnis Komponente
        # zu Norm nachpruefbar ist, ohne die xyz erneut zu lesen.
        'f_model_norm_max': (None if Fm is None else
                             float(np.linalg.norm(Fm, axis=1).max())),
        'f_dft_max': None if G is None else float(np.abs(G).max()),
        'f_err_max': None, 'f_err_mae': None,
        'barr_model': None, 'barr_dft': None, 'barr_rks': None,
        'err_barr': None,
        'rxne_model': None, 'rxne_dft': None, 'err_rxne': None,
        'scf_cyc_ts_uks': cyc_ts, 'scf_cyc_ts_rks': cyc_rks,
    }
    if G is not None and Fm is not None and len(Fm) == len(G):
        dF = Fm - (-G)                       # Modellkraft minus DFT-Kraft
        r['f_err_max'] = float(np.abs(dF).max())
        r['f_err_mae'] = float(np.abs(dF).mean())
    if em['TS'] is not None and em['R'] is not None:
        r['barr_model'] = em['TS'] - em['R']
    if e_ts is not None and e_r is not None:
        r['barr_dft'] = (e_ts - e_r) * HA_EV
    if e_rks is not None and e_r is not None:
        r['barr_rks'] = (e_rks - e_r) * HA_EV
    if r['barr_model'] is not None and r['barr_dft'] is not None:
        r['err_barr'] = r['barr_model'] - r['barr_dft']
    if em['P'] is not None and em['R'] is not None:
        r['rxne_model'] = em['P'] - em['R']
    if e_p is not None and e_r is not None:
        r['rxne_dft'] = (e_p - e_r) * HA_EV
    if r['rxne_model'] is not None and r['rxne_dft'] is not None:
        r['err_rxne'] = r['rxne_model'] - r['rxne_dft']
    rows.append(r)

COLS = ['rxn', 'model',
        'e_r_uks_ha', 'e_ts_uks_ha', 'e_p_uks_ha', 'e_ts_rks_ha',
        's2_r', 's2_ts', 's2_p',
        'unstable_ts', 'depth_ts_mev', 'depth_src',
        'f_model_max', 'f_model_norm_max', 'f_dft_max', 'f_err_max',
        'f_err_mae',
        'barr_model', 'barr_dft', 'barr_rks', 'err_barr',
        'rxne_model', 'rxne_dft', 'err_rxne',
        'scf_cyc_ts_uks', 'scf_cyc_ts_rks']
FMT = {'f_model_norm_max': '%.6f', 'e_r_uks_ha': '%.9f', 'e_ts_uks_ha': '%.9f', 'e_p_uks_ha': '%.9f',
       'e_ts_rks_ha': '%.9f', 's2_r': '%.6f', 's2_ts': '%.6f', 's2_p': '%.6f',
       'depth_ts_mev': '%.1f',
       'f_model_max': '%.6f', 'f_dft_max': '%.6f', 'f_err_max': '%.6f',
       'f_err_mae': '%.6f',
       'barr_model': '%.6f', 'barr_dft': '%.6f', 'barr_rks': '%.6f',
       'err_barr': '%.6f', 'rxne_model': '%.6f', 'rxne_dft': '%.6f',
       'err_rxne': '%.6f'}

os.makedirs(OUT, exist_ok=True)
rows.sort(key=lambda r: (r['rxn'], r['model']))
with open(f'{OUT}/omol25_model_geoms.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(COLS)
    for r in rows:
        w.writerow(['' if r[c] is None else
                    (FMT[c] % r[c] if c in FMT else r[c]) for c in COLS])

# ------------------------------------------------------------ Pruefungen
fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FEHL ') + msg)
    if not ok:
        fails.append(msg)


def col(k):
    return np.array([np.nan if r[k] is None else r[k] for r in rows], float)


print('OMOL25 AN DEN MODELLGEOMETRIEN')
print('=' * 78)
print('%d Zeilen geschrieben' % len(rows))
print()
print('Vollstaendigkeit')
for c in COLS[2:]:
    if c == 'depth_src':
        continue
    n = sum(1 for r in rows if r[c] is None)
    print('  %-16s fehlend %3d' % (c, n))
print()
print('Pruefungen')
check(len(rows) == 135, '135 Zeilen (%d)' % len(rows))
u = np.array([r['unstable_ts'] for r in rows])
check(u.sum() == 53, 'instabile Zeilen 53 (%d)' % int(u.sum()))
s2 = np.abs(col('s2_ts'))
lo = s2[s2 > 0].min()
check(lo > S2_BREAK,
      'die Schwelle %.2f liegt in einer leeren Zone: kleinster Wert ueber '
      'null ist %.6f' % (S2_BREAK, lo))
d = col('depth_ts_mev')
check(not np.isnan(d).any(), 'Tiefe fuer alle Zeilen belegt (%d fehlen)'
      % int(np.isnan(d).sum()))
check(np.abs(d[u == 0]).max() < 1.0,
      'Tiefe der stabilen Zeilen unter 1 meV, groesste %.4f'
      % np.abs(d[u == 0]).max())
check((d[u == 1] > 0).all(), 'Tiefe der instabilen Zeilen durchweg > 0')
src = [r['depth_src'] for r in rows]
# 53 instabile plus rxn7060/eSEN, das aus der Sheet-Pruefung einen RKS-Lauf
# hat, obwohl es stabil ist. Dessen Tiefe ist die Nullprobe der ganzen
# Konstruktion: gemessen -0.0008 meV, wo 0 stehen muss.
check(src.count('rks_sp') == 54,
      'Tiefe aus einem RKS-Lauf: 54 (%d)' % src.count('rks_sp'))
check(src.count('stabperform_stable') == 81,
      'Tiefe 0 laut STABPerform: 81 (%d)' % src.count('stabperform_stable'))
nul = [r for r in rows if r['unstable_ts'] == 0 and r['depth_src'] == 'rks_sp']
check(len(nul) == 1 and abs(nul[0]['depth_ts_mev']) < 0.01,
      'Nullprobe %s/%s: RKS gemessen, Tiefe %.4f meV'
      % (nul[0]['rxn'], nul[0]['model'], nul[0]['depth_ts_mev'])
      if nul else 'Nullprobe fehlt')

# Gegenprobe gegen die alte Tabelle: die gemeinsamen Spalten muessen gleich sein
old = {(r['rxn'], r['model']): r
       for r in csv.DictReader(open(f'{OUT}/omol25_compare.csv'))}
worst = 0.0
for r in rows:
    o = old.get((r['rxn'], r['model']))
    if not o:
        continue
    for a, b in (('f_dft_max', 'F_tzvpd'), ('barr_model', 'barr_model'),
                 ('barr_dft', 'barr_tzvpd'), ('f_err_mae', 'mae_force')):
        if r[a] is None or o[b] == '':
            continue
        worst = max(worst, abs(r[a] - float(o[b])))
check(worst < 1e-3,
      'stimmt mit omol25_compare.csv ueberein, groesste Abweichung %.2e' % worst)

if gaps:
    print()
    print('Luecken')
    for g in gaps:
        print('   ', g)
print()
print('geschrieben: results/omol25_model_geoms.csv')
if fails:
    raise SystemExit('ABBRUCH: %d Pruefung(en) fehlgeschlagen' % len(fails))
