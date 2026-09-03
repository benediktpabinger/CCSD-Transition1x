"""Haengen die Aussagen an den gewaehlten Schwellen?

Lokal ausfuehrbar.  Liest results/paper_rows_ext.csv, results/hinge_rows.csv
und results/saddle_residuals.csv und variiert jede Schwelle, die nicht
mathematisch festliegt.

results/threshold_sensitivity.txt
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
CINEB = 0.05


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def col(rows, k):
    return np.array([float(r[k]) if r[k] != '' else np.nan for r in rows])


def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if not y.any() or y.all():
        return np.nan
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    _, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    t = np.zeros(len(c))
    np.add.at(t, inv, r)
    r = (t / c)[inv]
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def spearman(a, b):
    def rk(x):
        x = np.asarray(x, float)
        o = np.argsort(x)
        r = np.empty(len(x), float)
        r[o] = np.arange(1, len(x) + 1)
        _, inv, c = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(c))
        np.add.at(s, inv, r)
        return (s / c)[inv]
    ra, rb = rk(a), rk(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


rows = load('paper_rows_ext.csv')
fm, fd = col(rows, 'F_model'), col(rows, 'F_dft')
lam, nfd, mce = col(rows, 'lambda_min'), col(rows, 'nfod'), col(rows, 'maxcomp_err')
uns = col(rows, 'unstable').astype(bool)
hinge = load('hinge_rows.csv')
frks = np.array([float(r['F_rks']) for r in hinge])
fbs = np.array([float(r['F_bs']) for r in hinge])
sr = [r for r in load('saddle_residuals.csv') if r['gueltig'] == '1']
res = np.array([float(r['maxgrad_evang']) for r in sr])

O = []
w = O.append
w('EMPFINDLICHKEIT DER SCHWELLEN')
w('=' * 76)
w('')
w('1  KALIBRIERUNG DER STUFE-1-SCHWELLE')
w('')
w('Restgradient konvergierter TS-Optimierungen dieser Arbeit, alle drei')
w('Startpunkte am Zielniveau (results/saddle_residuals.csv):')
w('')
w('   n = %d   Median %.4f   Spanne %.4f bis %.4f eV/A'
  % (len(res), np.median(res), res.min(), res.max()))
w('   0.15 liegt %.0f-fach ueber dem Median, %.1f-fach ueber dem'
  % (0.15 / np.median(res), 0.15 / res.max()))
w('   unguenstigsten Fall.')
w('')
w('2  STUFE-1-SCHWELLE VARIIERT   (Standard 0.15 eV/A)')
w('')
w('%7s %9s %8s %8s %8s %12s' % ('cut', 'Positive', 'AUC lam', 'AUC bin',
                                'AUC FOD', 'stille Ausf.'))
w('-' * 60)
for c in (0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50):
    y = fd >= c
    w('%7.2f %9d %8.3f %8.3f %8.3f %8d/%d'
      % (c, int(y.sum()), auc(-lam, y), auc(uns.astype(float), y), auc(nfd, y),
         int(((fm < CINEB) & y).sum()), int((fm < CINEB).sum())))
w('')
w('   AUC lam bleibt zwischen %.3f und %.3f -- nie in der Naehe von 0.5.'
  % (min(auc(-lam, fd >= c) for c in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)),
     max(auc(-lam, fd >= c) for c in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5))))
w('   ABER: unter 0.10 ist N_FOD gleich gut oder besser. Der Vorsprung der')
w('   Instabilitaetsanalyse existiert erst ab cut >= 0.10.')
w('')
w('3  NEB-ABBRUCHKRITERIUM VARIIERT   (Standard 0.05 eV/A, Stufe 1 fest 0.15)')
w('')
y = fd >= 0.15
for c in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
    m = fm < c
    w('   Modell meldet < %.2f:  %3d Zeilen, davon %2d keine Stationaerpunkte'
      ' = %2.0f %%' % (c, int(m.sum()), int((m & y).sum()),
                       100 * (m & y).sum() / max(int(m.sum()), 1)))
w('')
w('   Die Rate bleibt 13 bis 21 %. Der stille Ausfall haengt nicht an der')
w('   Wahl des Abbruchkriteriums.')
w('')
w('4  GRENZFAELLE AN DER STUFE-1-SCHWELLE')
w('')
near = (fd > 0.12) & (fd < 0.18)
w('   %d von %d Zeilen liegen innerhalb von +/-0.03 um 0.15.' % (near.sum(), len(rows)))
w('   Fuer sie ist das Urteil fragil, Beispiel:')
rx = np.array([r['rxn'] for r in rows])
md = np.array([r['model'] for r in rows])
for i in np.flatnonzero(near & (rx == 'rxn4513')):
    w('      %-9s %-6s F_DFT %.3f  ->  %s'
      % (rx[i], md[i], fd[i], 'AUSFALL' if fd[i] >= 0.15 else 'ok'))
w('')
w('5  HINGE-AUSSAGE VARIIERT')
w('')
w('%7s %16s %16s' % ('cut', 'stationaer RKS', 'stationaer BS'))
w('-' * 42)
for c in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
    w('%7.2f %12d/19 %15d/19' % (c, int((frks < c).sum()), int((fbs < c).sum())))
w('')
w('   "0 von 19" gilt bis %.3f eV/A, dem kleinsten F_bs (%s).'
  % (fbs.min(), hinge[int(np.argmin(fbs))]['rxn']))
w('   Der Faktor zwischen den Spalten (%.0fx bis %.0fx) ist schwellenfrei.'
  % ((fbs / frks).min(), (fbs / frks).max()))
w('')
w('6  WAS OHNE JEDE SCHWELLE STEHT')
w('')
w('   Median F_Modell    stabil %.4f   instabil %.4f'
  % (np.median(fm[~uns]), np.median(fm[uns])))
w('   Median F_DFT       stabil %.4f   instabil %.4f'
  % (np.median(fd[~uns]), np.median(fd[uns])))
w('   Spearman(-lambda_min, F_DFT)      %+.3f' % spearman(-lam, fd))
w('   Spearman(-lambda_min, maxcomp)    %+.3f' % spearman(-lam, mce))
w('   Median F_bs ueber die 19          %.3f eV/A' % np.median(fbs))
w('')
w('   Diese fuenf Groessen brauchen keinen Cutoff und tragen den Kern.')
w('')
w('7  DIE EINZIGE NICHT GEWAEHLTE SCHWELLE')
w('')
w('   lambda_min < 0 ist die exakte Grenze zwischen "die restringierte')
w('   Loesung ist ein Minimum im Orbitalraum" und "es existiert eine')
w('   tiefere". Kein Parameter, kein Spielraum.')
w('')
w('Erzeugt von pipeline/threshold_sensitivity.py')

txt = '\n'.join(O)
os.makedirs(RES, exist_ok=True)
open(os.path.join(RES, 'threshold_sensitivity.txt'), 'w',
     encoding='utf-8').write(txt + '\n')
print(txt)
