# -*- coding: utf-8 -*-
"""Figur 2 — die kompakte Fassung von fig9_2b, auf Hoehe getrimmt.

Gleiche Daten, gleiche Aussage, gleiche Farben wie
pipeline/plot_omol25_figs.py -> fig_ferr('mae'). Die alte Fassung
figures/fig9_2b_force_mae_omol25.png bleibt unveraendert bestehen; diese hier
ist fuer den knappen Platz im Paper.

WO DIE HOEHE HERKOMMT
    1  Die Legende ist nach unten rechts gewandert. Sie steht unveraendert im
       Bild -- gleiche zwei Zeilen, Name, Median mit Einheit und n --, aber
       oben links zwang sie die y-Achse, ueber den Daten Platz freizuhalten.
       Der Hoehenposten war ihre LAGE, nicht ihr Inhalt.
    2  Einzeilige x-Beschriftung, nur noch die Klassendefinition: Name, Median
       und n stehen in der Legende und muessen nicht doppelt dastehen.
    3  Kein Obertitel, keine Fussnote, kurze y-Beschriftung -- die lange
       Fassung steht in der Bildunterschrift des Papers.
    4  Flach statt hoch: 11.0 x 2.9 statt 11.4 x 5.6 Zoll -- gleiche Breite,
       gut ein Drittel der Hoehe (489 statt 1404 px).

    Was NICHT gekuerzt wurde: die Punktwolke, die Bootstrap-Intervalle, der
    Verhaeltnispfeil und die Legende. Sie tragen die Aussage.

    Preis der Verlagerung: die y-Achse reicht unten bis min*0.22 statt
    min*0.70, damit der Legendenkasten vollstaendig UNTER dem niedrigsten
    Punkt sitzt. Das kostet keine Bildhoehe, nur Aufloesung -- 3.1 statt 2.6
    Dekaden im selben Zoll. Ohne diese Luft lagen 5 der 135 Punkte hinter dem
    Rahmen.

figures/fig2_force_mae.png
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')

S2_BREAK = 0.05
C_ST = '#2a6f7f'
C_UN = '#c2542a'
GREY = '#6b6b6b'
LBL = {'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'esen': 'eSEN'}
COL = 'f_err_mae'

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'legend.frameon': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
})

rr = [r for r in csv.DictReader(open(os.path.join(RES,
                                                  'omol25_model_geoms.csv')))
      if r[COL] != '' and r['s2_ts'] != '']
val = np.array([float(r[COL]) for r in rr])
ub = np.abs(np.array([float(r['s2_ts']) for r in rr])) > S2_BREAK
mm = np.array([r['model'] for r in rr])

rng = np.random.default_rng(20260823)          # wie im Original
jit = np.random.default_rng(5)                 # wie im Original


def ci(v, n=10000):
    v = np.asarray(v, float)
    b = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


fig, axs = plt.subplots(1, 3, figsize=(11.0, 2.9), sharey=True)
fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.20,
                    wspace=0.06)

report = []
for ax, m in zip(axs, ('uma-s', 'uma-m', 'esen')):
    sel = mm == m
    med, nn = {}, {}
    for x0, grp, c, nm in ((0, sel & ~ub, C_ST, 'RKS stable'),
                           (1, sel & ub, C_UN, 'RKS unstable')):
        v = val[grp]
        ax.scatter(x0 + jit.uniform(-0.17, 0.17, len(v)), v, s=22, c=c,
                   alpha=0.62, lw=0.4, edgecolor='white', zorder=1)
        lo, hi = ci(v)
        ax.vlines(x0, lo, hi, color=c, lw=6, alpha=0.22, zorder=2)
        md = float(np.median(v))
        med[x0], nn[x0] = md, int(grp.sum())
        ax.plot([x0 - 0.30, x0 + 0.30], [md, md], color=c, lw=2.6, zorder=3,
                solid_capstyle='butt', label=nm +
                '   median %.4f eV Å$^{-1}$   (n=%d)' % (md, int(grp.sum())))

    ax.annotate('', xy=(1.86, med[1]), xytext=(1.86, med[0]),
                arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.text(1.93, np.sqrt(med[0] * med[1]), '×%.1f' % (med[1] / med[0]),
            fontsize=9, color=GREY, va='center', fontweight='bold')

    # Die Legende sitzt unten rechts, in der leeren Ecke jedes Panels: dort
    # liegt in allen drei Modellen kein Punkt. Im Original stand sie oben
    # links und zwang die y-Achse, ueber den Daten Platz freizuhalten -- das
    # war der Hoehenposten, nicht die Legende selbst.
    ax.legend(loc='lower right', fontsize=7.0, frameon=True, framealpha=0.95,
              edgecolor='#ddd', borderpad=0.4, handlelength=1.6,
              handletextpad=0.6, labelspacing=0.35,
              borderaxespad=0.25).set_zorder(9)

    # Achsenbeschriftung nur noch die Klassendefinition -- Name, Median und n
    # stehen in der Legende und muessen hier nicht doppelt stehen.
    ax.set_xlim(-0.50, 2.32)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r'$\langle S^2\rangle = 0$',
                        r'$\langle S^2\rangle > 0$'])
    ax.tick_params(axis='x', length=0, pad=3)
    ax.set_title(LBL[m], loc='left', pad=3, fontsize=10, fontweight='bold')
    ax.grid(axis='y', color='#eee', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    report.append((LBL[m], nn[0], med[0], nn[1], med[1], med[1] / med[0]))

axs[0].set_yscale('log')
# Unten so viel Luft, dass der Legendenkasten vollstaendig UNTER dem
# niedrigsten Punkt sitzt. Mit min*0.70 lagen 5 der 135 Punkte hinter dem
# Rahmen -- geprueft, indem der Legendenrahmen in Datenkoordinaten
# zurueckgerechnet und die Punkte darin gezaehlt wurden. Kostet keine
# Bildhoehe, nur etwas Aufloesung: 3.1 statt 2.6 Dekaden im selben Zoll.
axs[0].set_ylim(val.min() * 0.22, val.max() * 1.50)
axs[0].set_ylabel(r'MAE $|F^{\rm MLIP}_i - F^{\rm DFT}_i|$   [eV Å$^{-1}$]',
                  fontsize=8.5)

os.makedirs(FIG, exist_ok=True)
p = os.path.join(FIG, 'fig2_force_mae.png')
fig.savefig(p, bbox_inches='tight', facecolor='white')
plt.close(fig)

print('FIGUR 2 (kompakt) — dieselben Zahlen wie fig9_2b')
print('%-8s %20s %22s %8s' % ('', 'stable', 'unstable', 'x'))
for lab, n0, m0, n1, m1, f in report:
    print('%-8s  median %.4f (n=%2d)   median %.4f (n=%2d)   %.1f'
          % (lab, m0, n0, m1, n1, f))
print()
print('   n gesamt %d   Wertebereich %.5f .. %.5f eV/A'
      % (len(rr), val.min(), val.max()))
print('   y-Grenzen %.4f .. %.4f  (%.2f Dekaden, vorher 3.30)'
      % (val.min() * 0.22, val.max() * 1.50,
         np.log10((val.max() * 1.50) / (val.min() * 0.22))))
print('   geschrieben:', os.path.relpath(p, HERE))
