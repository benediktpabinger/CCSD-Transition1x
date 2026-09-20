# -*- coding: utf-8 -*-
"""Die drei Paperfiguren, fertig gerendert, ohne Handzuschnitt.

    Pictures/Fig1_v2.png         Restkraft Modell gegen DFT     (Results Fig 1)
                                 (Fig1.png ist der Stand ohne offene Marker)
    Pictures/fig2_force_mae.png  Kraftfehler MAE                (Results Fig 2)
    Pictures/fig3_v2.png         Barrierenfehler A + Spread B   (Results Fig 3)
                                 (fig3.png ist der Stand ohne offene Kreise)

Gleiche Daten, gleiche Rechnung, gleiche Farben und Seeds wie die Vorlagen
(fig_silent_v2 in plot_omol25_figs.py, plot_fig2.py, plot_fig3.py). Was sich
gegenueber den Vorlagen aendert, ist nur Beschriftung:

    * Gruppennamen  "RKS stable" -> "closed-shell",
                    "RKS unstable" -> "broken-symmetry"
      (Klassifikation nach <S^2> der Einzelpunktrechnung: 0 bzw. > 0)
    * Fig 1: kein Obertitel, kein Fusstext (steht in der Caption)
    * Fig 3A: keine Legende; n steht an den zweizeiligen Ticks;
              der Verhaeltnispfeil ist einmal als "MAE ratio" beschriftet
    * Fig 3B: Ticks mit den neuen Gruppennamen

Schreibt nur nach Pictures/. figures/ bleibt unberuehrt.

Lauf: python pipeline/plot_paper_figs_v2.py   (aus dem Repo-Wurzelverzeichnis)
"""
import collections
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, 'Pictures')

S2_BREAK = 0.05
C_ST = '#2a6f7f'          # closed-shell
C_UN = '#c2542a'          # broken-symmetry
GREY = '#6b6b6b'
GREEN = '#3b7d3b'
LBL = {'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'esen': 'eSEN'}
MODELS = ('uma-s', 'uma-m', 'esen')

NAME_ST = 'closed-shell'
NAME_UN = 'broken-symmetry'
S2_ST = r'$\langle S^2\rangle = 0$'
S2_UN = r'$\langle S^2\rangle > 0$'

BASE_RC = {
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9,
    'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
}

ROWS = list(csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv'),
                                encoding='utf-8')))

# Suchen, deren Band das Kriterium f_max < 0.05 nicht erreicht hat (23 von 135).
# Fig 3 zeichnet sie als offene Kreise; gerechnet wird weiter ueber alle.
UNCONV = {(r['rxn'], r['model'])
          for r in csv.DictReader(open(os.path.join(RES, 'neb_runs.csv'),
                                       encoding='utf-8'))
          if r['criterion_met'] == '0'}


def _scatter_open(ax, x, y, open_, c, s, alpha, zorder):
    """Punkte wie bisher; die mit open_ markierten als offene Kreise."""
    ax.scatter(x[~open_], y[~open_], s=s, c=c, alpha=alpha, lw=0.5,
               edgecolor='white', zorder=zorder)
    ax.scatter(x[open_], y[open_], s=s, facecolors='none', edgecolors=c,
               lw=1.3, alpha=0.95, zorder=zorder)


def _key_open(ax, label, **kw):
    """Schluessel fuer die offenen Kreise, einmal pro Panel."""
    key = [Line2D([], [], marker='o', ls='', ms=6.5, mfc='none', mec=GREY,
                  mew=1.3, label=label)]
    ax.legend(handles=key, fontsize=7.4, frameon=True, framealpha=0.96,
              edgecolor='#ccc', handlelength=1.0, borderpad=0.6, **kw)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('   geschrieben:', os.path.relpath(p, HERE))
    return p


# ------------------------------------------------------------------ Fig 1
def fig1():
    """fig_silent_v2 ohne Obertitel und Fusstext, neue Gruppennamen."""
    plt.rcParams.update(BASE_RC)
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.18,
                         'grid.linewidth': 0.6})
    rows = [r for r in ROWS if r['f_dft_max'] != '' and r['s2_ts'] != '']
    fm = np.array([float(r['f_model_max']) for r in rows])
    fd = np.array([float(r['f_dft_max']) for r in rows])
    s2 = np.abs(np.array([float(r['s2_ts']) for r in rows]))
    mdl = np.array([r['model'] for r in rows])
    brk = s2 > S2_BREAK
    unc = np.array([(r['rxn'], r['model']) in UNCONV for r in rows])

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 5.2), sharex=True, sharey=True)
    lo = min(fm.min(), fd.min()) * 0.55
    hi = max(fm.max(), fd.max()) * 2.2
    med, drawn = [], 0

    for ax, m in zip(axs, MODELS):
        sel = mdl == m
        s, u = sel & ~brk, sel & brk
        ax.plot([lo, hi], [lo, hi], color='#444', lw=1.2, ls='--', zorder=2,
                label='MLIP = DFT')
        # Band nicht konvergiert: gleiche Form, offen; n zaehlt weiter alle
        ax.scatter(fm[s & ~unc], fd[s & ~unc], s=34, c=C_ST, alpha=0.85, lw=0,
                   zorder=4,
                   label='%s,  %s   (n=%d)' % (NAME_ST, S2_ST, s.sum()))
        ax.scatter(fm[u & ~unc], fd[u & ~unc], s=40, c=C_UN, alpha=0.85, lw=0,
                   marker='D', zorder=4,
                   label='%s,  %s   (n=%d)' % (NAME_UN, S2_UN, u.sum()))
        ax.scatter(fm[s & unc], fd[s & unc], s=34, facecolors='none',
                   edgecolors=C_ST, lw=1.3, alpha=0.95, zorder=4)
        ax.scatter(fm[u & unc], fd[u & unc], s=40, facecolors='none',
                   edgecolors=C_UN, lw=1.3, alpha=0.95, marker='D', zorder=4)
        drawn += int(s.sum() + u.sum())
        for mm, c in ((s, C_ST), (u, C_UN)):
            x0, y0 = np.median(fm[mm]), np.median(fd[mm])
            ax.plot([x0], [y0], marker='+', ms=22, mec='white', mew=4.0,
                    ls='none', zorder=7)
            ax.plot([x0], [y0], marker='+', ms=22, mec=c, mew=1.8,
                    ls='none', zorder=8)
        med.append((LBL[m], np.median(fm[s]), np.median(fd[s]),
                    np.median(fm[u]), np.median(fd[u])))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(LBL[m], loc='left', pad=8)
        ax.set_xlabel(r'$\max_i |F_i^{\,\mathrm{MLIP}}|$  at the MLIP transition'
                      '\n'
                      r'state it produced   [eV Å$^{-1}$]')
        h, l = ax.get_legend_handles_labels()
        h.append(Line2D([], [], marker='+', ms=13, mec=GREY, mew=1.8,
                        ls='none'))
        l.append('group median')
        h.append(Line2D([], [], marker='o', ls='', ms=6, mfc='none', mec=GREY,
                        mew=1.3))
        l.append('open: band not converged')
        lg = ax.legend(h, l, loc='lower right', fontsize=7.5, frameon=True,
                       framealpha=0.95, edgecolor='#ddd', borderpad=0.5)
        lg.get_frame().set_facecolor('white')
        lg.get_frame().set_linewidth(0.6)

    axs[0].set_ylabel(r'$\max_i |F_i^{\,\mathrm{DFT}}|$  at the identical'
                      '\n'
                      r'geometry   [eV Å$^{-1}$]')
    p = _save(fig, 'Fig1_v2.png')
    print('   Fig 1: %d Punkte gezeichnet, %d Zeilen' % (drawn, len(rows)))
    for x in med:
        print('   %-6s model/DFT  closed-shell %.3f/%.3f   broken-symmetry %.3f/%.3f'
              % x)
    assert drawn == len(rows) == 135
    return p


# ------------------------------------------------------------------ Fig 2
def fig2():
    """plot_fig2.py mit neuen Gruppennamen in der Legende."""
    plt.rcParams.update(BASE_RC)
    plt.rcParams.update({'axes.linewidth': 0.8, 'axes.grid': False})
    COL = 'f_err_mae'
    rr = [r for r in ROWS if r[COL] != '' and r['s2_ts'] != '']
    val = np.array([float(r[COL]) for r in rr])
    ub = np.abs(np.array([float(r['s2_ts']) for r in rr])) > S2_BREAK
    mm = np.array([r['model'] for r in rr])

    rng = np.random.default_rng(20260823)
    jit = np.random.default_rng(5)

    def ci(v, n=10000):
        v = np.asarray(v, float)
        b = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    fig, axs = plt.subplots(1, 3, figsize=(11.0, 2.9), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.20,
                        wspace=0.06)
    report = []
    for ax, m in zip(axs, MODELS):
        sel = mm == m
        med, nn = {}, {}
        for x0, grp, c, nm in ((0, sel & ~ub, C_ST, NAME_ST),
                               (1, sel & ub, C_UN, NAME_UN)):
            v = val[grp]
            ax.scatter(x0 + jit.uniform(-0.17, 0.17, len(v)), v, s=22, c=c,
                       alpha=0.62, lw=0.4, edgecolor='white', zorder=1)
            lo, hi = ci(v)
            ax.vlines(x0, lo, hi, color=c, lw=6, alpha=0.22, zorder=2)
            md = float(np.median(v))
            med[x0], nn[x0] = md, int(grp.sum())
            ax.plot([x0 - 0.30, x0 + 0.30], [md, md], color=c, lw=2.6,
                    zorder=3, solid_capstyle='butt',
                    label=nm + '   median %.4f eV Å$^{-1}$   (n=%d)'
                    % (md, int(grp.sum())))
        ax.annotate('', xy=(1.86, med[1]), xytext=(1.86, med[0]),
                    arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.0,
                                    shrinkA=0, shrinkB=0))
        ax.text(1.93, np.sqrt(med[0] * med[1]), '×%.1f' % (med[1] / med[0]),
                fontsize=9, color=GREY, va='center', fontweight='bold')
        ax.legend(loc='lower right', fontsize=7.0, frameon=True,
                  framealpha=0.95, edgecolor='#ddd', borderpad=0.4,
                  handlelength=1.6, handletextpad=0.6, labelspacing=0.35,
                  borderaxespad=0.25).set_zorder(9)
        ax.set_xlim(-0.50, 2.32)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([S2_ST, S2_UN])
        ax.tick_params(axis='x', length=0, pad=3)
        ax.set_title(LBL[m], loc='left', pad=3, fontsize=10, fontweight='bold')
        ax.grid(axis='y', color='#eee', lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        report.append((LBL[m], nn[0], med[0], nn[1], med[1], med[1] / med[0]))

    axs[0].set_yscale('log')
    axs[0].set_ylim(val.min() * 0.12, val.max() * 1.50)
    axs[0].set_ylabel(r'MAE $|F^{\rm MLIP}_i - F^{\rm DFT}_i|$   [eV Å$^{-1}$]',
                      fontsize=8.5)
    p = _save(fig, 'fig2_force_mae.png')
    for lab, n0, m0, n1, m1, f in report:
        print('   %-6s closed-shell %.4f (n=%2d)   broken-symmetry %.4f (n=%2d)   x%.1f'
              % (lab, m0, n0, m1, n1, f))
    assert len(rr) == 135
    return p


# ------------------------------------------------------------------ Fig 3
def _draw_barrier_err(axs):
    """Panel A: plot_fig3._draw_barrier_err_c ohne Legende, n an den Ticks."""
    rr = [r for r in ROWS
          if all(r[k] != '' for k in ('barr_model', 'barr_dft', 's2_ts'))]

    def col(k):
        return np.array([float(r[k]) for r in rr])

    err = np.abs(col('barr_model') - col('barr_dft'))
    ub = np.abs(col('s2_ts')) > S2_BREAK
    mm = np.array([r['model'] for r in rr])
    rxn = np.array([r['rxn'] for r in rr])
    unc = np.array([(r['rxn'], r['model']) in UNCONV for r in rr])

    jit = np.random.default_rng(5)
    out = []
    for i, (ax, m) in enumerate(zip(axs, MODELS)):
        sel = mm == m
        summ, n = {}, {}
        for x0, grp, c in ((0, sel & ~ub, C_ST), (1, sel & ub, C_UN)):
            v = err[grp]
            _scatter_open(ax, x0 + jit.uniform(-0.15, 0.15, len(v)), v,
                          unc[grp], c, s=36, alpha=0.60, zorder=1)
            mn, md = float(v.mean()), float(np.median(v))
            summ[x0], n[x0] = (mn, md), int(grp.sum())
            ax.plot([x0 - 0.28, x0 + 0.28], [mn, mn], color=c, lw=2.8,
                    zorder=3, solid_capstyle='butt')
            ax.plot([x0 - 0.22, x0 + 0.22], [md, md], color=c, lw=1.6,
                    zorder=3, ls=(0, (2.2, 1.6)))
            ya, yb = mn, md
            if abs(np.log10(mn / md)) < 0.17:
                g = np.sqrt(mn * md)
                ya, yb = g * 10 ** 0.085, g * 10 ** -0.085
            ax.text(x0 + 0.32, ya, 'MAE %.1f meV' % (mn * 1000), fontsize=8,
                    color=c, va='center', ha='left', fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.18', fc='white',
                              ec='none', alpha=0.85))
            ax.text(x0 + 0.32, yb, 'median %.1f meV' % (md * 1000),
                    fontsize=8, color=c, va='center', ha='left',
                    bbox=dict(boxstyle='square,pad=0.18', fc='white',
                              ec='none', alpha=0.85))
            if x0 == 1:
                k = np.flatnonzero(grp)[int(np.argmax(v))]
                if err[k] > 0.1:
                    ax.annotate('%s   %+.2f eV' % (rxn[k], err[k]),
                                xy=(1.0, err[k]),
                                xytext=(-0.10, err[k] * 2.6),
                                fontsize=8, color=c, ha='left',
                                arrowprops=dict(arrowstyle='->', color=c,
                                                lw=0.9, shrinkB=6))
        (a0, _), (a1, _) = summ[0], summ[1]
        ax.annotate('', xy=(2.52, a1), xytext=(2.52, a0),
                    arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.1,
                                    shrinkA=0, shrinkB=0))
        ax.text(2.60, np.sqrt(a0 * a1), '×%.0f' % (a1 / a0), fontsize=9.5,
                color=GREY, va='center', fontweight='bold')
        if i == 0:
            # einmal sagen, was der Pfeil misst
            ax.text(2.52, a1 * 1.45, 'MAE ratio', fontsize=7.4, color=GREY,
                    ha='center', va='bottom')
            _key_open(ax, 'band not converged', loc='lower right')
        ax.set_title(LBL[m], loc='left', pad=8)
        ax.axhline(0.0434, color=GREEN, lw=1.0, ls='--', zorder=0)
        ax.set_xlim(-0.62, 3.02)
        ax.set_xticks([0, 1])
        # die zwei Gruppen stehen eng; die Namen werden am Bindestrich
        # umbrochen, damit sich die Ticks nicht beruehren
        ax.set_xticklabels(['closed-\nshell\n%s\nn = %d' % (S2_ST, n[0]),
                            'broken-\nsymmetry\n%s\nn = %d' % (S2_UN, n[1])],
                           fontsize=7.6, linespacing=1.25)
        out.append((LBL[m], n[0], summ[0], n[1], summ[1], a1 / a0))
    axs[0].set_yscale('log')
    axs[0].set_ylim(9e-5, 9.0)
    axs[0].set_ylabel('error of the forward barrier at frozen geometry\n'
                      r'$|\,\Delta E^{\ddag}_{\rm MLIP} - '
                      r'\Delta E^{\ddag}_{\rm DFT}\,|$   with   '
                      r'$\Delta E^{\ddag} = E(\mathrm{TS}) - E(\mathrm{R})$'
                      '   [eV]')
    axs[0].text(-0.56, 0.0434 * 1.28, 'chemical accuracy, 43 meV',
                fontsize=7.6, color=GREEN,
                bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))
    return out, len(rr)


def _draw_spread(ax):
    """Panel B: plot_fig3._draw_spread mit neuen Gruppennamen."""
    by = collections.defaultdict(dict)
    for r in ROWS:
        by[r['rxn']][r['model']] = r
    rx, spread, unst, unc = [], [], [], []
    for k, v in by.items():
        if len(v) < 3:
            continue
        b = np.array([float(v[m]['barr_dft']) for m in MODELS])
        rx.append(k)
        spread.append((b.max() - b.min()) * 1000.0)
        unst.append(any(v[m]['unstable_ts'] == '1' for m in MODELS))
        unc.append(any((k, m) in UNCONV for m in MODELS))
    rx = np.array(rx)
    spread, unst, unc = np.array(spread), np.array(unst), np.array(unc)
    CHEM = 43.4

    jit = np.random.default_rng(11)
    ax.axhspan(0.002, CHEM, color=C_ST, alpha=0.05, lw=0, zorder=0)
    ax.axhline(CHEM, color=GREEN, lw=1.1, ls='--', zorder=2)
    out = []
    for x0, sel, c in ((0, ~unst, C_ST), (1, unst, C_UN)):
        v = spread[sel]
        xj = x0 + jit.uniform(-0.17, 0.17, len(v))
        _scatter_open(ax, xj, v, unc[sel], c, s=44, alpha=0.70, zorder=3)
        mn, md = float(v.mean()), float(np.median(v))
        ax.plot([x0 - 0.26, x0 + 0.26], [mn, mn], color=c, lw=2.8, zorder=4,
                solid_capstyle='butt')
        ax.plot([x0 - 0.20, x0 + 0.20], [md, md], color=c, lw=1.7, zorder=4,
                ls=(0, (2.2, 1.6)))
        ax.text(x0 + 0.30, mn, 'mean %.0f meV' % mn, fontsize=8, color=c,
                va='center', ha='left', fontweight='bold')
        ax.text(x0 + 0.30, md, 'median %.2f meV' % md, fontsize=8, color=c,
                va='center', ha='left')
        ax.text(x0, 0.0052, '%d of %d\nabove the line'
                % (int((v > CHEM).sum()), len(v)),
                fontsize=8.6, color=c, ha='center', va='bottom',
                fontweight='bold', linespacing=1.5)
        for i in np.flatnonzero(sel):
            if spread[i] <= CHEM:
                continue
            kk = int(np.flatnonzero(np.flatnonzero(sel) == i)[0])
            ax.annotate(rx[i], xy=(xj[kk], spread[i]),
                        xytext=(x0 - 0.46, spread[i]),
                        fontsize=7.6, color=c, ha='right', va='center',
                        bbox=dict(boxstyle='square,pad=0.12', fc='white',
                                  ec='none', alpha=0.9),
                        arrowprops=dict(arrowstyle='-', lw=0.7, color=c,
                                        alpha=0.6, shrinkA=1, shrinkB=3))
        out.append((len(v), mn, md, int((v > CHEM).sum())))

    ax.set_yscale('log')
    ax.set_ylim(0.0032, 12000)
    ax.set_xlim(-0.95, 1.78)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['all three transition\nstates %s' % NAME_ST,
                        'at least one\n%s' % NAME_UN])
    ax.set_ylabel(r'max − min of $\Delta E^{\ddag}$ from DFT, over the three '
                  'model geometries\nof one reaction   [meV]')
    ax.text(-0.92, CHEM * 1.75, 'chemical accuracy, 43 meV', fontsize=7.8,
            color=GREEN, ha='left', va='center')
    _key_open(ax, 'at least one band\nnot converged', loc='upper left')
    return out, len(rx)


def fig3():
    plt.rcParams.update(BASE_RC)
    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.18,
                         'grid.linewidth': 0.6})
    fig = plt.figure(figsize=(15.6, 6.6))
    subfigs = fig.subfigures(1, 2, width_ratios=[2.85, 1.15], wspace=0.02)
    axs_left = subfigs[0].subplots(1, 3, sharey=True)
    a, na = _draw_barrier_err(axs_left)
    subfigs[0].text(0.006, 0.995, 'A', fontsize=16, fontweight='bold',
                    va='top', ha='left')
    ax_right = subfigs[1].subplots(1, 1)
    b, nb = _draw_spread(ax_right)
    subfigs[1].text(0.02, 0.995, 'B', fontsize=16, fontweight='bold',
                    va='top', ha='left')
    p = _save(fig, 'fig3_v2.png')
    print('   Fig 3A: %d Zeilen' % na)
    for lab, n0, (mn0, md0), n1, (mn1, md1), f in a:
        print('   %-6s closed-shell n=%2d MAE %5.1f med %4.1f   '
              'broken-symmetry n=%2d MAE %6.1f med %4.1f   x%.0f'
              % (lab, n0, mn0 * 1e3, md0 * 1e3, n1, mn1 * 1e3, md1 * 1e3, f))
    print('   Fig 3B: %d Reaktionen' % nb)
    for (n, mn, md, k), nm in zip(b, (NAME_ST, NAME_UN)):
        print('   %-16s n=%2d mean %6.1f med %6.2f  >43 meV: %d'
              % (nm, n, mn, md, k))
    assert na == 135 and nb == 45
    return p


if __name__ == '__main__':
    print('PAPERFIGUREN v2 -> Pictures/')
    fig1()
    fig2()
    fig3()
