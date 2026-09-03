# -*- coding: utf-8 -*-
"""Figure 3 -- combined view for the chapter text.

Left (panel A): the forward barrier error at frozen geometry, exactly as in
fig9_3c_barrier_error_omol25.png, but cropped to the chart itself -- no
suptitle, no caption paragraph.

Right (panel B): the spread of the DFT barrier over the three model
transition states of one reaction, exactly as in
fig9_5_barrier_spread_omol25.png, same crop.

Both panels reuse the identical data, computations and draw calls as
fig_barrier_err_c() / fig_spread() in plot_omol25_figs.py -- only the
suptitle/title and the caption text.py are dropped, and a panel letter is
placed in the top-left corner of each half instead.

Data: results/omol25_model_geoms.csv
"""
import collections
import csv
import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
S2_BREAK = 0.05

C_ST = '#2a6f7f'
C_UN = '#c2542a'
GREY = '#6b6b6b'
LBL = {'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'esen': 'eSEN'}

mpl.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9,
    'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.18, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
})


def _draw_barrier_err_c(axs):
    """Chart of fig9_3c_barrier_error_omol25, minus suptitle/caption."""
    rr = [r for r in csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv')))
          if all(r[k] != '' for k in ('barr_model', 'barr_dft', 's2_ts'))]

    def col(k):
        return np.array([float(r[k]) for r in rr])

    err = np.abs(col('barr_model') - col('barr_dft'))
    ub = np.abs(col('s2_ts')) > S2_BREAK
    mm = np.array([r['model'] for r in rr])
    rxn = np.array([r['rxn'] for r in rr])

    jit = np.random.default_rng(5)
    for ax, m in zip(axs, ('uma-s', 'uma-m', 'esen')):
        sel = mm == m
        summ, n = {}, {}
        for x0, grp, c in ((0, sel & ~ub, C_ST), (1, sel & ub, C_UN)):
            v = err[grp]
            ax.scatter(x0 + jit.uniform(-0.15, 0.15, len(v)), v, s=36, c=c,
                       alpha=0.60, lw=0.5, edgecolor='white', zorder=1)
            mn, md = float(v.mean()), float(np.median(v))
            summ[x0], n[x0] = (mn, md), int(grp.sum())
            ax.plot([x0 - 0.28, x0 + 0.28], [mn, mn], color=c, lw=2.8, zorder=3,
                    solid_capstyle='butt')
            ax.plot([x0 - 0.22, x0 + 0.22], [md, md], color=c, lw=1.6, zorder=3,
                    ls=(0, (2.2, 1.6)))
            ya, yb = mn, md
            if abs(np.log10(mn / md)) < 0.17:
                g = np.sqrt(mn * md)
                ya, yb = g * 10 ** 0.085, g * 10 ** -0.085
            ax.text(x0 + 0.32, ya, 'MAE %.1f meV' % (mn * 1000), fontsize=8,
                    color=c, va='center', ha='left', fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.18', fc='white',
                              ec='none', alpha=0.85))
            ax.text(x0 + 0.32, yb, 'median %.1f meV' % (md * 1000), fontsize=8,
                    color=c, va='center', ha='left',
                    bbox=dict(boxstyle='square,pad=0.18', fc='white',
                              ec='none', alpha=0.85))
            if x0 == 1:
                k = np.flatnonzero(grp)[int(np.argmax(v))]
                if err[k] > 0.1:
                    ax.annotate('%s   %+.2f eV' % (rxn[k], err[k]),
                                xy=(1.0, err[k]), xytext=(-0.10, err[k] * 2.6),
                                fontsize=8, color=c, ha='left',
                                arrowprops=dict(arrowstyle='->', color=c,
                                                lw=0.9, shrinkB=6))
        (a0, _), (a1, _) = summ[0], summ[1]
        ax.annotate('', xy=(2.52, a1), xytext=(2.52, a0),
                    arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.1,
                                    shrinkA=0, shrinkB=0))
        ax.text(2.60, np.sqrt(a0 * a1), '×%.0f' % (a1 / a0), fontsize=9.5,
                color=GREY, va='center', fontweight='bold')
        key = [Line2D([], [], marker='o', ls='', ms=6.5, mfc=C_ST, mec='white',
                      alpha=0.85, label='one reaction, RKS stable     n = %d'
                                        % n[0]),
               Line2D([], [], marker='o', ls='', ms=6.5, mfc=C_UN, mec='white',
                      alpha=0.85, label='one reaction, RKS unstable   n = %d'
                                        % n[1])]
        ax.legend(handles=key, loc='lower left', fontsize=7.4, frameon=True,
                  framealpha=0.96, edgecolor='#ccc', handlelength=1.0,
                  labelspacing=0.5, borderpad=0.7,
                  bbox_to_anchor=(0.005, 0.012))
        ax.set_title(LBL[m], loc='left', pad=8)
        ax.axhline(0.0434, color='#3b7d3b', lw=1.0, ls='--', zorder=0)
        ax.set_xlim(-0.62, 3.02)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['RKS stable\n' r'$\langle S^2\rangle = 0$',
                            'RKS unstable\n' r'$\langle S^2\rangle > 0$'])
    axs[0].set_yscale('log')
    axs[0].set_ylim(9e-5, 9.0)
    axs[0].set_ylabel('error of the forward barrier at frozen geometry\n'
                      r'$|\,\Delta E^{\ddag}_{\rm MLIP} - '
                      r'\Delta E^{\ddag}_{\rm DFT}\,|$   with   '
                      r'$\Delta E^{\ddag} = E(\mathrm{TS}) - E(\mathrm{R})$'
                      '   [eV]')
    axs[0].text(-0.56, 0.0434 * 1.28, 'chemical accuracy, 43 meV',
                fontsize=7.6, color='#3b7d3b',
                bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))


def _draw_spread(ax):
    """Chart of fig9_5_barrier_spread_omol25, minus title/caption."""
    rr = list(csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv'))))
    by = collections.defaultdict(dict)
    for r in rr:
        by[r['rxn']][r['model']] = r

    M = ('uma-s', 'uma-m', 'esen')
    rx, spread, unst = [], [], []
    for k, v in by.items():
        if len(v) < 3:
            continue
        b = np.array([float(v[m]['barr_dft']) for m in M])
        rx.append(k)
        spread.append((b.max() - b.min()) * 1000.0)
        unst.append(any(v[m]['unstable_ts'] == '1' for m in M))
    rx = np.array(rx)
    spread, unst = np.array(spread), np.array(unst)
    CHEM = 43.4

    jit = np.random.default_rng(11)
    ax.axhspan(0.002, CHEM, color=C_ST, alpha=0.05, lw=0, zorder=0)
    ax.axhline(CHEM, color='#3b7d3b', lw=1.1, ls='--', zorder=2)

    for x0, sel, c in ((0, ~unst, C_ST), (1, unst, C_UN)):
        v = spread[sel]
        xj = x0 + jit.uniform(-0.17, 0.17, len(v))
        ax.scatter(xj, v, s=44, c=c, alpha=0.70, lw=0.5, edgecolor='white',
                   zorder=3)
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

    ax.set_yscale('log')
    ax.set_ylim(0.0032, 12000)
    ax.set_xlim(-0.95, 1.78)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['all three transition\nstates RKS stable',
                        'at least one\nRKS unstable'])
    ax.set_ylabel(r'max − min of $\Delta E^{\ddag}$ from DFT, over the three '
                  'model geometries\nof one reaction   [meV]')
    ax.text(-0.92, CHEM * 1.75, 'chemical accuracy, 43 meV', fontsize=7.8,
            color='#3b7d3b', ha='left', va='center')


def fig3():
    fig = plt.figure(figsize=(15.6, 6.6))
    subfigs = fig.subfigures(1, 2, width_ratios=[2.85, 1.15], wspace=0.02)

    axs_left = subfigs[0].subplots(1, 3, sharey=True)
    _draw_barrier_err_c(axs_left)
    subfigs[0].text(0.006, 0.995, 'A', fontsize=16, fontweight='bold',
                    va='top', ha='left')

    ax_right = subfigs[1].subplots(1, 1)
    _draw_spread(ax_right)
    subfigs[1].text(0.02, 0.995, 'B', fontsize=16, fontweight='bold',
                    va='top', ha='left')

    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig3.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


fig3()
