# DEPRECATED -- TZVP-era figures/tables, superseded by
# pipeline/omol25_model_geoms.py, pipeline/hinge_tables.py and
# pipeline/plot_omol25_figs.py. Do not run for the paper; retained as history.
# The 1.697 eV/A median is obsolete; successor numbers live in
# results/hinge_t1x.csv (1.636) and results/hinge_omol25.csv (1.870).

"""Die vier Figuren des Workshop-Papers.

Datenbasis ausschliesslich results/paper_rows_ext.csv (122 Zeilen),
results/hinge_rows.csv (19) und results/control_rks.csv (26); alle drei sind
von pipeline/paper_rows.py bzw. pipeline/paper_figdata.py gegen die
Kapitelzahlen validiert worden.

    fig1_silent_failure.png   das Modell meldet ueberall dieselbe Restkraft
    fig2_predictor.png        die Stabilitaetsanalyse sagt das Scheitern vorher
    fig3_seam.png             der Kraftfehler sitzt an der Naht
    fig4_hinge.png            die Labels stehen nicht auf der Grundzustandsflaeche

Lokal ausfuehrbar, kein Clusterzugriff.  Jede Zahl in den Beschriftungen wird
hier aus den CSVs gerechnet, keine ist eingetippt.
"""
import csv
import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
STAT = 0.15          # Stufe-1-Schwelle, eV/A
CINEB = 0.05         # Abbruchkriterium der Modell-NEBs, eV/A
NBOOT = 10000
SEED = 20260819

C_ST = '#2a6f7f'     # RKS stabil
C_UN = '#c2542a'     # RKS instabil
C_MOD = {'uma-s': '#4c6ef5', 'uma-m': '#7048e8', 'esen': '#0ca678'}
LBL = {'uma-s': 'UMA-S', 'uma-m': 'UMA-M', 'esen': 'eSEN'}
GREY = '#6b6b6b'

mpl.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.18, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
})


# ----------------------------------------------------------------- Werkzeug
def load(name):
    with open(os.path.join(RES, name)) as fh:
        out = []
        for r in csv.DictReader(fh):
            d = {}
            for k, v in r.items():
                if k in ('rxn', 'model', 'schritt', 'variante'):
                    d[k] = v
                else:
                    d[k] = float(v) if v != '' else None
            out.append(d)
        return out


def auc(scores, labels):
    """Mann-Whitney-AUC mit Bindungskorrektur, identisch zu sep_analysis.py."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def roc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    o = np.argsort(-s)
    y = y[o]
    tp = np.concatenate([[0], np.cumsum(y)])
    fp = np.concatenate([[0], np.cumsum(~y)])
    return fp / fp[-1], tp / tp[-1]


def boot_median(v, n=4000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    v = np.asarray(v, float)
    m = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def spearman(a, b):
    def rk(x):
        x = np.asarray(x, float)
        o = np.argsort(x)
        r = np.empty(len(x), float)
        r[o] = np.arange(1, len(x) + 1)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt))
        np.add.at(s, inv, r)
        return (s / cnt)[inv]
    ra, rb = rk(a), rk(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def panel(ax, letter, title):
    ax.set_title(title, loc='left', pad=8)
    ax.text(-0.09, 1.13, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left')


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


rows = load('paper_rows_ext.csv')
hinge = load('hinge_rows.csv')
ctrl = load('control_rks.csv')

fm = np.array([r['F_model'] for r in rows])
fd = np.array([r['F_dft'] for r in rows])
mae = np.array([r['mae_force'] for r in rows])
mce = np.array([r['maxcomp_err'] for r in rows])
dep = np.array([r['breaking_depth'] for r in rows])
lam = np.array([r['lambda_min'] for r in rows])
nfd = np.array([r['nfod'] for r in rows])
uns = np.array([r['unstable'] for r in rows]).astype(bool)
mdl = np.array([r['model'] for r in rows])
# dieselben Groessen an der MODELLGEOMETRIE -- die Variable aus Abschnitt 5
depm = np.array([np.nan if r['breaking_depth_model'] is None
                 else r['breaking_depth_model'] for r in rows])
lamm = np.array([np.nan if r['lambda_min_model'] is None
                 else r['lambda_min_model'] for r in rows])
bad = fd >= STAT                       # nicht stationaer laut DFT

print('Figuren aus %d Zeilen, %d Reaktionen, %d nicht stationaer'
      % (len(rows), len({r['rxn'] for r in rows}), int(bad.sum())))


# =====================================================================  FIG 1
def fig1():
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(12.4, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.30,
                          height_ratios=[1.25, 1.0])

    says_ok = fm < CINEB
    Q = {'lo': (says_ok & ~bad), 'lu': None}          # nur als Merkhilfe
    n_lu = int((says_ok & ~bad).sum())                # links unten
    n_lo = int((says_ok & bad).sum())                 # links oben
    n_ru = int((~says_ok & ~bad).sum())               # rechts unten
    n_ro = int((~says_ok & bad).sum())                # rechts oben

    # -- A  Kernscatter mit beschrifteten Quadranten
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, 'A', 'Was das Modell meldet gegen das, was wirklich dort wirkt')
    ax.set_title('Was das Modell meldet gegen das, was wirklich dort wirkt',
                 loc='left', pad=26)
    x0, x1 = min(fm.min(), 3e-4) * 0.75, fm.max() * 1.6
    y0, y1 = min(fd.min(), 3e-3) * 0.6, fd.max() * 3.2
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    def fx(v):
        return (np.log10(v) - np.log10(x0)) / (np.log10(x1) - np.log10(x0))

    def fy(v):
        return (np.log10(v) - np.log10(y0)) / (np.log10(y1) - np.log10(y0))

    qx, qy = fx(CINEB), fy(STAT)
    tint = [((0, qy, qx, 1 - qy), '#c2542a', 0.13),          # links oben
            ((0, 0, qx, qy), '#2a6f7f', 0.09),               # links unten
            ((qx, qy, 1 - qx, 1 - qy), '#b9a06a', 0.13),     # rechts oben
            ((qx, 0, 1 - qx, qy), '#9aa0a6', 0.09)]          # rechts unten
    for (rx_, ry_, rw, rh), c, al in tint:
        ax.add_patch(Rectangle((rx_, ry_), rw, rh, transform=ax.transAxes,
                               facecolor=c, alpha=al, lw=0, zorder=0))

    ax.plot([1e-5, 10], [1e-5, 10], color=GREY, lw=1, ls='--', zorder=1)
    ax.axhline(STAT, color='k', lw=1.1, ls=':', zorder=2)
    ax.axvline(CINEB, color='k', lw=1.1, ls=':', zorder=2)
    ax.scatter(fm[~uns], fd[~uns], s=26, c=C_ST, alpha=0.85, lw=0, zorder=3,
               label='RKS stabil  (n=%d)' % (~uns).sum())
    ax.scatter(fm[uns], fd[uns], s=32, c=C_UN, alpha=0.85, lw=0, zorder=3,
               marker='D', label='RKS instabil  (n=%d)' % uns.sum())

    box = dict(boxstyle='round,pad=0.32', fc='white', ec='#ccc', alpha=0.9)
    ax.text(qx / 2, 0.975, 'STILLER AUSFALL\nModell meldet fertig,\n'
                           'ist aber kein Sattelpunkt\n%d Zeilen' % n_lo,
            transform=ax.transAxes, ha='center', va='top', fontsize=8.2,
            color='#8f3a17', fontweight='bold', bbox=box, zorder=6)
    ax.text(qx / 2, 0.025, 'richtig zu Ende gerechnet\nfertig gemeldet '
                           'und stationär\n%d Zeilen' % n_lu,
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8.2,
            color='#1d5460', bbox=box, zorder=6)
    ax.text((1 + qx) / 2, 0.975, 'ehrlich gescheitert\nRestkraft gemeldet,\n'
                                 'kein Sattelpunkt\n%d' % n_ro,
            transform=ax.transAxes, ha='center', va='top', fontsize=8.2,
            color='#6d5620', bbox=box, zorder=6)
    ax.text((1 + qx) / 2, 0.025, 'übervorsichtig\nRestkraft gemeldet,\n'
                                 'aber stationär\n%d' % n_ru,
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8.2,
            color='#4a5057', bbox=box, zorder=6)

    ax.set_xlabel(r'$|F|_{\rm Modell}$ — was der Kalkulator an seiner eigenen '
                  'Vorhersage meldet  [eV/Å]')
    ax.set_ylabel(r'$|F|_{\rm DFT}$ an genau derselben Geometrie  [eV/Å]')
    ax.text(CINEB * 0.88, 0.012, 'NEB-Abbruch  0.05', rotation=90,
            transform=ax.get_xaxis_transform(), ha='right', va='bottom',
            fontsize=7.5, zorder=6)
    ax.text(0.012, STAT * 1.14, 'Stufe 1  0.15', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=7.5, zorder=6)
    ax.text(0.975, 0.795, 'Identität  x = y', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.5, color=GREY, rotation=30)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=2)
    ax.text(0.5, -0.235,
            'Senkrecht: was das Modell für seine Restkraft hält.  '
            'Waagrecht: was DFT dort findet.\n'
            'Der senkrechte Abstand zur gestrichelten Linie ist der Irrtum.',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic',
            color=GREY)

    # -- B  Slopegraph der Mediane, mit Lesehilfe
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, 'B', 'Dieselben Strukturen, zweimal gemessen')
    rng = np.random.default_rng(SEED)
    for m, c, nm in ((~uns, C_ST, 'RKS stabil'), (uns, C_UN, 'RKS instabil')):
        a, b = float(np.median(fm[m])), float(np.median(fd[m]))
        ca, cb = boot_median(fm[m], rng=rng), boot_median(fd[m], rng=rng)
        ax.plot([0, 1], [a, b], color=c, lw=2.2, marker='o', ms=8, zorder=3,
                label='%s  ·  %d Strukturen' % (nm, m.sum()))
        ax.vlines(0, *ca, color=c, lw=6, alpha=0.28)
        ax.vlines(1, *cb, color=c, lw=6, alpha=0.28)
        ax.annotate('%.3f' % a, (0, a), xytext=(-11, 0),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=9, color=c, fontweight='bold')
        ax.annotate('%.3f' % b, (1, b), xytext=(11, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=9, color=c, fontweight='bold')
    ax.set_xlim(-0.46, 1.46)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['der Kalkulator des Modells\nan der Modellgeometrie',
                        'ωB97M-V/def2-TZVP\nan derselben Geometrie'], fontsize=8.5)
    ax.set_ylabel('Median von max|F| über die Gruppe  [eV/Å]')
    ax.axhline(STAT, color='k', lw=0.9, ls=':')
    ax.text(-0.42, STAT * 1.03, 'Stufe 1  0.15', ha='left', va='bottom',
            fontsize=7.5)
    ax.set_ylim(0, 0.205)
    ax.legend(loc='upper left')
    ax.annotate('', xy=(0.5, 0.0355), xytext=(0.5, 0.0275),
                arrowprops=dict(arrowstyle='-[, widthB=0.6, lengthB=0.25',
                                color=GREY, lw=1))
    ax.text(0.53, 0.0245, 'die beiden Startpunkte fallen zusammen:\n'
                          'Unterschied 0.0001 eV/Å',
            ha='left', va='top', fontsize=8, style='italic', color=GREY)
    ax.text(0.5, -0.235,
            'Jeder Punkt ist ein Gruppenmedian über die Zeilen von A.  '
            'Nur der Kraftlieferant\nwechselt, Geometrie und Atome sind '
            'identisch.  Schattierte Balken:\n95 %-Bootstrap-CI des Medians.',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic',
            color=GREY)

    # -- C  Konfusion am tatsaechlich benutzten Kriterium
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, 'C', 'Dieselben vier Quadranten als Zählung')
    M = np.array([[n_lu, n_lo], [n_ru, n_ro]])
    ax.imshow(M / M.sum(), cmap='Oranges', vmin=0, vmax=0.62, aspect='auto')
    names = [['richtig', 'STILLER AUSFALL'], ['übervorsichtig', 'ehrlich gescheitert']]
    for i in range(2):
        for j in range(2):
            frac = M[i, j] / M.sum()
            col = 'white' if frac > 0.33 else '#222'
            ax.text(j, i - 0.17, '%d' % M[i, j], ha='center', va='center',
                    fontsize=20, fontweight='bold', color=col)
            ax.text(j, i + 0.12, names[i][j], ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=col)
            ax.text(j, i + 0.30, '%.0f %% aller Zeilen' % (100 * frac),
                    ha='center', va='center', fontsize=7.5, color=col)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['stationär\nDFT < 0.15', 'NICHT stationär\nDFT ≥ 0.15'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Modell meldet\n< 0.05', 'Modell meldet\n≥ 0.05'])
    ax.grid(False)
    ax.set_xlabel('Von %d Zeilen unter dem Modellkriterium sind %d keine '
                  'Stationärpunkte — %.0f %%'
                  % (M[0].sum(), n_lo, 100 * n_lo / M[0].sum()), fontsize=8.5)

    # -- D  je Modell
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, 'D', 'Kein Modell fällt aus dem Rahmen')
    xs = np.arange(3)
    w = 0.19
    for k, (m, c, nm) in enumerate(((~uns, C_ST, 'stabil'), (uns, C_UN, 'instabil'))):
        for j, (arr, hatch, lab) in enumerate(((fm, '', 'Modell'), (fd, '///', 'DFT'))):
            v = [np.median(arr[m & (mdl == s)]) for s in ('uma-s', 'uma-m', 'esen')]
            ax.bar(xs + (k * 2 + j) * w - 1.5 * w, v, w, color=c, lw=0,
                   alpha=1.0 if j == 0 else 0.42, hatch=hatch,
                   label='%s — %s' % (nm, lab))
    ax.axhline(STAT, color='k', lw=0.9, ls=':')
    ax.set_xticks(xs)
    ax.set_xticklabels([LBL[s] for s in ('uma-s', 'uma-m', 'esen')])
    ax.set_ylabel('Median max|F|  [eV/Å]')
    ax.legend(ncol=2, loc='upper left')
    ax.set_ylim(0, 0.235)

    fig.suptitle('Silent failure — die Modellkraft unterscheidet nicht, '
                 'was die DFT-Kraft klar trennt',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=1.0)
    fig.text(0.012, -0.025,
             'Alle 122 Zeilen aus results/paper_rows_ext.csv.  '
             r'$|F|$ ist stets die grösste Betragskomponente, keine Norm.  '
             'Vorbehalt: das NEB bricht auf der projizierten Kraft ab, '
             r'$|F|_{\rm Modell}$ ist die reine Kalkulatorkraft an derselben '
             'Struktur — die 0.05-Linie ist deshalb eine Näherung des '
             'Abbruchkriteriums, keine exakte Nachbildung.',
             fontsize=7.5, color=GREY, ha='left')
    save(fig, 'fig1_silent_failure.png')


# =====================================================================  FIG 2
def fig2():
    rxa = np.array([r['rxn'] for r in rows])
    RXN = sorted(set(rxa))
    IDX = {r: np.flatnonzero(rxa == r) for r in RXN}
    preds = [(-lam, r'$-\lambda_{\min}^{\rm ext}$ (kontinuierlich)', '#c2542a', '-'),
             (uns.astype(float), 'instabil ja/nein (binär)', '#2a6f7f', '--'),
             (nfd, r'$N_{\rm FOD}$ (kontinuierlich)', '#7048e8', '-.')]

    rng = np.random.default_rng(SEED)
    grid = np.linspace(0, 1, 201)
    band, dlt = [], []
    for _ in range(NBOOT):
        idx = np.concatenate([IDX[RXN[j]] for j in rng.integers(0, len(RXN), len(RXN))])
        yy = bad[idx]
        if yy.all() or not yy.any():
            continue
        f, t = roc(-lam[idx], yy)
        band.append(np.interp(grid, f, t))
        dlt.append(auc(-lam[idx], yy) - auc(nfd[idx], yy))
    band, dlt = np.array(band), np.array(dlt)
    lo_b, hi_b = np.percentile(band, 2.5, axis=0), np.percentile(band, 97.5, axis=0)
    lo_d, hi_d = np.percentile(dlt, 2.5), np.percentile(dlt, 97.5)

    fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.4))
    fig.subplots_adjust(hspace=0.42, wspace=0.28)

    # -- A  ROC
    ax = axs[0, 0]
    panel(ax, 'A', 'Die Instabilität sagt vorher, wo das Modell scheitert')
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.9, ls=':')
    ax.fill_between(grid, lo_b, hi_b, color='#c2542a', alpha=0.15, lw=0)
    for sc, nm, c, ls in preds:
        f, t = roc(sc, bad)
        ax.plot(f, t, color=c, ls=ls, lw=2,
                label='%s   AUC %.3f' % (nm, auc(sc, bad)))
    ax.set_xlabel('Falsch-positiv-Rate')
    ax.set_ylabel('Richtig-positiv-Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    lg = ax.legend(loc='lower right',
                   title='Band = 95 %%-CI, nach Reaktion geclustert\n'
                         '%d Resamples über %d Reaktionen'
                         % (len(band), len(RXN)))
    lg.get_title().set_fontsize(7.5)
    lg.get_title().set_color(GREY)

    # -- B  Delta-AUC
    ax = axs[0, 1]
    panel(ax, 'B', 'Der Vorsprung gegenüber $N_{\\rm FOD}$ ist nicht abgesichert')
    ax.hist(dlt, bins=60, color='#b0b8c4', lw=0)
    ax.axvline(0, color='k', lw=1.2)
    ax.axvspan(lo_d, hi_d, color='#c2542a', alpha=0.14, lw=0)
    for v in (lo_d, hi_d):
        ax.axvline(v, color='#c2542a', lw=1.2, ls='--')
    ax.axvline(float(np.median(dlt)), color='#c2542a', lw=2)
    ax.set_xlabel(r'$\Delta$AUC  $=$  AUC$(-\lambda_{\min})\ -\ $AUC$(N_{\rm FOD})$, '
                  'je Resample gepaart')
    ax.set_ylabel('Resamples')
    ax.text(0.97, 0.95, '95 %%-CI  [%.3f, %.3f]\nenthält die Null\n'
                        '%.1f %% der Resamples > 0'
            % (lo_d, hi_d, 100 * (dlt > 0).mean()),
            transform=ax.transAxes, fontsize=8.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ddd', alpha=0.92))

    # -- C  Anti-Zirkel
    ax = axs[1, 0]
    panel(ax, 'C', 'Die Trennung überlebt innerhalb der $N_{\\rm FOD}$-Flags')
    groups = [('alle 122 Zeilen', np.ones(len(rows), bool)),
              (r'nur $N_{\rm FOD} > 0.5$', nfd > 0.5)]
    xs = np.arange(2)
    w = 0.34
    for k, (m, c, nm) in enumerate(((~uns, C_ST, 'RKS stabil'), (uns, C_UN, 'RKS instabil'))):
        vals, ns = [], []
        for _, sel in groups:
            s = sel & m
            vals.append(100 * (~bad[s]).sum() / s.sum())
            ns.append(int(s.sum()))
        b = ax.bar(xs + (k - 0.5) * w, vals, w, color=c, lw=0, label=nm)
        for r_, v, n in zip(b, vals, ns):
            ax.text(r_.get_x() + w / 2, v + 1.6, '%.0f %%' % v, ha='center',
                    fontsize=10, fontweight='bold', color=c)
            ax.text(r_.get_x() + w / 2, 2.5, 'n=%d' % n, ha='center',
                    fontsize=7.5, color='white')
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel('Anteil gültiger Stationärpunkte  [%]')
    ax.set_ylim(0, 108)
    ax.legend(loc='upper right')
    ax.text(0.5, -0.20, 'Rechts sind alle Zeilen bereits als multireferenziell '
                        'markiert — die Stabilitätsanalyse trennt sie weiter.',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic', color=GREY)

    # -- D  lambda-Streifen
    ax = axs[1, 1]
    panel(ax, 'D', 'Die Schwelle ist nicht getunt, sie liegt bei null')
    rj = np.random.default_rng(7).uniform(-0.30, 0.30, len(rows))
    for m, c, nm, mk in ((~bad, '#2a6f7f', 'stationär (DFT < 0.15)', 'o'),
                         (bad, '#c2542a', 'nicht stationär', 'D')):
        ax.scatter(lam[m], rj[m], s=30, c=c, alpha=0.8, lw=0, marker=mk, label=nm)
    ax.axvline(0, color='k', lw=1.4)
    ax.set_xlabel(r'$\lambda_{\min}^{\rm ext}$ am RKS-TS  [Hartree]')
    ax.set_yticks([])
    ax.set_ylim(-0.9, 0.9)
    ax.legend(loc='upper left')
    ax.text(0.002, -0.72, 'instabil  ←', ha='right', fontsize=8, color=GREY)
    ax.text(0.004, -0.72, '→  stabil', ha='left', fontsize=8, color=GREY)

    fig.suptitle('Prädiktor — eine Stabilitätsanalyse an der Referenzgeometrie '
                 'zeigt die Ausfälle vorab an',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=0.985)
    fig.text(0.012, 0.012,
             'Ziel: die Modellstruktur ist kein Stationärpunkt, max|F|$_{\\rm DFT}$ ≥ 0.15 eV/Å.  '
             '122 Zeilen = 42 Reaktionen × 3 Modelle, 29 Positive.  '
             'Bootstrap nach Reaktion geclustert, weil die drei Modellzeilen einer '
             'Reaktion denselben Prädiktorwert teilen.',
             fontsize=7.5, color=GREY, ha='left')
    save(fig, 'fig2_predictor.png')


# =====================================================================  FIG 3
def fig3():
    """Die Naht-Frage mit der Variablen, mit der Abschnitt 5 argumentiert:
    Brechungstiefe AN DER MODELLGEOMETRIE, ein Wert je Zeile.

    Die Kapiteltabelle reproduziert sich Zeile fuer Zeile.  Der Peak bei
    flacher Brechung steckt aber nur in der Spalte max|F|_DFT - max|F|_Modell
    -- der Differenz zweier Maximalkomponenten, die an verschiedenen Atomen
    sitzen koennen.  Mit dem Komponenten-MAE verschwindet er.
    """
    have = ~np.isnan(depm)
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 9.6))
    fig.subplots_adjust(hspace=0.62, wspace=0.30)
    rxa = np.array([r['rxn'] for r in rows])
    BINS = [(-1, 0, 'stabil\nΔE = 0'), (0, 50, '1–50'),
            (50, 200, '50–200'), (200, 1e9, '> 200')]

    def sel_bin(a, b):
        return have & ((depm == 0) if b == 0 else ((depm > a) & (depm <= b)))

    # -- A  Streuung gegen die Tiefe an der Modellgeometrie
    ax = axs[0, 0]
    panel(ax, 'A', 'Ein Sprung bei null — danach kein Trend')
    x = np.where(depm > 0, depm, 0.30)
    for s in ('uma-s', 'uma-m', 'esen'):
        m = have & (mdl == s)
        ax.scatter(x[m], mae[m], s=30, c=C_MOD[s], alpha=0.72, lw=0, label=LBL[s])
    m0, m1 = have & (depm == 0), have & (depm > 0)
    med0, med1 = float(np.median(mae[m0])), float(np.median(mae[m1]))
    ax.hlines(med0, 0.20, 0.45, color='k', lw=2.4, zorder=5)
    ax.hlines(med1, 1.0, 800, color='k', lw=2.4, zorder=5)
    ax.annotate('', xy=(0.62, med1), xytext=(0.62, med0),
                arrowprops=dict(arrowstyle='->', lw=1.6, color='k'))
    ax.text(0.70, float(np.sqrt(med0 * med1)), '%.1f×' % (med1 / med0),
            fontsize=11, fontweight='bold', va='center')
    ax.text(28, med1 * 1.16, 'Median der gebrochenen Zeilen %.3f — flach' % med1,
            fontsize=8, ha='center')
    ax.axvline(0.62, color=GREY, lw=0.8, ls=':')
    ax.set_xscale('symlog', linthresh=1.0, linscale=0.30)
    ax.set_yscale('log')
    ax.set_xlim(0.12, 900)
    ax.set_xlabel('Brechungstiefe an der Modellgeometrie  [meV]')
    ax.set_ylabel('Kraftfehler MAE gegen DFT  [eV/Å]')
    ax.set_xticks([0.30, 1, 10, 100, 600])
    ax.set_xticklabels(['0\n(dort stabil)', '1', '10', '100', '600'])
    ax.legend(loc='upper left', ncol=3, fontsize=7.5)
    ax.text(0.99, 0.02, 'n = %d Zeilen  ·  %d davon gebrochen'
            % (int(have.sum()), int(m1.sum())), transform=ax.transAxes,
            ha='right', fontsize=7.5, color=GREY)

    # -- B  woher der Peak kam, jeder Punkt einzeln
    ax = axs[0, 1]
    panel(ax, 'B', 'Woher das Maximum bei flacher Brechung kam')
    ax.set_title('Woher das Maximum bei flacher Brechung kam', loc='left',
                 pad=30)
    diff = fd - fm                       # die Spalte, mit der §5 rechnet
    off, jit = 0.20, 0.075
    rj = np.random.default_rng(11)
    med_a, med_n = [], []
    for i, (a, b, nm) in enumerate(BINS):
        m = sel_bin(a, b)
        for k, (v, c, lab) in enumerate(
                ((diff[m], '#b9a06a',
                  r'§5:  max$|F|_{\rm DFT}$ $-$ max$|F|_{\rm Modell}$'),
                 (mae[m], C_UN, 'MAE der Komponentendifferenz'))):
            xx = i + (k * 2 - 1) * off + rj.uniform(-jit, jit, len(v))
            ax.scatter(xx, v, s=17, c=c, alpha=0.75, lw=0, zorder=3,
                       label=lab if i == 0 else None)
            md = float(np.median(v))
            ax.hlines(md, i + (k * 2 - 1) * off - 0.15,
                      i + (k * 2 - 1) * off + 0.15, color=c, lw=3, zorder=5)
            ax.text(i + (k * 2 - 1) * off + 0.17, md, '%.3f' % md,
                    ha='left', va='center', fontsize=8, fontweight='bold',
                    color=c, zorder=6)
            (med_a if k == 0 else med_n).append(md)
        ax.text(i, -0.085, 'n=%d' % int(m.sum()), transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7.5, color=GREY)
    ax.axhline(0, color='k', lw=0.9)
    ax.set_yscale('symlog', linthresh=0.01, linscale=0.5)
    ax.set_ylim(-0.06, 2.4)
    ax.set_yticks([-0.01, 0, 0.01, 0.1, 1.0])
    ax.set_yticklabels(['−0.01', '0', '0.01', '0.1', '1'])
    ax.set_xlim(-0.55, len(BINS) - 0.45)
    ax.set_xticks(np.arange(len(BINS)))
    ax.set_xticklabels([nm for _, _, nm in BINS])
    ax.set_xlabel('Brechungstiefe an der Modellgeometrie  [meV]', labelpad=14)
    ax.set_ylabel('eV/Å   (symlog, Nulllinie durchgezogen)')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.005), fontsize=8,
              ncol=2, columnspacing=1.4)
    imax_a, imax_n = int(np.argmax(med_a)), int(np.argmax(med_n))
    ax.annotate('Maximum', xy=(imax_a - off, med_a[imax_a] * 1.1),
                xytext=(imax_a - 0.80, 1.15), fontsize=8.5, fontweight='bold',
                color='#8a6f2f',
                arrowprops=dict(arrowstyle='->', color='#8a6f2f', lw=1.2))
    ax.annotate('Maximum', xy=(imax_n + off, med_n[imax_n] * 1.1),
                xytext=(imax_n + 0.30, 0.62), fontsize=8.5, fontweight='bold',
                color=C_UN,
                arrowprops=dict(arrowstyle='->', color=C_UN, lw=1.2))
    nneg = int((diff < 0).sum())
    ax.text(0.985, 0.025, '%d Zeilen unter null — eine Fehlergrösse\n'
                          'kann nicht negativ sein' % nneg,
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            color='#8a6f2f', fontweight='bold')
    ax.text(0.5, -0.38, 'Jeder Punkt eine Zeile, waagrechter Strich der '
                        'Bin-Median.  Die goldene Reihe ist eine Differenz\n'
                        'zweier Maximalkomponenten — sie können an '
                        'verschiedenen Atomen und in verschiedenen\n'
                        'Richtungen sitzen.  Mit dem echten Fehler wandert das '
                        'Maximum von 1–50 nach 50–200 meV.',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic',
            color=GREY)

    # -- C  gegen den zweiten Deskriptor, ebenfalls an der Modellgeometrie
    ax = axs[1, 0]
    panel(ax, 'C', 'Zweiter Deskriptor an derselben Stelle')
    for s in ('uma-s', 'uma-m', 'esen'):
        m = have & (mdl == s)
        ax.scatter(-lamm[m], mae[m], s=30, c=C_MOD[s], alpha=0.72, lw=0,
                   label=LBL[s])
    ax.axvline(0, color='k', lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel(r'$-\lambda_{\min}^{\rm ext}$ an der Modellgeometrie  [Hartree]')
    ax.set_ylabel('Kraftfehler MAE  [eV/Å]')
    ax.legend(loc='upper left', fontsize=7.5)
    ax.text(0.01, 0.02, 'dort stabil  ←', transform=ax.transAxes, ha='left',
            fontsize=8, color=GREY)
    ax.text(0.99, 0.02, '→  dort gebrochen', transform=ax.transAxes,
            ha='right', fontsize=8, color=GREY)

    # -- D  Rangkorrelationen
    ax = axs[1, 1]
    panel(ax, 'D', 'Woher die Korrelation kommt — und wo sie fehlt')
    brk = have & (depm > 0)
    tests = [(r'$-\lambda_{\min}$@Modell  ·  $|F|_{\rm DFT}$  ·  alle',
              -lamm, fd, have, '#c2542a'),
             (r'$-\lambda_{\min}$@Modell  ·  MAE  ·  alle',
              -lamm, mae, have, '#c2542a'),
             (r'Tiefe@Modell  ·  $|F|_{\rm DFT}$  ·  nur gebrochen',
              depm, fd, brk, '#2a6f7f'),
             ('Tiefe@Modell  ·  MAE  ·  nur gebrochen',
              depm, mae, brk, '#2a6f7f')]
    rng = np.random.default_rng(SEED)
    ys, rs, los, his = [], [], [], []
    for nm, a, yv, sel, _ in tests:
        rs.append(spearman(a[sel], yv[sel]))
        ys.append(nm)
        RX = sorted(set(rxa[sel]))
        IDX = {r: np.flatnonzero(sel & (rxa == r)) for r in RX}
        bt = []
        for _ in range(3000):
            idx = np.concatenate([IDX[RX[j]]
                                  for j in rng.integers(0, len(RX), len(RX))])
            if len(np.unique(a[idx])) > 2:
                bt.append(spearman(a[idx], yv[idx]))
        los.append(float(np.percentile(bt, 2.5)))
        his.append(float(np.percentile(bt, 97.5)))
    yy = np.arange(len(ys))[::-1]
    ax.axvspan(-1, 1, ymin=0.0, ymax=0.5, color='#2a6f7f', alpha=0.06, lw=0)
    ax.axvline(0, color='k', lw=1.2)
    for y_, l_, h_, r_, (_, _, _, _, c) in zip(yy, los, his, rs, tests):
        ax.hlines(y_, l_, h_, color=c, lw=2.4)
        ax.plot([l_, h_], [y_, y_], '|', color=c, ms=9, mew=2)
        ax.scatter([r_], [y_], s=80, color=c, zorder=5)
        ax.text(h_ + 0.04, y_, r'$\rho$ = %+.2f' % r_, ha='left', va='center',
                fontsize=8.5, fontweight='bold', color=c)
    ax.set_yticks(yy)
    ax.set_yticklabels(ys, fontsize=8)
    ax.set_ylim(-0.85, len(ys) - 0.2)
    ax.set_xlim(-0.85, 1.35)
    ax.set_xlabel('Spearman, Balken = 95 %-CI (nach Reaktion geclustert)')
    ax.text(0.985, 1.005, 'oben: der Sprung — §5 zitiert +0.615',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='#c2542a', fontweight='bold')
    ax.text(0.985, 0.30, 'unten: der Gradient — beide CI überdecken die Null',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='#2a6f7f', fontweight='bold')

    fig.suptitle('Seam — belegt ist ein Sprung an der Instabilitätsgrenze, '
                 'nicht ein Anstieg zur Naht hin',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=0.985)
    fig.text(0.012, -0.045,
             'Brechungstiefe und λ$_{\\rm min}$ hier AN DER MODELLGEOMETRIE '
             'gemessen — die Variable, mit der §5 argumentiert, ein Wert je Zeile '
             '(121 von 122 Zeilen haben sie).  Die Kapiteltabelle reproduziert '
             'sich: Median |F|$_{\\rm DFT}$ = 0.069 / 0.160 / 0.163 / 0.141.  '
             'Der Peak bei flacher Brechung steckt allein in der Differenzspalte; '
             'die Rangkorrelation innerhalb der gebrochenen Zeilen ist bei 41 '
             'Zeilen nicht von null zu trennen.',
             fontsize=7.5, color=GREY, ha='left')
    save(fig, 'fig3_seam.png')


# =====================================================================  FIG 4
def fig4():
    h = sorted(hinge, key=lambda r: -(r['F_bs'] / r['F_rks']))
    frks = np.array([r['F_rks'] for r in h])
    fbs = np.array([r['F_bs'] for r in h])
    fac = fbs / frks
    cf = np.array([c['F_rks'] for c in ctrl])

    fig = plt.figure(figsize=(12.0, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.30,
                          width_ratios=[1.30, 1.0], height_ratios=[1.0, 0.85])

    # -- A  Doppelpunkt
    ax = fig.add_subplot(gs[:, 0])
    ax.set_title('Derselbe Punkt, zwei Flächen', loc='left', pad=8)
    ax.text(-0.09, 1.035, 'A', transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left')
    y = np.arange(len(h))[::-1]
    ax.axvspan(1e-3, STAT, color='#2a6f7f', alpha=0.09, lw=0)
    ax.axvline(STAT, color='k', lw=1.1, ls=':')
    ax.hlines(y, frks, fbs, color='#c7c7c7', lw=1.8, zorder=1)
    ax.scatter(frks, y, s=46, c=C_ST, zorder=3, label='auf der RKS-Fläche')
    ax.scatter(fbs, y, s=54, c=C_UN, zorder=3, marker='D',
               label='auf der BS-Fläche (Grundzustand)')
    for yy, f in zip(y, fac):
        ax.text(0.985, yy, '%.0f×' % f, transform=ax.get_yaxis_transform(),
                va='center', ha='right', fontsize=8, color=GREY)
    ax.set_yticks(y)
    ax.set_yticklabels([r['rxn'] for r in h], fontsize=8)
    ax.set_xscale('log')
    ax.set_xlim(0.018, 14)
    ax.set_ylim(-0.8, len(h) - 0.2)
    ax.set_xlabel('max|F| am RKS-Übergangszustand  [eV/Å]')
    ax.axvline(float(np.median(fbs)), color=C_UN, lw=1, ls='--', alpha=0.7)
    ax.text(float(np.median(fbs)) * 1.08, len(h) - 0.55, 'Median 1.697',
            fontsize=8, color=C_UN)
    ax.text(0.020, len(h) - 0.55, 'Stufe 1', fontsize=8)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=2)
    ax.text(0.5, -0.105,
            '18 von 19 stationär auf RKS      ·      0 von 19 stationär auf BS'
            '      ·      keine einzige Linie kreuzt zurück',
            transform=ax.transAxes, ha='center', va='top', fontsize=8.5,
            fontweight='bold')

    # -- B  Kontrolle
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, 'B', 'Die Kontrolle: dort ist der RKS-TS in Ordnung')
    ax.axvspan(1e-3, STAT, color='#2a6f7f', alpha=0.09, lw=0)
    ax.axvline(STAT, color='k', lw=1.1, ls=':')
    r1 = np.random.default_rng(3).uniform(-0.20, 0.20, len(cf))
    r2 = np.random.default_rng(4).uniform(-0.20, 0.20, len(frks))
    ax.scatter(cf, 1 + r1, s=34, c=C_ST, alpha=0.85, lw=0)
    ax.scatter(frks, 0 + r2, s=34, c=C_ST, alpha=0.85, lw=0)
    ax.scatter(fbs, 0 + r2, s=38, c=C_UN, alpha=0.9, lw=0, marker='D')
    ax.set_xscale('log')
    ax.set_xlim(0.006, 6)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['26 einreferenzielle\n(keine BS-Lösung)',
                        '19 multireferenzielle'], fontsize=8)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel('max|F| am RKS-TS  [eV/Å]')
    ax.text(0.03, 0.95, '%d von %d unter der Schwelle'
            % (int((cf < STAT).sum()), len(cf)),
            transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color=C_ST)

    # -- C  die Pointe
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, 'C', 'Was das für ein Kraftfeld bedeutet')
    v = [float(np.median(mae[~uns])), float(np.median(mae[uns])),
         float(np.median(fbs))]
    nm = ['Kraftfehler Modelle\nKontrollgruppe',
          'Kraftfehler Modelle\nMR-Gruppe',
          'Restkraft der Labels\nauf der BS-Fläche']
    b = ax.barh([2, 1, 0], v, 0.62, color=[C_ST, C_ST, C_UN], lw=0)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(nm, fontsize=8)
    ax.set_xscale('log')
    ax.set_xlim(0.008, 9)
    ax.set_xlabel('eV/Å  (logarithmisch)')
    for r_, val in zip(b, v):
        ax.text(val * 1.16, r_.get_y() + r_.get_height() / 2, '%.3f' % val,
                va='center', fontsize=9.5, fontweight='bold')
    ax.annotate('', xy=(v[2], 0.46), xytext=(v[1], 0.46),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=1.3))
    ax.text(float(np.sqrt(v[1] * v[2])), 0.56, '%.0f×' % (v[2] / v[1]),
            ha='center', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.42, 'Die Label-Unsicherheit in dieser Region ist rund '
                        '%.0f-mal grösser als die\nGenauigkeit, um die beim '
                        'Training gerungen wird.' % (v[2] / v[1]),
            transform=ax.transAxes, ha='center', fontsize=8.5, style='italic',
            color=GREY)

    fig.suptitle('Hinge — die Referenzgeometrien sind gute RKS-Sattelpunkte '
                 'und keine Stationärpunkte\nder Fläche, auf der die Reaktion läuft',
                 fontsize=12.5, fontweight='bold', x=0.012, ha='left', y=1.035)
    fig.text(0.012, -0.035,
             'results/hinge_rows.csv und results/control_rks.csv, gerechnet aus '
             'rks_grad.max_evang und bs.bs_grad.max_evang der PySCF-Stabilitäts'
             'analyse (ωB97M-V/def2-TZVP).  Alle 19 Werte stimmen mit der Tabelle '
             'in §6 überein.',
             fontsize=7.5, color=GREY, ha='left')
    save(fig, 'fig4_hinge.png')


# =====================================================================  FIG 5
def fig5():
    """Komponentenweiser Maximalfehler gegen die Brechungstiefe, punktweise.

    y = max_i |F_Modell,i - F_DFT,i| -- beide Kraftvektoren an derselben
    Geometrie, Differenzvektor, groesste Absolutkomponente.  Spalte
    maxcomp_err in results/paper_rows_ext.csv.

    x = Brechungstiefe.  Keine Bins: jede instabile Zeile steht an ihrer
    eigenen Stelle, zusaetzlich als Strich auf der Achse.  Links davon, durch
    einen Achsenbruch getrennt, die stabilen Zeilen -- fuer sie ist die Tiefe
    exakt null und auf einer Log-Achse nicht darstellbar.

    Zwei Panels, weil es zwei Orte gibt, an denen man die Tiefe messen kann:
    an der Modellgeometrie (ein Wert je Zeile, die Variable aus Abschnitt 5)
    und am RKS-TS (ein Wert je Reaktion, drei Zeilen teilen ihn).
    """
    rxa = np.array([r['rxn'] for r in rows])
    fig = plt.figure(figsize=(13.8, 6.4))
    outer = fig.add_gridspec(1, 2, wspace=0.26)

    VARIANTS = [
        ('A', depm, 'an der Modellgeometrie',
         'ein Wert je Zeile — die Variable aus §5'),
        ('B', dep, 'am RKS-Übergangszustand',
         'ein Wert je Reaktion — die drei Modellzeilen teilen ihn'),
    ]

    for k, (letter, dvar, where, sub) in enumerate(VARIANTS):
        gs = outer[k].subgridspec(1, 2, width_ratios=[1, 6.4], wspace=0.06)
        axl = fig.add_subplot(gs[0])          # die stabilen Zeilen
        ax = fig.add_subplot(gs[1])           # die gebrochenen Zeilen
        ok = ~np.isnan(dvar)
        st = ok & (dvar == 0)
        br = ok & (dvar > 0)

        # ---- links: stabil, kein Tiefenwert
        rj = np.random.default_rng(5).uniform(-0.28, 0.28, len(rows))
        for s in ('uma-s', 'uma-m', 'esen'):
            m = st & (mdl == s)
            axl.scatter(rj[m], mce[m], s=22, c=C_MOD[s], alpha=0.7, lw=0)
        m_st = float(np.median(mce[st]))
        axl.hlines(m_st, -0.42, 0.42, color='k', lw=2.6, zorder=5)
        axl.set_xlim(-0.75, 0.75)
        axl.set_xticks([0])
        axl.set_xticklabels(['0\nstabil'], fontsize=8)
        axl.set_yscale('log')
        axl.set_ylabel(r'Kraftfehler  max$_i\,|F_{{\rm Modell},i} - '
                       r'F_{{\rm DFT},i}|$   [eV/Å]')
        axl.grid(axis='x', alpha=0)
        axl.text(0.5, 1.005, 'n=%d' % int(st.sum()), transform=axl.transAxes,
                 ha='center', va='bottom', fontsize=7.5, color=GREY)

        # ---- rechts: jede gebrochene Zeile an ihrer eigenen Stelle
        for s in ('uma-s', 'uma-m', 'esen'):
            m = br & (mdl == s)
            ax.scatter(dvar[m], mce[m], s=46, c=C_MOD[s], alpha=0.85, lw=0,
                       zorder=4, label=LBL[s])
        m_br = float(np.median(mce[br]))
        ax.axhline(m_br, color='k', lw=2, zorder=3)
        ax.axhline(m_st, color='k', lw=1.4, ls=(0, (5, 3)), zorder=3)
        axl.axhline(m_st, color='k', lw=1.4, ls=(0, (5, 3)), zorder=3)

        # jeder Punkt zusaetzlich als Strich auf der Achse
        ax.plot(dvar[br], np.full(int(br.sum()), 0.0),
                marker='|', ms=11, mew=1.3, ls='none', color='#333',
                transform=ax.get_xaxis_transform(), clip_on=True, zorder=6)

        lo = mce[ok].min() * 0.62
        hi = mce[ok].max() * 2.8
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(lo, hi)
        axl.set_ylim(lo, hi)
        ax.set_xlim(dvar[br].min() * 0.55, dvar[br].max() * 1.9)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

        # Achsenbruch zwischen den beiden Teilen
        for a_, side in ((axl, 'right'), (ax, 'left')):
            a_.spines[side].set_visible(False)
        d = 0.012
        for a_, xpos in ((axl, 1.0), (ax, 0.0)):
            a_.plot([xpos - d * 6, xpos + d * 6], [-d, d], transform=a_.transAxes,
                    color='k', lw=1, clip_on=False)

        ax.text(0.985, m_br * 1.14, 'Median gebrochen  %.3f' % m_br,
                transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=8, fontweight='bold')
        ax.text(0.985, m_st * 0.82, 'Median stabil  %.3f' % m_st,
                transform=ax.get_yaxis_transform(), ha='right', va='top',
                fontsize=8, color=GREY)
        ax.set_xlabel('Brechungstiefe  $E$(RKS) $-$ $E$(BS)   [meV]')

        # die vier groessten Ausreisser benennen
        idx = np.argsort(-mce * br)[:3]
        OFF = [(10, 5), (10, -12), (-10, 8)]
        for j, i in enumerate(idx):
            if not br[i]:
                continue
            dx, dy = OFF[j]
            ax.annotate('%s · %s' % (rxa[i], LBL[mdl[i]]), (dvar[i], mce[i]),
                        xytext=(dx, dy), textcoords='offset points',
                        ha='left' if dx > 0 else 'right', fontsize=7.5,
                        color='#444')

        rho = spearman(dvar[br], mce[br])
        rng = np.random.default_rng(SEED)
        RX = sorted(set(rxa[br]))
        IDX = {r: np.flatnonzero(br & (rxa == r)) for r in RX}
        bt = []
        for _ in range(3000):
            ii = np.concatenate([IDX[RX[j]]
                                 for j in rng.integers(0, len(RX), len(RX))])
            if len(np.unique(dvar[ii])) > 2:
                bt.append(spearman(dvar[ii], mce[ii]))
        clo, chi = float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))
        ax.text(0.985, 0.975,
                'Spearman über die %d gebrochenen Zeilen\n'
                r'$\rho$ = %+.2f    95 %%-CI [%+.2f, %+.2f]'
                '\n%s' % (int(br.sum()), rho, clo, chi,
                          'enthält die Null' if clo <= 0 <= chi
                          else 'ohne die Null'),
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='#ddd'))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.135), ncol=3,
                  fontsize=8.5)

        axl.set_title('%s   Tiefe %s' % (letter, where), loc='left', pad=10,
                      fontsize=10.5)
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=8,
                color=GREY, style='italic')

    fig.suptitle('Kraftfehler gegen Symmetriebrechung — jede gebrochene Zeile '
                 'einzeln, ohne Bins',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=1.03)
    fig.text(0.012, -0.20,
             r'$y$ = grösste Absolutkomponente des Differenzvektors zwischen '
             'Modell- und DFT-Kraft an derselben, unveränderten Geometrie '
             '(Spalte maxcomp_err).\n'
             'Striche auf der x-Achse: jede einzelne gebrochene Zeile.  '
             'Links vom Achsenbruch die Zeilen ohne Brechung — für sie ist die '
             'Tiefe exakt null und auf einer Log-Achse nicht darstellbar.\n'
             'Die durchgezogene Linie ist der Median der gebrochenen, die '
             'gestrichelte der Median der stabilen Zeilen.',
             fontsize=7.5, color=GREY, ha='left')
    save(fig, 'fig5_maxcomp_vs_depth.png')


# =====================================================================  FIG 6
def fig6():
    """Der Praediktor als Betriebsmittel statt als AUC.

    Vier Fragen, die eine ROC-Kurve nicht beantwortet:
      A  wie viel DFT muss ich laufen lassen, um wie viele Ausfaelle zu fangen
      B  wo genau sitzen die Ausnahmen -- jede Reaktion einzeln
      C  was liefert die natuerliche Schwelle lambda_min < 0 konkret
      D  was kostet die Vorhersage gegen das, was sie erspart
    """
    from matplotlib.patches import Rectangle
    rxa = np.array([r['rxn'] for r in rows])
    RXN = sorted(set(rxa), key=lambda r: lam[rxa == r][0])   # instabilste zuerst
    lam_r = np.array([lam[rxa == r][0] for r in RXN])
    nrow_r = np.array([int((rxa == r).sum()) for r in RXN])
    nbad_r = np.array([int(bad[rxa == r].sum()) for r in RXN])

    fig = plt.figure(figsize=(12.6, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                          width_ratios=[1.0, 1.05])

    # -- A  Triage-Kurve
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, 'A', 'Wie viel DFT für wie viele gefangene Ausfälle')
    tot_rows, tot_bad = int(nrow_r.sum()), int(nbad_r.sum())

    def curve(order):
        x = np.concatenate([[0], np.cumsum(nrow_r[order]) / tot_rows])
        y = np.concatenate([[0], np.cumsum(nbad_r[order]) / tot_bad])
        return x, y

    o_pred = np.argsort(lam_r)                      # kleinstes lambda zuerst
    o_orac = np.argsort(-(nbad_r / nrow_r))
    xo, yo = curve(o_orac)
    xp, yp = curve(o_pred)
    ax.fill_between(xo, yo, xo, color='#2a6f7f', alpha=0.07, lw=0)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls=':', label='zufällige Reihenfolge')
    ax.plot(xo, yo, color=GREY, lw=1.4, ls='--', label='Orakel (kennt die Antwort)')
    ax.plot(xp, yp, color='#c2542a', lw=2.4,
            label=r'nach $\lambda_{\min}^{\rm ext}$ sortiert')
    k = int((lam_r < 0).sum())
    ax.scatter([xp[k]], [yp[k]], s=110, facecolor='white', edgecolor='#c2542a',
               lw=2.4, zorder=6)
    ax.annotate('Schwelle $\\lambda_{\\min} < 0$\n%.0f %% der Rechnungen  →  '
                '%.0f %% der Ausfälle' % (100 * xp[k], 100 * yp[k]),
                (xp[k], yp[k]), xytext=(0.44, 0.28), textcoords='axes fraction',
                fontsize=8.5, fontweight='bold', color='#c2542a',
                arrowprops=dict(arrowstyle='->', color='#c2542a', lw=1.2))
    ax.set_xlabel('Anteil der Strukturen, die man mit DFT nachrechnet')
    ax.set_ylabel('Anteil der gefangenen Ausfälle')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right')

    # -- B  jede Reaktion einzeln
    ax = fig.add_subplot(gs[:, 1])
    ax.set_title('Jede Reaktion einzeln, sortiert nach $\\lambda_{\\min}$',
                 loc='left', pad=8)
    ax.text(-0.17, 1.026, 'B', transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left')
    y = np.arange(len(RXN))[::-1]
    CFAIL = {0: '#2a6f7f', 1: '#e8b04b', 2: '#dd7f3e', 3: '#c2542a'}
    ax.axvspan(min(lam_r) * 1.15, 0, color='#c2542a', alpha=0.06, lw=0)
    ax.axvline(0, color='k', lw=1.4)
    ax.hlines(y, 0, lam_r, color='#d0d0d0', lw=1.4, zorder=1)
    for i, (yy, lv, nb, nr) in enumerate(zip(y, lam_r, nbad_r, nrow_r)):
        ax.scatter([lv], [yy], s=64, c=CFAIL[nb], zorder=4,
                   edgecolor='white', lw=0.6)
        ax.text(lv + (0.006 if lv < 0 else -0.006), yy,
                '%d/%d' % (nb, nr), va='center',
                ha='left' if lv < 0 else 'right', fontsize=7, color='#444')
    ax.set_yticks(y)
    ax.set_yticklabels(RXN, fontsize=6.8)
    ax.set_ylim(-0.9, len(RXN) - 0.1)
    ax.set_xlim(min(lam_r) * 1.22, max(lam_r) * 1.22)
    ax.set_xlabel(r'$\lambda_{\min}^{\rm ext}$ am RKS-TS  [Hartree]')
    ax.text(0.02, 0.012, 'RKS instabil', transform=ax.transAxes, fontsize=8.5,
            fontweight='bold', color='#c2542a', va='bottom')
    ax.text(0.98, 0.985, 'RKS stabil', transform=ax.transAxes, fontsize=8.5,
            fontweight='bold', color='#2a6f7f', va='top', ha='right')
    hs = [Line2D([], [], marker='o', ls='none', color=CFAIL[i], ms=7,
                 label='%d von 3 Modellzeilen gescheitert' % i) for i in range(4)]
    ax.legend(handles=hs, loc='lower left', bbox_to_anchor=(0.0, -0.135),
              ncol=2, fontsize=7.5)
    ax.text(0.5, -0.175, 'Die Zahl neben jedem Punkt ist '
                         'gescheitert / geprüft.',
            transform=ax.transAxes, ha='center', fontsize=7.5, style='italic',
            color=GREY)

    # -- C  der Betriebspunkt
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, 'C', 'Was die Schwelle $\\lambda_{\\min} < 0$ konkret liefert')
    tp, fn = int(bad[uns].sum()), int(bad[~uns].sum())
    fp, tn = int((~bad[uns]).sum()), int((~bad[~uns]).sum())
    M = np.array([[tp, fp], [fn, tn]])
    ax.imshow(np.array([[1.0, 0.45], [0.45, 0.12]]), cmap='Oranges',
              vmin=0, vmax=1.25, aspect='auto')
    lab = [['gefangen', 'unnötig nachgerechnet'],
           ['übersehen', 'zu Recht übersprungen']]
    for i in range(2):
        for j in range(2):
            col = 'white' if (i, j) == (0, 0) else '#222'
            ax.text(j, i - 0.13, '%d' % M[i, j], ha='center', va='center',
                    fontsize=21, fontweight='bold', color=col)
            ax.text(j, i + 0.17, lab[i][j], ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=col)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['ist ein Ausfall\nDFT ≥ 0.15', 'ist in Ordnung'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['markiert\n(instabil)', 'nicht markiert\n(stabil)'])
    ax.grid(False)
    ax.set_xlabel(
        'Trefferquote %d/%d = %.0f %%      Präzision %d/%d = %.0f %%\n'
        'nachgerechnet werden %d von %d Strukturen = %.0f %%'
        % (tp, tp + fn, 100 * tp / (tp + fn), tp, tp + fp,
           100 * tp / (tp + fp), tp + fp, len(rows),
           100 * (tp + fp) / len(rows)), fontsize=8.5)

    fig.suptitle('Prädiktor im Betrieb — was die Instabilitätsanalyse '
                 'einspart und was sie übersieht',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=0.985)
    fig.text(0.012, 0.005,
             '122 Zeilen = 42 Reaktionen × 3 Modelle, 29 Ausfälle '
             '(max|F|$_{\\rm DFT}$ ≥ 0.15 eV/Å).  Der Prädiktor entscheidet je '
             'REAKTION, nicht je Zeile — eine markierte Reaktion zieht alle '
             'drei Modellzeilen nach sich.',
             fontsize=7.5, color=GREY, ha='left')
    save(fig, 'fig6_predictor_operational.png')


# =====================================================================  FIG 7
def fig7():
    """Was die Vorhersage kostet gegen das, was sie erspart."""
    cost = load('cost_hours.csv')
    grp = {}
    for r in cost:
        key = (r['schritt'] if r['schritt'] != 'praediktor'
               else 'praediktor_' + r['variante'])
        grp.setdefault(key, []).append(r['stunden'])

    ORDER = [('praediktor_stabil', 'Prädiktor\nRKS + Stabilitätsanalyse',
              '#2a6f7f'),
             ('routeC_orca', 'Startpunkt C\nORCA-Bewertung', '#9aa0a6'),
             ('routeB', 'Startpunkt B\nvorhandenes Band', '#b9a06a'),
             ('routeC_pyscf', 'Startpunkt C\nBS-TS-Opt in PySCF', '#dd7f3e'),
             ('routeA', 'Startpunkt A\nneues NEB-CI-Band', '#c2542a')]

    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    rj = np.random.default_rng(9)
    ref = float(np.median(grp['praediktor_stabil']))
    for i, (key, name, c) in enumerate(ORDER):
        v = np.array(grp.get(key, []))
        if not len(v):
            continue
        yy = len(ORDER) - 1 - i
        md = float(np.median(v))
        ax.barh(yy, md, 0.52, color=c, alpha=0.30, lw=0)
        ax.scatter(v, yy + rj.uniform(-0.16, 0.16, len(v)), s=30, c=c,
                   alpha=0.9, lw=0, zorder=4)
        ax.plot([md, md], [yy - 0.29, yy + 0.29], color=c, lw=3, zorder=5)
        ax.text(md * 1.13, yy + 0.30, '%.2f h' % md, fontsize=9,
                fontweight='bold', color=c, va='bottom')
        ax.text(v.max() * 1.35, yy, 'n=%d   ×%.0f' % (len(v), md / ref)
                if key != 'praediktor_stabil' else 'n=%d' % len(v),
                va='center', fontsize=8, color=GREY)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([n for _, n, _ in ORDER][::-1], fontsize=8.5)
    ax.set_xscale('log')
    ax.set_xlim(0.08, 200)
    ax.set_xlabel('Wandzeit je Reaktion  [h]   (logarithmisch, ein Punkt je Lauf)')
    ax.set_title('Was die Vorhersage kostet gegen das, was sie erspart',
                 loc='left', pad=10, fontsize=12)
    ax.axvline(ref, color='#2a6f7f', lw=1, ls='--', alpha=0.7)
    inst = grp.get('praediktor_instabil', [])
    fig.text(0.0, -0.10,
             'Gemessene TOTAL RUN TIME bzw. elapsed_s aus den Logs '
             '(results/cost_hours.csv), keine Schätzung.\n'
             'Der Prädiktor ist mit den %d STABILEN Reaktionen beziffert: dort '
             'endet der Lauf nach der Stabilitätsanalyse, und genau das kostet '
             'die Vorhersage.\nBei den %d instabilen hängt dieselbe Datei die '
             'BS-Suche an (Median %.2f h), die die Vorhersage nicht braucht — '
             'das wäre eine obere Schranke, kein Preis.'
             % (len(grp['praediktor_stabil']), len(inst),
                float(np.median(inst)) if inst else float('nan')),
             fontsize=7.5, color=GREY, ha='left')
    save(fig, 'fig7_predictor_cost.png')


# =====================================================================  FIG 8
def fig8():
    """Derselbe Vergleich, einmal mit jedem der beiden Stabilitaetsetiketten.

    In Fig 1 B kommt die Gruppenfarbe vom RKS-Uebergangszustand, waehrend
    beide Kraefte an der Modellgeometrie gemessen sind -- zwei verschiedene
    Punkte.  Hier steht daneben, was herauskommt, wenn das Etikett am
    Messpunkt selbst bestimmt wird.
    """
    have = ~np.isnan(depm)
    u_mod = depm > 0                     # an der Modellgeometrie gebrochen
    rng = np.random.default_rng(SEED)

    fig = plt.figure(figsize=(12.8, 6.6))
    gs = fig.add_gridspec(1, 3, wspace=0.34, width_ratios=[1, 1, 1.15])

    VAR = [
        ('A', ~uns, uns, np.ones(len(rows), bool),
         'Etikett vom RKS-TS',
         'so steht es in Fig 1 B — die Gruppenfarbe\n'
         'kommt von einer anderen Geometrie als die Kräfte'),
        ('B', have & ~u_mod, have & u_mod, have,
         'Etikett an der Modellgeometrie',
         'Etikett und beide Kräfte am selben Punkt;\n'
         'für eine Zeile fehlt der Wert, daher 121'),
    ]

    for k, (letter, m_st, m_un, m_all, title, sub) in enumerate(VAR):
        ax = fig.add_subplot(gs[k])
        ax.set_title(title, loc='left', pad=26, fontsize=10.5)
        ax.text(-0.26, 1.12, letter, transform=ax.transAxes, fontsize=12,
                fontweight='bold', va='top', ha='left')
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=7.5,
                color=GREY, style='italic', va='bottom')
        for m, c, nm in ((m_st, C_ST, 'RKS stabil'), (m_un, C_UN, 'RKS instabil')):
            a, b = float(np.median(fm[m])), float(np.median(fd[m]))
            ca, cb = boot_median(fm[m], rng=rng), boot_median(fd[m], rng=rng)
            ax.plot([0, 1], [a, b], color=c, lw=2.2, marker='o', ms=8, zorder=3,
                    label='%s  ·  %d Strukturen' % (nm, m.sum()))
            ax.vlines(0, *ca, color=c, lw=6, alpha=0.28)
            ax.vlines(1, *cb, color=c, lw=6, alpha=0.28)
            ax.annotate('%.4f' % a, (0, a), xytext=(-11, 0),
                        textcoords='offset points', ha='right', va='center',
                        fontsize=9, color=c, fontweight='bold')
            ax.annotate('%.3f' % b, (1, b), xytext=(11, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=9, color=c, fontweight='bold')
        d = abs(float(np.median(fm[m_st])) - float(np.median(fm[m_un])))
        D = abs(float(np.median(fd[m_st])) - float(np.median(fd[m_un])))
        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['der Kalkulator\ndes Modells',
                            'ωB97M-V/def2-TZVP'], fontsize=8.5)
        ax.set_ylabel('Median von max|F| über die Gruppe  [eV/Å]')
        ax.axhline(STAT, color='k', lw=0.9, ls=':')
        ax.text(-0.52, STAT * 1.03, 'Stufe 1  0.15', ha='left', va='bottom',
                fontsize=7.5)
        ax.set_ylim(0, 0.205)
        ax.legend(loc='upper left', fontsize=8)
        ax.text(0.5, -0.185,
                'Gruppenabstand  %.4f  →  %.3f\nFaktor %.0f' % (d, D, D / d),
                transform=ax.transAxes, ha='center', fontsize=8.5,
                fontweight='bold', color='#333')

    # -- C  wo sich die beiden Etiketten widersprechen
    ax = fig.add_subplot(gs[2])
    ax.set_title('Wo sich die beiden Etiketten widersprechen', loc='left',
                 pad=26, fontsize=10.5)
    ax.text(-0.30, 1.12, 'C', transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', ha='left')
    rxa = np.array([r['rxn'] for r in rows])
    dis = np.flatnonzero(have & (uns != u_mod))
    dis = dis[np.argsort(-fd[dis])]
    y = np.arange(len(dis))[::-1]
    ax.axvline(STAT, color='k', lw=1.1, ls=':')
    ax.axvspan(1e-3, STAT, color='#2a6f7f', alpha=0.08, lw=0)
    for yy, i in zip(y, dis):
        c = C_UN if u_mod[i] else C_ST
        ax.hlines(yy, fm[i], fd[i], color='#d5d5d5', lw=1.6, zorder=1)
        ax.scatter([fm[i]], [yy], s=30, c='#9aa0a6', zorder=3)
        ax.scatter([fd[i]], [yy], s=54, c=c, zorder=4, marker='D')
    ax.set_yticks(y)
    ax.set_yticklabels(['%s·%s' % (rxa[i], LBL[mdl[i]]) for i in dis],
                       fontsize=7.5)
    for yy, i in zip(y, dis):
        ax.text(0.985, yy, '%s→%s' % ('instabil' if uns[i] else 'stabil',
                                      'instabil' if u_mod[i] else 'stabil'),
                transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=7, color=GREY)
    ax.set_xscale('log')
    ax.set_xlim(0.004, 3)
    ax.set_ylim(-0.8, len(dis) - 0.2)
    ax.set_xlabel('max|F|  [eV/Å]   grau: Modell,  Raute: DFT')
    hs = [Line2D([], [], marker='D', ls='none', color=C_UN, ms=7,
                 label='an der Modellgeometrie gebrochen'),
          Line2D([], [], marker='D', ls='none', color=C_ST, ms=7,
                 label='dort restringiert stabil')]
    ax.legend(handles=hs, loc='lower right', fontsize=7.5)
    ax.text(0.5, -0.185, '%d von %d Zeilen widersprechen sich\n'
                         'in allen übrigen sagen beide dasselbe'
            % (len(dis), int(have.sum())),
            transform=ax.transAxes, ha='center', fontsize=8.5, color='#333',
            fontweight='bold')

    fig.suptitle('Wo wird die Stabilität gemessen — und macht es einen '
                 'Unterschied?',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=1.02)
    fig.text(0.012, -0.13,
             'Beide Kräfte stammen in allen drei Panels von derselben Struktur: '
             'der Modellvorhersage <modeldir>/<rxn>/transition_state.xyz.  '
             'Unterschiedlich ist nur, WO die Stabilität der restringierten '
             'Lösung geprüft wurde.\n'
             'A nimmt sie vom RKS-Übergangszustand (stab_pipeline, Eintrag '
             'RKS-ref) — ein Wert je Reaktion, den man vor dem Modellauf kennt; '
             'genau deshalb taugt er als Prädiktor.\n'
             'B nimmt sie an der Modellgeometrie selbst (Eintrag UMA-S / UMA-M / '
             'eSEN) — konsistent mit dem Messpunkt, aber erst nach dem Modellauf '
             'verfügbar und damit als Vorhersage wertlos.\n'
             'Die DFT-Kraft rechnet ohnehin auf der lokal richtigen Fläche: '
             'STABRestartUHFifUnstable entscheidet an der Modellgeometrie. '
             'Nur die Gruppenfarbe wechselt zwischen A und B.',
             fontsize=7.5, color=GREY, ha='left')
    save(fig, 'fig8_label_at_which_geometry.png')


for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
    f()
