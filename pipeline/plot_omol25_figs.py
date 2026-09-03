"""Figures built only from the OMol25-level run.

One recipe, one level of theory, one geometry per row:

    ! UKS wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3
    %scf STABPerform true / STABRestartUHFifUnstable true
         Thresh 1e-12 / TCut 1e-13 end

Everything in these figures comes out of that single point and the EnGrad that
reads its orbitals: the RKS stable / unstable label from <S^2>, the DFT force
from the gradient, the energies from the three structures. Nothing is borrowed
from another level, another geometry or another code.

Data: results/omol25_model_geoms.csv  (siehe die gleichnamige .md daneben)
"""
import csv
import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
S2_BREAK = 0.05          # frei: <S^2> ist 0 oder >= 0.0579, nichts dazwischen
CINEB = 0.05             # the model NEBs' own stopping criterion, eV/A

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

rows = [r for r in csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv')))
        if r['f_dft_max'] != '' and r['s2_ts'] != '']
fm = np.array([float(r['f_model_max']) for r in rows])
fd = np.array([float(r['f_dft_max']) for r in rows])
s2 = np.abs(np.array([float(r['s2_ts']) for r in rows]))
mdl = np.array([r['model'] for r in rows])
brk = s2 > S2_BREAK
S2MIN = float(s2[s2 > 0].min())    # kleinster <S^2> ueber null

print('%d rows   %d RKS stable   %d RKS unstable' % (len(rows), (~brk).sum(), brk.sum()))


def fig_silent():
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 5.0), sharex=True, sharey=True)
    lo = min(fm.min(), fd.min()) * 0.55
    hi = max(fm.max(), fd.max()) * 2.2

    for ax, m in zip(axs, ('uma-s', 'uma-m', 'esen')):
        sel = mdl == m
        s, u = sel & ~brk, sel & brk
        ax.plot([lo, hi], [lo, hi], color='#444', lw=1.2, ls='--', zorder=2)
        ax.axvline(CINEB, color=GREY, lw=0.9, ls=':', zorder=1)
        ax.scatter(fm[s], fd[s], s=34, c=C_ST, alpha=0.85, lw=0, zorder=4,
                   label='RKS stable   ⟨S²⟩ = 0        (n=%d)' % s.sum())
        ax.scatter(fm[u], fd[u], s=40, c=C_UN, alpha=0.85, lw=0, marker='D',
                   zorder=4, label='RKS unstable  ⟨S²⟩ > 0   (n=%d)' % u.sum())

        # group medians as short crosshairs on the axes
        for mm, c in ((s, C_ST), (u, C_UN)):
            ax.plot([np.median(fm[mm])], [np.median(fd[mm])], marker='+',
                    ms=17, mew=2.6, color=c, zorder=6)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(LBL[m], loc='left', pad=8)
        ax.set_xlabel('|F| the model reports   [eV/Å]')
        ax.legend(loc='upper left', fontsize=7.5)

        txt = ('median, RKS stable      %.3f → %.3f   ×%.1f\n'
               'median, RKS unstable  %.3f → %.3f   ×%.1f'
               % (np.median(fm[s]), np.median(fd[s]),
                  np.median(fd[s]) / np.median(fm[s]),
                  np.median(fm[u]), np.median(fd[u]),
                  np.median(fd[u]) / np.median(fm[u])))
        ax.text(0.985, 0.025, txt, transform=ax.transAxes, ha='right',
                va='bottom', fontsize=7.5, family='DejaVu Sans Mono',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ddd'))

    axs[0].set_ylabel('|F| from DFT at the identical geometry   [eV/Å]')
    axs[0].annotate('agreement', xy=(0.06, 0.06), xytext=(0.02, 0.30),
                    fontsize=8, color='#444', rotation=0,
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1))
    axs[0].text(CINEB * 0.85, lo * 1.5, 'NEB stopping criterion 0.05',
                rotation=90, ha='right', va='bottom', fontsize=7.5, color=GREY)

    fig.suptitle('Silent failure — the model reports a small residual force '
                 'wherever it stops, and DFT does not agree',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=1.02)
    fig.text(0.012, -0.10,
             'Every quantity from one calculation per structure: '
             'ωB97M-V/def2-TZVPD, DEFGRID3, Thresh 1e-12, RIJCOSX, ORCA 5.0.4. '
             'The DFT force is an EnGrad reading the orbitals of that same '
             'single point, so it sits on whichever surface STABPerform '
             'selected.\n'
             'RKS stable / unstable is ⟨S²⟩ of that same single point at the '
             'same geometry — not borrowed from the reference structure. '
             '⟨S²⟩ is exactly 0 in ' + str(int((~brk).sum())) + ' rows '
             'and at least ' + ('%.4f' % S2MIN) + ' in '
             + str(int(brk.sum())) + '; nothing lies '
             'in between, so the split needs no chosen threshold.\n'
             'Dashed line: |F|$_{\\rm DFT}$ = |F|$_{\\rm model}$. The vertical '
             'distance from it is the model\'s error. Crosses mark the group '
             'medians.',
             fontsize=7.5, color=GREY, ha='left')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_silent_failure_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


def fig_silent_v2():
    """Figure 9.1 — same content as fig 9, without the stopping-criterion line
    and without the median box, and with the plotted quantity stated exactly."""
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 5.2), sharex=True, sharey=True)
    lo = min(fm.min(), fd.min()) * 0.55
    hi = max(fm.max(), fd.max()) * 2.2
    med = []

    for ax, m in zip(axs, ('uma-s', 'uma-m', 'esen')):
        sel = mdl == m
        s, u = sel & ~brk, sel & brk
        # the diagonal goes into the legend, so no floating text can collide
        ax.plot([lo, hi], [lo, hi], color='#444', lw=1.2, ls='--', zorder=2,
                label='MLIP = DFT')
        ax.scatter(fm[s], fd[s], s=34, c=C_ST, alpha=0.85, lw=0, zorder=4,
                   label=r'RKS stable,  $\langle S^2\rangle = 0$   (n=%d)' % s.sum())
        ax.scatter(fm[u], fd[u], s=40, c=C_UN, alpha=0.85, lw=0, marker='D',
                   zorder=4,
                   label=r'RKS unstable,  $\langle S^2\rangle > 0$   (n=%d)' % u.sum())
        # medians: filled plus, white separation ring, thin dark outline — the
        # plain '+' disappeared inside the point cloud
        for mm, c in ((s, C_ST), (u, C_UN)):
            x0, y0 = np.median(fm[mm]), np.median(fd[mm])
            # thin open cross: white underlay for contrast, nothing filled in,
            # so the points behind it stay visible
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
        # lower right is empty in every panel: everything sits on or above the
        # diagonal, so the legend cannot cover a data point there
        from matplotlib.lines import Line2D
        h, l = ax.get_legend_handles_labels()
        h.append(Line2D([], [], marker='+', ms=13, mec='#6b6b6b', mew=1.8,
                        ls='none'))
        l.append('group median')
        lg = ax.legend(h, l, loc='lower right', fontsize=7.5, frameon=True,
                       framealpha=0.95, edgecolor='#ddd', borderpad=0.5)
        lg.get_frame().set_facecolor('white')
        lg.get_frame().set_linewidth(0.6)

    axs[0].set_ylabel(r'$\max_i |F_i^{\,\mathrm{DFT}}|$  at the identical'
                      '\n'
                      r'geometry   [eV Å$^{-1}$]')

    tab = ' · '.join('%s %.3f/%.3f, %.3f/%.3f' % x for x in med)
    fig.suptitle('Silent failure — the residual force an MLIP reports at its own '
                 'transition state does not track the force that is there',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=1.03)
    fig.text(0.012, -0.30,
             r'$F_i$ denotes a Cartesian force component and $\max_i$ runs over '
             r'all $3N$ of them. This is neither a per-atom norm nor a'
             '\nroot-mean-square; over these structures the two conventions '
             'differ by a measured factor of 0.61 to 1.00.\n\n'
             'Both axes are evaluated at the same, unrelaxed geometry — the '
             'transition state each MLIP produced in its own NEB.\n'
             'The DFT side is ωB97M-V/def2-TZVPD, DEFGRID3, Thresh 1e-12, '
             'TCut 1e-13, RIJCOSX, ORCA 5.0.4. The gradient is an EnGrad\n'
             'reading the converged orbitals of a preceding single point run '
             'with STABPerform, so it is evaluated on the\nelectronic ground '
             'state at that geometry.\n\n'
             r'The RKS stable / unstable classification is $\langle S^2\rangle$ '
             'of that same single point: exactly 0 for '
             + str(int((~brk).sum())) + ' of the ' + str(len(rows)) + '\n'
             'structures and at least ' + ('%.4f' % S2MIN) + ' for the '
             'remaining ' + str(int(brk.sum())) + ', with no value in '
             'between, so the two classes separate\nwithout a chosen cut-off.\n\n'
             'Crosses mark group medians as MLIP/DFT, stable then unstable — '
             + tab + '.',
             fontsize=7.5, color=GREY, ha='left')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_1_silent_failure_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


def fig_slope(loud=False):
    """Figure 9.2 — group medians of the same two quantities, one panel per
    MLIP. Shows that the two classes are indistinguishable on the MLIP side and
    separated on the DFT side."""
    rng = np.random.default_rng(20260819)

    def ci(v, n=10000):
        v = np.asarray(v, float)
        b = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    fig, axs = plt.subplots(1, 3, figsize=(11.4, 5.4), sharey=True)
    jit = np.random.default_rng(5)
    for ax, m in zip(axs, ('uma-s', 'uma-m', 'esen')):
        sel = mdl == m
        for grp, c, nm in ((sel & ~brk, C_ST, 'RKS stable'),
                           (sel & brk, C_UN, 'RKS unstable')):
            # every row shown, so the reader sees where the median comes from
            sp, al, ew = (34, 0.62, 0.5) if loud else (13, 0.30, 0.0)
            for x0, v in ((0, fm[grp]), (1, fd[grp])):
                ax.scatter(x0 + jit.uniform(-0.16, 0.16, len(v)) if loud
                           else x0 + jit.uniform(-0.085, 0.085, len(v)), v,
                           s=sp, c=c, alpha=al, lw=ew, edgecolor='white',
                           zorder=1)
            a, b = float(np.median(fm[grp])), float(np.median(fd[grp]))
            ca, cb = ci(fm[grp]), ci(fd[grp])
            bw, ba = (4, 0.22) if loud else (6, 0.35)
            ax.vlines(0, *ca, color=c, lw=bw, alpha=ba, zorder=2)
            ax.vlines(1, *cb, color=c, lw=bw, alpha=ba, zorder=2)
            ax.plot([0, 1], [a, b], color=c, lw=2.2, marker='o', ms=8, zorder=3,
                    label=r'%s,  $\langle S^2\rangle %s 0$   (n=%d)'
                          % (nm, '=' if c == C_ST else '>', grp.sum()))
            ax.annotate('%.3f' % a, (0, a), xytext=(-11, 0),
                        textcoords='offset points', ha='right', va='center',
                        fontsize=8.5, color=c, fontweight='bold')
            ax.annotate('%.3f' % b, (1, b), xytext=(11, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=8.5, color=c, fontweight='bold')
        ax.set_xlim(-0.70, 1.70)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['predicted by\nthe MLIP',
                            'ωB97M-V/\ndef2-TZVPD'], fontsize=8.5)
        ax.set_title(LBL[m], loc='left', pad=8)
        ax.legend(loc='lower left', fontsize=7.5, frameon=True, framealpha=0.95,
                  edgecolor='#ddd')
    axs[0].set_ylabel(r'$\max_i |F_i|$  evaluated at the transition state'
                      '\n'
                      r'the MLIP produced   [eV Å$^{-1}$]')
    axs[0].set_yscale('log')
    axs[0].set_ylim(min(fm.min(), fd.min()) * 0.6, max(fm.max(), fd.max()) * 2.5)

    fig.suptitle('The same structures, measured twice — only the DFT side '
                 'separates the two classes',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=1.02)
    fig.text(0.012, -0.34,
             r'Left: $\max_i |F_i|$ as returned by the MLIP at the transition '
             'state it produced. Right: the same quantity from a DFT\n'
             'gradient at that identical, unrelaxed geometry. Only the provider '
             'of the force changes; atoms and coordinates do not.\n\n'
             r'$F_i$ is a Cartesian force component and $\max_i$ runs over all '
             r'$3N$ of them. DFT is ωB97M-V/def2-TZVPD, DEFGRID3,'
             '\nThresh 1e-12, TCut 1e-13, RIJCOSX, ORCA 5.0.4; the gradient '
             'reads the orbitals of a preceding single point with\n'
             'STABPerform and therefore lies on the electronic ground state '
             'there.\n\n'
             r'Classes are $\langle S^2\rangle$ of that same single point, which '
             'is exactly 0 or at least ' + ('%.4f' % S2MIN)
             + ' — no cut-off is chosen.\n'
             'Faint dots are the individual structures, one per reaction, '
             'jittered horizontally only — their vertical\nposition is the '
             'measured value. Shaded bars are 95 % percentile bootstrap '
             'intervals of the median,\n10 000 resamples. The axis is '
             'logarithmic because the individual values span four decades.',
             fontsize=7.5, color=GREY, ha='left')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_2a_force_medians_omol25.png' if loud
                     else 'fig9_2_force_medians_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


def fig_energy():
    """Energies at exactly the same geometries as fig 9.2.

    The force figure asks what the surface does at the structure the MLIP
    stopped at. This one asks what the surface is worth there. Same rows, same
    single points, same class labels -- only the quantity changes, from the
    residual force to the two energy differences built from the three
    structures the MLIP produced:

        barrier          E(TS) - E(R)
        reaction energy  E(P)  - E(R)

    Both sides of every pair are read at identical, unrelaxed geometries, so
    the geometry cancels and what is left is the energy alone.
    """
    er = [r for r in csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv')))
          if all(r[k] != '' for k in ('barr_model', 'barr_dft', 'rxne_model',
                                      'rxne_dft', 's2_ts'))]

    def col(k):
        return np.array([float(r[k]) for r in er])

    bm, bd = col('barr_model'), col('barr_dft')
    rm, rd = col('rxne_model'), col('rxne_dft')
    em = np.array([r['model'] for r in er])
    ub = np.abs(col('s2_ts')) > S2_BREAK

    rng = np.random.default_rng(20260823)

    def ci(v, n=10000):
        v = np.asarray(v, float)
        b = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    fig, axs = plt.subplots(2, 3, figsize=(11.4, 9.6),
                            gridspec_kw=dict(height_ratios=[1.0, 1.1],
                                             hspace=0.30, wspace=0.14))
    for row in axs:
        for a in row[1:]:
            a.sharey(row[0])
            a.tick_params(labelleft=False)
    jit = np.random.default_rng(5)

    # -- top row: the barrier itself, MLIP against DFT at the same geometry --
    for ax, m in zip(axs[0], ('uma-s', 'uma-m', 'esen')):
        sel = em == m
        for grp, c, nm in ((sel & ~ub, C_ST, 'RKS stable'),
                           (sel & ub, C_UN, 'RKS unstable')):
            for va, vb in zip(bm[grp], bd[grp]):
                ax.plot([0, 1], [va, vb], color=c, lw=0.5, alpha=0.32, zorder=0)
            for x0, v in ((0, bm[grp]), (1, bd[grp])):
                ax.scatter(x0 + jit.uniform(-0.13, 0.13, len(v)), v, s=30, c=c,
                           alpha=0.58, lw=0.5, edgecolor='white', zorder=1)
            a, b = float(np.median(bm[grp])), float(np.median(bd[grp]))
            ca, cb = ci(bm[grp]), ci(bd[grp])
            ax.vlines(0, *ca, color=c, lw=4, alpha=0.22, zorder=2)
            ax.vlines(1, *cb, color=c, lw=4, alpha=0.22, zorder=2)
            ax.plot([0, 1], [a, b], color=c, lw=2.2, marker='o', ms=8, zorder=3,
                    markeredgecolor='white', markeredgewidth=1.1,
                    label='%s   %.3f → %.3f eV   (n=%d)'
                          % (nm, a, b, int(grp.sum())))
        ax.set_yscale('log')
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['predicted by\nthe MLIP',
                            'ωB97M-V/\ndef2-TZVPD'])
        ax.set_title(LBL[m], loc='left', pad=8)
        ax.legend(loc='upper left', fontsize=7.2, frameon=True,
                  framealpha=0.95, edgecolor='#ddd')
    axs[0][0].set_ylim(0.95, 30)
    axs[0][0].set_yticks([1, 2, 3, 5, 8])
    axs[0][0].set_yticklabels(['1', '2', '3', '5', '8'])
    axs[0][0].set_ylabel('reaction barrier  '
                         r'$\Delta E^{\ddag} = E(\mathrm{TS}) - E(\mathrm{R})$'
                         '\nbuilt from the geometries the MLIP produced   [eV]')

    # -- bottom row: the residual, the only thing there is to see ------------
    FLOOR = 1e-5
    for ax, m in zip(axs[1], ('uma-s', 'uma-m', 'esen')):
        sel = em == m
        ax.axhspan(FLOOR * 0.5, 0.0434, color=C_ST, alpha=0.05, lw=0, zorder=0)
        ax.axhline(0.0434, color='#3b7d3b', lw=1.0, ls='--', zorder=2)
        ax.axhline(0.0257, color=GREY, lw=1.0, ls=':', zorder=2)
        for x0, a, b in ((0, bm, bd), (2, rm, rd)):
            for dx, grp, c in ((0.0, sel & ~ub, C_ST), (0.8, sel & ub, C_UN)):
                v = np.clip(np.abs(a[grp] - b[grp]), FLOOR, None)
                ax.scatter(x0 + dx + jit.uniform(-0.19, 0.19, len(v)), v, s=28,
                           c=c, alpha=0.55, lw=0.5, edgecolor='white', zorder=1)
                md = float(np.median(v))
                lo, hi = ci(v)
                ax.vlines(x0 + dx, lo, hi, color=c, lw=5, alpha=0.22, zorder=2)
                ax.plot([x0 + dx - 0.30, x0 + dx + 0.30], [md, md], color=c,
                        lw=2.4, zorder=3, solid_capstyle='butt')
                ax.text(x0 + dx, 1.4e-5, '%.1f' % (md * 1000), ha='center',
                        va='bottom', fontsize=8, color=c, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(4.2e-6, 6.0)
        ax.set_xlim(-0.60, 3.40)
        ax.set_xticks([0, 0.8, 2, 2.8])
        ax.set_xticklabels(['stable', 'unstable', 'stable', 'unstable'],
                           fontsize=7.5)
        ax.axvline(1.4, color='#ccc', lw=0.8, zorder=0)
        for xc, t in ((0.4, r'barrier  $\Delta E^{\ddag}$'),
                      (2.4, r'reaction energy  $\Delta E$')):
            ax.text(xc, 3.3, t, ha='center', va='center', fontsize=8.5,
                    color='#333')
        ax.set_title(LBL[m], loc='left', pad=8)
    axs[1][0].set_ylabel('|MLIP − DFT| for that energy difference,\n'
                         'both read at the identical geometry   [eV]')
    axs[1][0].text(-0.52, 0.0434 * 1.3, 'chemical accuracy, 43 meV',
                   fontsize=7.2, color='#3b7d3b',
                   bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))
    axs[1][0].text(-0.52, 0.0257 / 3.1, r'$k_{\mathrm{B}}T$ at 298 K, 26 meV',
                   fontsize=7.2, color=GREY,
                   bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))

    fig.suptitle('At those same geometries the energy is right — '
                 'what the MLIP gets wrong is where it put the atoms',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=0.965)
    fig.text(0.012, 0.055,
             'The same ' + str(len(er)) + ' rows, the same single points and the same class '
             'labels as the force figure. One ωB97M-V/def2-TZVPD, '
             'DEFGRID3, Thresh 1e-12 calculation per\nstructure, ORCA 5.0.4, '
             'STABPerform selecting the surface. MLIP energy and DFT energy '
             'are taken at exactly the same three unrelaxed geometries — '
             'the ones the\nMLIP itself produced — so the geometry '
             'cancels out of every difference and the residual is energy '
             'alone.\n'
             'Top: the barrier the MLIP reports, and the barrier DFT gives at '
             'those geometries. Every thin line is one reaction; they are '
             'flat, and the two medians coincide to\nwithin the marker in all '
             'six groups. Bottom: the same comparison row by row as an '
             'absolute difference on a logarithmic axis, which is the only '
             'way the residual\nbecomes visible at all. Values are clipped at '
             '0.01 meV from below. Bold numbers along the bottom edge are '
             'the group medians in meV. Shaded bars are 95 % percentile\n'
             'bootstrap intervals of the median, 10 000 resamples.\n'
             'Read against the force figure: the residual force at these '
             'geometries rises by a factor of two to four when DFT is asked, '
             'while the energy moves by a few meV.',
             fontsize=7.5, color=GREY, ha='left', va='top')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_3_energies_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


FERR = {
    'mae': dict(
        col='f_err_mae', tag='b', slug='mae',
        ylab='mean absolute difference between the MLIP force field and DFT,\n'
             'over all 3N Cartesian components at the transition state\n'
             'the MLIP produced   [eV Å$^{-1}$]',
        title='The force error at the transition state, component by '
              'component — and what the RKS instability does to it',
        defn=r'the absolute differences averaged over all 3N components:  '
             r'MAE $= \langle\, |F_i^{\rm MLIP} - F_i^{\rm DFT}| \,\rangle_i$',
        ylim=(1.1e-3, 2.2)),
    'maxcomp': dict(
        col='f_err_max', tag='c', slug='maxcomp',
        ylab='largest single-component disagreement between the MLIP force '
             'field\nand DFT at the transition state the MLIP produced'
             '\n[eV Å$^{-1}$]',
        title='The worst single force component at the transition state — '
              'and what the RKS instability does to it',
        defn=r'the largest of the absolute differences over all 3N '
             r'components:  $\max_i\, |F_i^{\rm MLIP} - F_i^{\rm DFT}|$',
        ylim=(4e-3, 9.0)),
}


def fig_ferr(kind='mae'):
    """Figures 9.2b and 9.2c — the force error itself, not the residual force.

    9.2 and 9.2a compare two maxima: what the MLIP reports at its own
    transition state, and what DFT reports there. Neither is an error; the
    difference of two maxima is not the distance between the two force fields.
    These two figures take that distance directly. For every structure the
    model force vector and the DFT force vector are subtracted component by
    component, and the 3N absolute differences are reduced either by their
    mean (9.2b) or by their maximum (9.2c):

        9.2b   MAE       = mean_i | F_i(model) - F_i(DFT) |
        9.2c   max comp  = max_i  | F_i(model) - F_i(DFT) |

    Both fields are evaluated at the identical, unrelaxed geometry the MLIP
    produced, so this is a pure force-field discrepancy with no geometry in it.
    """
    cf = FERR[kind]
    rr = [r for r in csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv')))
          if r[cf['col']] != '' and r['s2_ts'] != '']
    val = np.array([float(r[cf['col']]) for r in rr])
    ub = np.abs(np.array([float(r['s2_ts']) for r in rr])) > S2_BREAK
    mm = np.array([r['model'] for r in rr])

    rng = np.random.default_rng(20260823)

    def ci(v, n=10000):
        v = np.asarray(v, float)
        b = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    fig, axs = plt.subplots(1, 3, figsize=(11.4, 5.6), sharey=True)
    jit = np.random.default_rng(5)
    for ax, m in zip(axs, ('uma-s', 'uma-m', 'esen')):
        sel = mm == m
        med = {}
        for x0, grp, c, nm in ((0, sel & ~ub, C_ST, 'RKS stable'),
                               (1, sel & ub, C_UN, 'RKS unstable')):
            v = val[grp]
            ax.scatter(x0 + jit.uniform(-0.17, 0.17, len(v)), v, s=34, c=c,
                       alpha=0.62, lw=0.5, edgecolor='white', zorder=1)
            lo, hi = ci(v)
            ax.vlines(x0, lo, hi, color=c, lw=6, alpha=0.22, zorder=2)
            md = float(np.median(v))
            med[x0] = md
            ax.plot([x0 - 0.30, x0 + 0.30], [md, md], color=c, lw=2.6, zorder=3,
                    solid_capstyle='butt',
                    label='%s   median %.4f eV Å$^{-1}$   (n=%d)'
                          % (nm, md, int(grp.sum())))
        ax.annotate('', xy=(1.42, med[1]), xytext=(1.42, med[0]),
                    arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.0,
                                    shrinkA=0, shrinkB=0))
        ax.text(1.47, np.sqrt(med[0] * med[1]), '×%.1f' % (med[1] / med[0]),
                fontsize=9, color=GREY, va='center', fontweight='bold')
        ax.set_xlim(-0.55, 1.85)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['RKS stable\n' r'$\langle S^2\rangle = 0$',
                            'RKS unstable\n' r'$\langle S^2\rangle > 0$'])
        ax.set_title(LBL[m], loc='left', pad=8)
        ax.legend(loc='upper left', fontsize=7.4, frameon=True,
                  framealpha=0.95, edgecolor='#ddd')
    axs[0].set_yscale('log')
    axs[0].set_ylim(*cf['ylim'])
    axs[0].set_ylabel(cf['ylab'])

    fig.suptitle(cf['title'], fontsize=13, fontweight='bold', x=0.012,
                 ha='left', y=1.015)
    fig.text(0.012, -0.20,
             'For each of the %d structures the model force vector and the '
             'DFT force vector are subtracted component by component, and %s.\n'
             'Both fields are read at the identical, unrelaxed geometry the '
             'MLIP produced, so no geometry difference enters — this is the '
             'distance between the two force\nfields alone. Unlike the '
             'quantity in figures 9.1 and 9.2, which is a difference of two '
             'maxima and can be negative, this is a genuine error and is '
             'positive by construction.\n'
             'DFT is one ωB97M-V/def2-TZVPD, DEFGRID3, Thresh 1e-12 EnGrad '
             'per structure, ORCA 5.0.4, reading the orbitals of the single '
             'point in which STABPerform selected\nthe surface. Classes are '
             r'$\langle S^2\rangle$ of that same single point, exactly 0 or '
             'at least %.4f, so no cut-off is chosen. Faint dots are individual '
             'structures, jittered horizontally\nonly. Shaded bars are 95 %% '
             'percentile bootstrap intervals of the median, 10 000 resamples. '
             'The grey arrow gives the ratio of the two medians.'
             % (len(rr), cf['defn'], S2MIN),
             fontsize=7.5, color=GREY, ha='left')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_2%s_force_%s_omol25.png' % (cf['tag'], cf['slug']))
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


def fig_energy_err():
    """Figure 9.3a — the energy error, forward and reverse barrier.

    9.3 showed that the barrier the MLIP reports and the barrier DFT gives at
    the same geometry sit on top of each other. This figure is the error
    itself, per row, in the same layout as 9.2b/9.2c so the two can be read
    against one another. Both barriers are built from the same three
    structures the MLIP produced, and both sides of each difference are
    evaluated at those unrelaxed geometries:

        forward   E(TS) - E(R)
        reverse   E(TS) - E(P)   = (E(TS) - E(R)) - (E(P) - E(R))

    The group summary is the mean absolute error, which is what MAE means; the
    median is printed next to it because a handful of structures with a broken
    reference determinant move the mean by more than an order of magnitude.
    """
    rr = [r for r in csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv')))
          if all(r[k] != '' for k in ('barr_model', 'barr_dft', 'rxne_model',
                                      'rxne_dft', 's2_ts'))]

    def col(k):
        return np.array([float(r[k]) for r in rr])

    bm, bd = col('barr_model'), col('barr_dft')
    vm, vd = bm - col('rxne_model'), bd - col('rxne_dft')
    ub = np.abs(col('s2_ts')) > S2_BREAK
    mm = np.array([r['model'] for r in rr])

    rng = np.random.default_rng(20260823)

    def ci(v, n=10000):
        v = np.asarray(v, float)
        b = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    fig, axs = plt.subplots(2, 3, figsize=(11.4, 9.6), sharey=True,
                            gridspec_kw=dict(hspace=0.34, wspace=0.10))
    jit = np.random.default_rng(5)
    FLOOR = 3e-5
    ROWS = ((0, bm, bd, r'forward barrier   $E(\mathrm{TS}) - E(\mathrm{R})$'),
            (1, vm, vd, r'reverse barrier   $E(\mathrm{TS}) - E(\mathrm{P})$'))
    for ri, a, b, rlab in ROWS:
        for ax, m in zip(axs[ri], ('uma-s', 'uma-m', 'esen')):
            sel = mm == m
            ax.axhline(0.0434, color='#3b7d3b', lw=1.0, ls='--', zorder=2)
            ax.axhline(0.0257, color=GREY, lw=1.0, ls=':', zorder=2)
            bar = {}
            for x0, grp, c, nm in ((0, sel & ~ub, C_ST, 'RKS stable'),
                                   (1, sel & ub, C_UN, 'RKS unstable')):
                v = np.clip(np.abs(a[grp] - b[grp]), FLOOR, None)
                ax.scatter(x0 + jit.uniform(-0.17, 0.17, len(v)), v, s=32, c=c,
                           alpha=0.60, lw=0.5, edgecolor='white', zorder=1)
                mn, md = float(v.mean()), float(np.median(v))
                lo, hi = ci(v)
                bar[x0] = mn
                ax.vlines(x0, lo, hi, color=c, lw=6, alpha=0.22, zorder=2)
                ax.plot([x0 - 0.30, x0 + 0.30], [mn, mn], color=c, lw=2.6,
                        zorder=3, solid_capstyle='butt',
                        label='%s   MAE %.0f   median %.1f meV   (n=%d)'
                              % (nm, mn * 1000, md * 1000, int(grp.sum())))
                ax.plot([x0 - 0.20, x0 + 0.20], [md, md], color=c, lw=1.4,
                        ls=(0, (2, 1.6)), zorder=3)
            ax.annotate('', xy=(1.42, bar[1]), xytext=(1.42, bar[0]),
                        arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.0,
                                        shrinkA=0, shrinkB=0))
            ax.text(1.47, np.sqrt(bar[0] * bar[1]), '×%.0f' % (bar[1] / bar[0]),
                    fontsize=9, color=GREY, va='center', fontweight='bold')
            ax.set_xlim(-0.55, 1.90)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['RKS stable\n' r'$\langle S^2\rangle = 0$',
                                'RKS unstable\n' r'$\langle S^2\rangle > 0$'])
            ax.set_title('%s  ·  %s' % (LBL[m], 'forward' if ri == 0
                                        else 'reverse'), loc='left', pad=8)
            ax.legend(loc='upper left', fontsize=6.9, frameon=True,
                      framealpha=0.95, edgecolor='#ddd',
                      handlelength=1.4, borderpad=0.4, labelspacing=0.35)
        axs[ri][0].set_ylabel(rlab + '\n'
                              '|MLIP − DFT| at the identical geometry   [eV]')
    axs[0][0].set_yscale('log')
    axs[0][0].set_ylim(2.2e-5, 9.0)
    axs[0][0].text(-0.52, 0.0434 * 1.3, 'chemical accuracy, 43 meV',
                   fontsize=7.2, color='#3b7d3b',
                   bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))
    axs[0][0].text(-0.52, 0.0257 / 3.1, r'$k_{\mathrm{B}}T$ at 298 K, 26 meV',
                   fontsize=7.2, color=GREY,
                   bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))
    axs[1][0].text(-0.52, 0.0434 * 1.3, 'chemical accuracy, 43 meV',
                   fontsize=7.2, color='#3b7d3b',
                   bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))
    axs[1][0].text(-0.52, 0.0257 / 3.1, r'$k_{\mathrm{B}}T$ at 298 K, 26 meV',
                   fontsize=7.2, color=GREY,
                   bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))

    fig.suptitle('The energy error at those same geometries, forward and '
                 'reverse — small, and concentrated in a few broken rows',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=0.955)
    fig.text(0.012, 0.048,
             'The same %d rows and the same class labels as figures 9.2b and '
             '9.3. Both barriers are built from the three structures the MLIP '
             'produced, and both the MLIP\nenergy and the DFT energy are read '
             'at those identical, unrelaxed geometries, so the geometry '
             'cancels and the residual is energy alone. The reverse barrier '
             'is formed\nas E(TS) − E(P) = ΔE‡ − ΔE from the same single '
             'points, not from a separate calculation.\n'
             'Solid bar: the mean absolute error, which is what MAE means. '
             'Dashed bar: the median of the same values. They differ by more '
             'than an order of magnitude in the\nunstable groups because '
             'three structures — one per model — miss by 0.6 to 2.7 eV and '
             'carry the mean on their own; the typical row is a few meV out. '
             'Reporting only the\nMAE would describe none of the rows. Shaded '
             'bars are 95 %% percentile bootstrap intervals of the mean, '
             '10 000 resamples. Values are clipped at 0.03 meV from below,\n'
             'and the grey arrow gives the ratio of the two means. In the '
             'legend the MAE is in meV, as is the median.\n'
             'DFT is one ωB97M-V/def2-TZVPD, DEFGRID3, Thresh 1e-12 single '
             'point per structure, ORCA 5.0.4, with STABPerform selecting the '
             'surface.' % len(rr),
             fontsize=7.5, color=GREY, ha='left', va='top')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_3a_energy_error_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


def fig_barrier_err():
    """Figure 9.3b — the forward barrier error alone.

    Same quantity as the top row of 9.3a, with nothing else on the page.
    One dot is one (reaction, model) pair:

        | [E_MLIP(TS) - E_MLIP(R)] - [E_DFT(TS) - E_DFT(R)] |

    with both sides read at the two unrelaxed structures the MLIP produced.

    The group is summarised twice on purpose. The mean absolute error is what
    MAE means and is what a table would report; the median is what a typical
    row actually looks like. In the unstable groups they differ by a factor of
    three to twenty-four, because one structure per model misses by 0.6 to
    2.7 eV. That structure is named in the panel so the gap is accounted for.
    """
    from matplotlib.lines import Line2D

    rr = [r for r in csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv')))
          if all(r[k] != '' for k in ('barr_model', 'barr_dft', 's2_ts'))]

    def col(k):
        return np.array([float(r[k]) for r in rr])

    err = np.abs(col('barr_model') - col('barr_dft'))
    ub = np.abs(col('s2_ts')) > S2_BREAK
    mm = np.array([r['model'] for r in rr])
    rxn = np.array([r['rxn'] for r in rr])

    rng = np.random.default_rng(20260823)

    def ci(v, n=10000):
        v = np.asarray(v, float)
        b = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    fig, axs = plt.subplots(1, 3, figsize=(12.6, 6.4), sharey=True)
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
            lo, hi = ci(v)
            ax.vlines(x0, lo, hi, color=c, lw=7, alpha=0.20, zorder=2)
            ax.plot([x0 - 0.28, x0 + 0.28], [mn, mn], color=c, lw=2.8, zorder=3,
                    solid_capstyle='butt')
            ax.plot([x0 - 0.22, x0 + 0.22], [md, md], color=c, lw=1.6, zorder=3,
                    ls=(0, (2.2, 1.6)))
            if x0 == 1:                       # name the row that sets the mean
                k = np.flatnonzero(grp)[int(np.argmax(v))]
                if err[k] > 0.1:
                    ax.annotate('%s   %+.2f eV' % (rxn[k], err[k]),
                                xy=(1.0, err[k]), xytext=(-0.05, err[k] * 2.6),
                                fontsize=7.8, color=c, ha='left',
                                arrowprops=dict(arrowstyle='->', color=c,
                                                lw=0.9, shrinkB=6))
        (a0, d0), (a1, d1) = summ[0], summ[1]
        ax.annotate('', xy=(1.38, a1), xytext=(1.38, a0),
                    arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.1,
                                    shrinkA=0, shrinkB=0))
        ax.text(1.45, np.sqrt(a0 * a1), '×%.0f' % (a1 / a0), fontsize=9.5,
                color=GREY, va='center', fontweight='bold')
        key = [
            Line2D([], [], marker='o', ls='', ms=6.5, mfc=C_ST, mec='white',
                   alpha=0.85,
                   label='RKS stable    ' r'$\langle S^2\rangle = 0$'
                         '    (n=%d)' % n[0]),
            Line2D([], [], marker='o', ls='', ms=6.5, mfc=C_UN, mec='white',
                   alpha=0.85,
                   label='RKS unstable  ' r'$\langle S^2\rangle > 0$'
                         '    (n=%d)' % n[1]),
            Line2D([], [], color=GREY, lw=2.8,
                   label='MAE       %5.1f   →  %5.1f meV' % (a0 * 1000, a1 * 1000)),
            Line2D([], [], color=GREY, lw=1.6, ls=(0, (2.2, 1.6)),
                   label='median    %5.1f   →  %5.1f meV' % (d0 * 1000, d1 * 1000)),
            Line2D([], [], color=GREY, lw=7, alpha=0.20,
                   label='95 % bootstrap CI of the MAE'),
        ]
        ax.legend(handles=key, loc='lower left', fontsize=7.0, frameon=True,
                  framealpha=0.96, edgecolor='#ccc', handlelength=1.7,
                  labelspacing=0.5, borderpad=0.7,
                  bbox_to_anchor=(0.515, 0.015))
        ax.set_title(LBL[m], loc='left', pad=8)
        ax.axhline(0.0434, color='#3b7d3b', lw=1.0, ls='--', zorder=0)
        ax.set_xlim(-0.62, 3.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['RKS stable', 'RKS unstable'])
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

    fig.suptitle('How far off is the barrier, once the geometry is taken out '
                 'of the question?',
                 fontsize=13.5, fontweight='bold', x=0.012, ha='left', y=1.005)
    fig.text(0.012, -0.155,
             'One dot is one reaction evaluated with one MLIP: the barrier '
             'that MLIP reports from its own reactant and transition state, '
             'minus the barrier ωB97M-V/def2-TZVPD gives at those\nsame two '
             'structures, unrelaxed. Because both sides are read at identical '
             'geometries, no geometry error can enter — this is the energy '
             'error alone. Horizontal position carries no meaning; the\ndots '
             'are jittered so they do not overlap. %d reactions × 3 MLIPs, '
             'minus %d rows whose set of single points is incomplete, gives '
             '%d rows, and every one of them is drawn.\n'
             'Classes come from the same DFT single point: '
             r'$\langle S^2\rangle$ of the transition state, which is exactly '
             '0 in %d rows and at least %.4f in %d, with nothing in between, '
             'so no threshold is chosen. The green line\nis the conventional '
             'chemical-accuracy target of 1 kcal/mol; it is not a criterion '
             'used anywhere in this work. The grey arrow gives the ratio of '
             'the two MAEs.\n'
             'Mean and median are both drawn because they disagree. In every '
             'unstable group one structure — the one named in the panel — '
             'misses by more than half an electronvolt and sets the\nMAE '
             'almost by itself, which is why the MAE sits above the whole '
             'visible cloud while the median stays inside it. DFT: ORCA '
             '5.0.4, DEFGRID3, Thresh 1e-12, STABPerform selecting the '
             'surface.'
             % (len(set(rxn)), 3 * len(set(rxn)) - len(rr), len(rr),
                int((~ub).sum()), S2MIN, int(ub.sum())),
             fontsize=7.6, color=GREY, ha='left')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_3b_barrier_error_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


def fig_barrier_err_c():
    """Figure 9.3c — 9.3b stripped down: every number sits next to its mark.

    Same quantity, same rows, same classes. What changed: the mean and the
    median are labelled where they are drawn instead of in a legend block, the
    bootstrap interval is gone, and the class definition moved onto the axis
    where the classes are named.
    """
    from matplotlib.lines import Line2D

    rr = [r for r in csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv')))
          if all(r[k] != '' for k in ('barr_model', 'barr_dft', 's2_ts'))]

    def col(k):
        return np.array([float(r[k]) for r in rr])

    err = np.abs(col('barr_model') - col('barr_dft'))
    ub = np.abs(col('s2_ts')) > S2_BREAK
    mm = np.array([r['model'] for r in rr])
    rxn = np.array([r['rxn'] for r in rr])

    fig, axs = plt.subplots(1, 3, figsize=(12.6, 6.4), sharey=True)
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
            # the two labels are pushed apart only when they would collide
            ya, yb = mn, md
            if abs(np.log10(mn / md)) < 0.17:
                g = np.sqrt(mn * md)
                ya, yb = g * 10 ** 0.085, g * 10 ** -0.085
            ax.text(x0 + 0.32, ya, 'MAE %.1f meV' % (mn * 1000), fontsize=8,
                    color=c, va='center', ha='left', fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.18', fc='white', ec='none', alpha=0.85))
            ax.text(x0 + 0.32, yb, 'median %.1f meV' % (md * 1000), fontsize=8,
                    color=c, va='center', ha='left', bbox=dict(boxstyle='square,pad=0.18', fc='white', ec='none', alpha=0.85))
            if x0 == 1:                       # name the row that sets the mean
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

    fig.suptitle('How far off is the barrier, once the geometry is taken out '
                 'of the question?',
                 fontsize=13.5, fontweight='bold', x=0.012, ha='left', y=1.005)
    fig.text(0.012, -0.155,
             'One dot is one reaction evaluated with one MLIP: the barrier '
             'that MLIP reports from its own reactant and transition state, '
             'minus the barrier ωB97M-V/def2-TZVPD gives at those\nsame two '
             'structures, unrelaxed. Because both sides are read at identical '
             'geometries, no geometry error can enter — this is the energy '
             'error alone. Horizontal position carries no meaning; the\ndots '
             'are jittered so they do not overlap. %d reactions × 3 MLIPs, '
             'minus %d rows whose set of single points is incomplete, gives '
             '%d rows, and every one of them is drawn.\n'
             r'A structure counts as RKS unstable when $\langle S^2\rangle$ '
             'of that same DFT single point is above zero, which means the '
             'restricted closed-shell determinant is not the ground state '
             'there.\nThe value is exactly 0 in %d rows and at least %.4f '
             'in %d, with nothing in between, so no threshold is chosen. The '
             'green line is the conventional chemical-accuracy target of\n'
             '1 kcal/mol; it is not a criterion used anywhere in this work. '
             'The grey arrow gives the ratio of the two MAEs.\n'
             'Mean and median are both drawn because they disagree. In every '
             'unstable group one structure — the one named in the panel — '
             'misses by more than half an electronvolt and sets the\nMAE '
             'almost by itself, which is why the MAE sits above the whole '
             'visible cloud while the median stays inside it. DFT: ORCA '
             '5.0.4, DEFGRID3, Thresh 1e-12, STABPerform selecting the '
             'surface.'
             % (len(set(rxn)), 3 * len(set(rxn)) - len(rr), len(rr),
                int((~ub).sum()), S2MIN, int(ub.sum())),
             fontsize=7.6, color=GREY, ha='left')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_3c_barrier_error_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


def fig_geometry():
    """Figure 9.4 — was die Geometrieverschiebung allein anrichtet.

    Jede Reaktion hat drei vorhergesagte Uebergangszustaende, einen je MLIP.
    An allen dreien wird dieselbe DFT-Rechnung ausgefuehrt: dasselbe
    Funktional, derselbe Basissatz, dasselbe Gitter, derselbe Code, derselbe
    Nullpunkt bis auf das jeweils eigene Modell-Edukt. Der einzige
    Unterschied zwischen den drei Zahlen ist, wo die Atome stehen.

    Was die drei Barrieren innerhalb einer Reaktion auseinandertreibt, ist
    damit vollstaendig Geometrie -- kein Modellfehler in der Energie, kein
    Niveauwechsel, kein Basissatzeffekt.

        links   die drei Werte je Reaktion, als Abstand zum Mittel der drei
        rechts  dieselbe Spannweite gegen den geometrischen Abstand der drei
                Uebergangszustaende, gemessen als groesste paarweise
                Kabsch-RMSD
    """
    import collections
    from matplotlib.lines import Line2D

    rr = list(csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv'))))
    rmsd = {r['rxn']: float(r['rmsd_max']) for r in
            csv.DictReader(open(os.path.join(RES, 'model_ts_rmsd.csv')))}
    by = collections.defaultdict(dict)
    for r in rr:
        by[r['rxn']][r['model']] = r

    MODELS = ('uma-s', 'uma-m', 'esen')
    MARK = {'uma-s': 'o', 'uma-m': 's', 'esen': '^'}
    rx, bar, spread, unst, geo = [], [], [], [], []
    for k, v in by.items():
        if len(v) < 3 or k not in rmsd:
            continue
        b = np.array([float(v[m]['barr_dft']) for m in MODELS])
        rx.append(k)
        bar.append(b)
        spread.append((b.max() - b.min()) * 1000.0)
        unst.append(any(v[m]['unstable_ts'] == '1' for m in MODELS))
        geo.append(rmsd[k])
    bar = np.array(bar)
    spread, unst, geo = np.array(spread), np.array(unst), np.array(geo)
    rx = np.array(rx)
    dev = (bar - bar.mean(axis=1, keepdims=True)) * 1000.0     # meV

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.2, 6.2),
        gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.26))

    # ---- A  die drei Werte je Reaktion -------------------------------------
    o = np.argsort(spread)
    x = np.arange(len(o))
    for xi, i in zip(x, o):
        c = C_UN if unst[i] else C_ST
        axA.vlines(xi, dev[i].min(), dev[i].max(), color=c, lw=1.0, alpha=0.40,
                   zorder=1)
        for j, m in enumerate(MODELS):
            axA.scatter(xi, dev[i][j], s=30, marker=MARK[m], c=c, alpha=0.80,
                        lw=0.4, edgecolor='white', zorder=2)
    axA.axhline(0, color='#bbb', lw=0.9, zorder=0)
    for lv, ls, col, lab in ((43.4, '--', '#3b7d3b', 'chemische Genauigkeit, 43 meV'),
                             (-43.4, '--', '#3b7d3b', None)):
        axA.axhline(lv, color=col, lw=1.0, ls=ls, zorder=0)
    axA.set_yscale('symlog', linthresh=1.0, linscale=0.7)
    axA.set_ylim(-4000, 4000)
    axA.set_xlim(-1, len(o))
    axA.set_xticks([])
    axA.set_xlabel('45 Reaktionen, nach Spannweite sortiert')
    axA.set_ylabel('DFT-Barriere an der jeweiligen Modellgeometrie,\n'
                   'als Abstand zum Mittel der drei Werte derselben Reaktion'
                   '   [meV]')
    axA.text(0.012, 0.965, 'linear innerhalb ±1 meV, darüber logarithmisch',
             transform=axA.transAxes, fontsize=7.4, color=GREY, va='top')
    axA.text(-0.5, 60, 'chemische Genauigkeit, 43 meV', fontsize=7.4,
             color='#3b7d3b', va='bottom')
    # feste Leiter, weil rxn4113 und rxn0894 fast auf derselben Hoehe liegen
    LAD = [(-4, 3200.0), (-9, 1100.0), (-15, 420.0), (-21, 150.0)]
    for k, i in enumerate(np.argsort(-spread)[:4]):
        xi = int(np.flatnonzero(o == i)[0])
        dx, yt = LAD[k]
        axA.annotate(rx[i], xy=(xi, dev[i].max()), xytext=(xi + dx, yt),
                     fontsize=7.6, color=C_UN if unst[i] else C_ST,
                     ha='right', va='center',
                     arrowprops=dict(arrowstyle='->', lw=0.8, shrinkB=4,
                                     color=C_UN if unst[i] else C_ST))
    axA.set_title('A   Dieselbe Reaktion, drei Modellgeometrien', loc='left',
                  pad=8)

    # ---- B  Spannweite gegen geometrischen Abstand -------------------------
    axB.axhline(43.4, color='#3b7d3b', lw=1.0, ls='--', zorder=0)
    axB.axhline(25.7, color=GREY, lw=1.0, ls=':', zorder=0)
    for sel, c, nm in ((~unst, C_ST, 'alle drei TS RKS-stabil'),
                       (unst, C_UN, 'mindestens einer RKS-instabil')):
        axB.scatter(geo[sel], spread[sel], s=52, c=c, alpha=0.72, lw=0.5,
                    edgecolor='white', zorder=2,
                    label='%s   (n=%d)' % (nm, int(sel.sum())))
    axB.set_xscale('log')
    axB.set_yscale('log')
    axB.set_xlim(2e-4, 6)
    axB.set_ylim(0.02, 9000)
    axB.set_xlabel('geometrischer Abstand der drei Übergangszustände\n'
                   'größte paarweise Kabsch-RMSD   [Å]')
    axB.set_ylabel('Spannweite der DFT-Barriere über die drei\n'
                   'Modellgeometrien derselben Reaktion   [meV]')
    axB.legend(loc='upper left', fontsize=7.8, frameon=True, framealpha=0.95,
               edgecolor='#ddd')
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    axB.text(0.975, 0.055,
             'Spearman  ρ = %+.2f  (alle 45)\n'
             '          %+.2f  stabil,  %+.2f  instabil'
             % (np.corrcoef(rk(geo), rk(spread))[0, 1],
                np.corrcoef(rk(geo[~unst]), rk(spread[~unst]))[0, 1],
                np.corrcoef(rk(geo[unst]), rk(spread[unst]))[0, 1]),
             transform=axB.transAxes, ha='right', va='bottom', fontsize=8,
             family='DejaVu Sans Mono',
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ddd'))
    axB.text(2.6e-4, 43.4 * 1.25, 'chemische Genauigkeit, 43 meV', fontsize=7.4,
             color='#3b7d3b')
    axB.text(2.6e-4, 25.7 / 2.2, r'$k_{\mathrm{B}}T$ bei 298 K, 26 meV',
             fontsize=7.4, color=GREY)
    for i in np.argsort(-spread)[:4]:
        axB.annotate(rx[i], xy=(geo[i], spread[i]),
                     xytext=(geo[i] * 0.28, spread[i] * 1.9), fontsize=7.6,
                     color=C_UN if unst[i] else C_ST, ha='right',
                     arrowprops=dict(arrowstyle='->', lw=0.8,
                                     color=C_UN if unst[i] else C_ST))
    axB.set_title('B   Wie weit auseinander, wie viel Energie', loc='left',
                  pad=8)

    key = [Line2D([], [], marker=MARK[m], ls='', ms=6.5, mfc=GREY,
                  mec='white', label=LBL[m]) for m in MODELS]
    axA.legend(handles=key, loc='lower right', fontsize=7.8, frameon=True,
               framealpha=0.95, edgecolor='#ddd', ncol=3, handletextpad=0.3,
               columnspacing=1.0)

    fig.suptitle('Was allein die Geometrie anrichtet: dieselbe DFT-Rechnung '
                 'an den drei Übergangszuständen einer Reaktion',
                 fontsize=13, fontweight='bold', x=0.012, ha='left', y=1.01)
    fig.text(0.012, -0.135,
             'Jede Reaktion liefert drei Übergangszustände, einen je MLIP. An '
             'allen dreien läuft dieselbe Rechnung: ωB97M-V/def2-TZVPD, '
             'DEFGRID3, Thresh 1e-12, ORCA 5.0.4,\nSTABPerform wählt die '
             'Fläche. Die Barriere wird jeweils gegen das Edukt desselben '
             'Modells gemessen, das in allen %d Zeilen geschlossenschalig '
             'ist. Was die drei Zahlen einer\nReaktion auseinandertreibt, ist '
             'deshalb vollständig Geometrie — kein Energiefehler des Modells, '
             'kein Niveauwechsel, kein Basissatzeffekt.\n'
             'Links steht jede Reaktion für eine senkrechte Linie mit drei '
             'Punkten. Rechts dieselbe Spannweite gegen den rein '
             'geometrischen Abstand der drei Strukturen, gemessen als größte\n'
             'paarweise Kabsch-RMSD. Eine Reaktion heißt hier instabil, wenn '
             'mindestens einer ihrer drei Übergangszustände ⟨S²⟩ > 0 hat; das '
             'sind 18 von 45.\n'
             'Median der Spannweite: %.1f meV bei den stabilen, %.1f meV bei '
             'den instabilen Reaktionen. %d der 45 bleiben unter 0.1 meV, %d '
             'überschreiten die chemische Genauigkeit.'
             % (3 * len(rx), float(np.median(spread[~unst])),
                float(np.median(spread[unst])),
                int((spread < 0.1).sum()), int((spread > 43.4).sum())),
             fontsize=7.6, color=GREY, ha='left')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_4_geometry_effect_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


def fig_spread():
    """Figure 9.5 — the spread itself, one point per reaction.

    Same quantity as the right panel of 9.4, without the detour: max minus min
    of the three DFT barriers computed at the three model transition states of
    one reaction. 45 points instead of 135; the statement is the axis. Half
    column.

    Mean and median are both drawn because they disagree by more than an order
    of magnitude in both groups: one reaction per group carries the mean
    almost by itself.
    """
    import collections

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

    fig, ax = plt.subplots(figsize=(5.8, 6.8))
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
            k = int(np.flatnonzero(np.flatnonzero(sel) == i)[0])
            ax.annotate(rx[i], xy=(xj[k], spread[i]),
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
    ax.set_title('How far the barrier drifts on geometry alone',
                 loc='left', pad=10, fontsize=12)

    fig.text(0.0, -0.125,
             'One dot is one reaction. Its three transition states come from '
             'UMA-S, UMA-M and eSEN, and the same calculation runs at all\n'
             'three: ωB97M-V/def2-TZVPD, DEFGRID3, Thresh 1e-12, ORCA 5.0.4, '
             'STABPerform selecting the surface. Each barrier is measured\n'
             'against the reactant of the same model, closed-shell in every '
             'case. What drives the three numbers apart is therefore geometry\n'
             'and nothing else. A reaction counts as unstable when at least '
             r'one of its three transition states has $\langle S^2\rangle$ > 0.'
             '\nSolid bar: the mean. Dashed bar: the median. They differ by '
             'more than an order of magnitude in both groups because one\n'
             'reaction per group carries the mean almost alone — %s at %.0f '
             'meV among the stable, %s at %.0f meV among the unstable.\n'
             'Horizontal position is jittered only and carries no information.'
             % (rx[~unst][np.argmax(spread[~unst])], spread[~unst].max(),
                rx[unst][np.argmax(spread[unst])], spread[unst].max()),
             fontsize=7.4, color=GREY, ha='left')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_5_barrier_spread_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))


def fig_depth():
    """Figure 9.6 — breaking depth against residual force, one panel per MLIP.

    The breaking depth is E_RKS(TS) - E_BS(TS) at the geometry the MLIP
    stopped at: how far the broken solution lies below the restricted one.
    It is exactly 0 for the 82 rows where STABPerform found no second
    solution, and between 0.6 and 3986 meV for the other 53.

    The residual force is max_i |F_i| from the DFT gradient at that same
    point, on the surface STABPerform selected -- the quantity the
    stationarity test in this work is built on.

    The zero gets a column of its own, separated by a gap, because it is not a
    small depth: at those points there is no second solution at all. The step
    from that column into the logarithmic region is the whole effect. Inside
    the region the decade medians are flat, which is the second half of the
    statement and the reason they are drawn.
    """
    from matplotlib.lines import Line2D

    rr = list(csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv'))))
    dep = np.array([float(r['depth_ts_mev']) for r in rr])
    frc = np.array([float(r['f_dft_max']) for r in rr])
    mdl = np.array([r['model'] for r in rr])
    ub = dep > 0
    STAGE1 = 0.15
    BINS = ((0.4, 50.0), (50.0, 500.0), (500.0, 6000.0))

    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    jit = np.random.default_rng(7)

    fig = plt.figure(figsize=(13.0, 6.0))
    outer = fig.add_gridspec(1, 3, wspace=0.24)
    axL = []
    for gi, m in enumerate(('uma-s', 'uma-m', 'esen')):
        sub = outer[gi].subgridspec(1, 2, width_ratios=[0.34, 1.0], wspace=0.06)
        a0 = fig.add_subplot(sub[0], sharey=axL[0][0] if axL else None)
        a1 = fig.add_subplot(sub[1], sharey=a0)
        axL.append((a0, a1))
        sel = mdl == m
        z, p = sel & ~ub, sel & ub

        # -- Nullspalte
        a0.scatter(jit.uniform(-0.24, 0.24, int(z.sum())), frc[z], s=32,
                   c=C_ST, alpha=0.62, lw=0.5, edgecolor='white', zorder=2)
        m0 = float(np.median(frc[z]))
        a0.plot([-0.34, 0.34], [m0, m0], color=C_ST, lw=2.6, zorder=3,
                solid_capstyle='butt')
        a0.set_xlim(-0.6, 0.6)
        a0.set_xticks([0])
        a0.set_xticklabels(['0'])
        a0.set_xlabel('keine\nzweite\nLösung', fontsize=7.4, color=GREY,
                      labelpad=2)
        a0.axhline(STAGE1, color='k', lw=1.0, ls='--', zorder=1)
        a0.spines['right'].set_visible(False)

        # -- Logbereich
        a1.axhline(STAGE1, color='k', lw=1.0, ls='--', zorder=1)
        a1.scatter(dep[p], frc[p], s=34, c=C_UN, alpha=0.66, lw=0.5,
                   edgecolor='white', zorder=2)
        m1 = float(np.median(frc[p]))
        for lo, hi in BINS:
            s = p & (dep > lo) & (dep <= hi)
            if not s.sum():
                continue
            a1.plot([lo, hi], [np.median(frc[s])] * 2, color=C_UN, lw=2.6,
                    zorder=3, solid_capstyle='butt')
            a1.text(np.sqrt(lo * hi), np.median(frc[s]) * 1.16,
                    'n=%d' % int(s.sum()), fontsize=7.0, color=C_UN,
                    ha='center', va='bottom')
        a1.set_xscale('log')
        a1.set_xlim(0.4, 6000)
        a1.spines['left'].set_visible(False)
        a1.tick_params(which='both', labelleft=False, left=False)
        a1.set_xlabel('%s\ngebrochen um … meV' % LBL[m], fontsize=8.6,
                      labelpad=2)
        a1.xaxis.get_label().set_fontweight('bold')

        # Sprung und Zusammenhang beschriften
        a1.annotate('', xy=(1.05, m1), xytext=(1.05, m0),
                    arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.0,
                                    shrinkA=0, shrinkB=0))
        a1.text(1.25, np.sqrt(m0 * m1), '×%.1f' % (m1 / m0), fontsize=9,
                color=GREY, va='center', fontweight='bold')
        a1.text(0.97, 0.035,
                'ρ = %+.2f  innerhalb\nder %d gebrochenen Zeilen'
                % (np.corrcoef(rk(dep[p]), rk(frc[p]))[0, 1], int(p.sum())),
                transform=a1.transAxes, ha='right', va='bottom', fontsize=7.6,
                color=GREY,
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#ddd'))
        a0.set_title(LBL[m], loc='left', pad=8)

    axL[0][0].set_yscale('log')
    axL[0][0].set_ylim(0.0045, 2.6)
    axL[0][0].set_ylabel('Restkraft der DFT-Rechnung am Modell-'
                         'Übergangszustand\n' r'$\max_i |F_i|$   [eV Å$^{-1}$]')
    axL[0][0].text(-0.56, STAGE1 * 1.14, 'Stufe 1: 0.15', fontsize=7.4,
                   color='k', clip_on=False,
                   bbox=dict(boxstyle='square,pad=0.14', fc='white',
                             ec='none', alpha=0.9))

    key = [Line2D([], [], marker='o', ls='', ms=6.5, mfc=C_ST, mec='white',
                  alpha=0.85, label='Tiefe = 0: die restringierte Lösung ist '
                                    'der Grundzustand'),
           Line2D([], [], marker='o', ls='', ms=6.5, mfc=C_UN, mec='white',
                  alpha=0.85, label='Tiefe > 0: eine tiefere gebrochene '
                                    'Lösung existiert'),
           Line2D([], [], color=GREY, lw=2.6,
                  label='Median, im Logbereich je Dekade')]
    fig.legend(handles=key, loc='lower center', ncol=3, fontsize=8.2,
               frameon=False, bbox_to_anchor=(0.5, -0.10), handlelength=1.6,
               columnspacing=2.0)

    RHOMAX = float(np.ceil(max(
        abs(np.corrcoef(rk(dep[ub & (mdl == m)]),
                        rk(frc[ub & (mdl == m)]))[0, 1])
        for m in ('uma-s', 'uma-m', 'esen')) * 10) / 10)
    NRXN = len({r['rxn'] for r, k in zip(rr, ub) if k})

    fig.suptitle('Die Brechungstiefe sagt, ob — nicht wieviel',
                 fontsize=13.5, fontweight='bold', x=0.012, ha='left', y=1.02)
    fig.text(0.012, -0.30,
             'Brechungstiefe ist E_RKS(TS) − E_BS(TS) an genau dem Punkt, an '
             'dem das Modell stehengeblieben ist: wie weit die gebrochene '
             'Lösung unter der restringierten liegt.\nSie ist exakt 0 in den '
             + str(int((~ub).sum())) + ' Zeilen, in denen STABPerform keine '
             'zweite Lösung gefunden hat, und liegt zwischen '
             + ('%.1f' % dep[ub].min()) + ' und ' + ('%.0f' % dep[ub].max())
             + ' meV in den übrigen ' + str(int(ub.sum())) + '. '
             'Die Null steht deshalb in\neiner eigenen, abgesetzten Spalte — '
             'dort ist die Tiefe nicht klein, sondern es gibt keine zweite '
             'Fläche. Die Restkraft ist max|F| aus dem DFT-Gradienten am '
             'selben Punkt, auf der\nvon STABPerform gewählten Fläche; alles '
             'auf ωB97M-V/def2-TZVPD, DEFGRID3, Thresh 1e-12, ORCA 5.0.4.\n'
             'Der Sprung von der Nullspalte in den Logbereich ist der ganze '
             'Effekt: Median ' + ('%.3f' % np.median(frc[~ub])) + ' gegen '
             + ('%.3f' % np.median(frc[ub])) + ' eV/Å über alle Zeilen. '
             'Innerhalb des Logbereichs passiert nichts mehr — die\n'
             'Dekadenmediane liegen bei '
             + ', '.join('%.3f' % np.median(frc[ub & (dep > lo) & (dep <= hi)])
                         for lo, hi in BINS)
             + ' eV/Å, über drei Dekaden Tiefe also gleich, und ρ bleibt in '
             'allen drei Panels betragsmäßig unter ' + ('%.1f' % RHOMAX)
             + '.\n'
             'Vorbehalt: die ' + str(int(ub.sum())) + ' gebrochenen Zeilen '
             'sind nur ' + str(NRXN) + ' Reaktionen, '
             'und die Tiefe ist innerhalb einer Reaktion über die drei '
             'Modelle nahezu konstant. Das wirksame n ist eher '
             + str(NRXN) + ' als ' + str(int(ub.sum())) + '.',
             fontsize=7.5, color=GREY, ha='left')
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, 'fig9_6_depth_vs_force_omol25.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ', os.path.relpath(p, HERE))



def selfcheck():
    """Prueft die Aussagen der Bildunterschriften, die keine Einsetzung sind.

    Eingesetzte Zahlen koennen nicht veralten -- sie werden bei jedem Rendern
    aus der Tabelle geholt. Was veralten kann, sind Saetze: "nichts liegt
    dazwischen", "alle Edukte sind geschlossenschalig", "die Flaechen fallen
    dort zusammen". Die stehen hier, jede mit der Rechnung daneben, die sie
    stuetzt. Ein Fehlschlag bricht ab, statt eine falsche Unterschrift zu
    drucken.

    Kein Automatismus: dieser Block faengt nur, wofuer eine Zeile darin steht.
    """
    bad = []

    def need(ok, claim):
        print('  %s  %s' % ('ok  ' if ok else 'FEHL', claim))
        if not ok:
            bad.append(claim)

    allr = list(csv.DictReader(open(os.path.join(RES, 'omol25_model_geoms.csv'))))
    col = lambda k: np.array([float(r[k]) for r in allr])
    s2t, s2r = np.abs(col('s2_ts')), np.abs(col('s2_r'))
    dep, un = col('depth_ts_mev'), col('unstable_ts').astype(bool)
    comp, norm = col('f_model_max'), col('f_model_norm_max')

    print('Aussagen der Bildunterschriften')
    need(s2t[s2t > 0].min() > S2_BREAK,
         'fig9, 9.1, 9.2*, 9.3b, 9.3c: "no value in between" — kleinster '
         '<S^2> ueber null ist %.6f, die Schwelle liegt bei %.2f'
         % (s2t[s2t > 0].min(), S2_BREAK))
    need((s2r == 0).all(),
         'fig9.4: "das Edukt ist in allen Zeilen geschlossenschalig" — '
         '%d von %d mit <S^2>(R) = 0' % (int((s2r == 0).sum()), len(allr)))
    need(np.abs(dep[~un]).max() < 1.0,
         'fig9.6: "exakt 0" fuer die stabilen Zeilen — groesste gemessene '
         'Tiefe dort %.4f meV' % np.abs(dep[~un]).max())
    need((dep[un] > 0).all(),
         'fig9.6: die gebrochenen Zeilen haben durchweg Tiefe > 0')
    rat = comp / norm
    need(round(rat.min(), 2) == 0.61 and round(rat.max(), 2) == 1.00,
         'fig9.1: "the two conventions differ by a measured factor of 0.61 '
         'to 1.00" — gemessen %.4f bis %.4f' % (rat.min(), rat.max()))
    rr = {}
    for r in allr:
        rr.setdefault(r['rxn'], set()).add(r['model'])
    need(all(len(v) == 3 for v in rr.values()),
         'fig9.4, 9.5: "drei Modellgeometrien je Reaktion" — %d von %d '
         'Reaktionen vollstaendig'
         % (sum(1 for v in rr.values() if len(v) == 3), len(rr)))

    if bad:
        raise SystemExit('ABBRUCH: %d Aussage(n) stimmen nicht mehr' % len(bad))
    print()


selfcheck()
fig_energy()
fig_silent()
fig_silent_v2()
fig_slope()
fig_slope(loud=True)
fig_ferr()
fig_ferr("maxcomp")
fig_energy_err()
fig_barrier_err()
fig_barrier_err_c()
fig_geometry()
fig_spread()
fig_depth()
