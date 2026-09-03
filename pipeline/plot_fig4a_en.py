# DEPRECATED -- TZVP-era figures/tables, superseded by
# pipeline/omol25_model_geoms.py, pipeline/hinge_tables.py and
# pipeline/plot_omol25_figs.py. Do not run for the paper; retained as history.
# The 1.697 eV/A median is obsolete; successor numbers live in
# results/hinge_t1x.csv (1.636) and results/hinge_omol25.csv (1.870).

"""Figure 4, panel A, in English, as a standalone figure.

The same point, two surfaces: for each multireference reaction the residual
force at the RKS transition state, measured once on the RKS surface it was
optimised on and once on the broken-symmetry surface that is the ground state
there. Same nuclear coordinates in both cases -- only the electronic solution
differs.

Data: results/hinge_rows.csv (19 rows, F_rks and F_bs, PySCF wB97M-V/def2-TZVP
via stab_pipeline, at ~/orca_neb_results/<rxn>/transition_state.xyz).
The level is def2-TZVP, not the def2-TZVPD audit level -- this panel predates
the OMol25-level recomputation and is the only figure still on the old level.

figures/fig4a_hinge_en.png
"""
import csv
import os

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
STAT = 0.15                      # stationarity threshold used in this work
C_ST = '#2a6f7f'
C_UN = '#c2542a'
GREY = '#6b6b6b'

mpl.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.18, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
})

rows = list(csv.DictReader(open(os.path.join(RES, 'hinge_rows.csv'))))
rows.sort(key=lambda r: -(float(r['F_bs']) / float(r['F_rks'])))
frks = np.array([float(r['F_rks']) for r in rows])
fbs = np.array([float(r['F_bs']) for r in rows])
fac = fbs / frks
med = float(np.median(fbs))

fig, ax = plt.subplots(figsize=(7.6, 6.4))
y = np.arange(len(rows))[::-1]

ax.axvspan(1e-3, STAT, color=C_ST, alpha=0.08, lw=0)
ax.axvline(STAT, color='k', lw=1.1, ls=':', zorder=2)
ax.axvline(med, color=C_UN, lw=1.0, ls='--', alpha=0.7, zorder=2)

ax.hlines(y, frks, fbs, color='#c7c7c7', lw=1.8, zorder=1)
ax.scatter(frks, y, s=46, c=C_ST, zorder=3,
           label='on the RKS surface   (n=%d)' % len(rows))
ax.scatter(fbs, y, s=54, c=C_UN, marker='D', zorder=3,
           label='on the broken-symmetry surface, the ground state')
for yy, f in zip(y, fac):
    ax.text(0.985, yy, '×%.0f' % f, transform=ax.get_yaxis_transform(),
            va='center', ha='right', fontsize=8, color=GREY)

ax.set_yticks(y)
ax.set_yticklabels([r['rxn'] for r in rows], fontsize=8)
ax.set_xscale('log')
ax.set_xlim(0.018, 14)
ax.set_ylim(-0.8, len(rows) - 0.2)
ax.set_xlabel(r'residual force $\max_i |F_i|$ at the RKS transition state'
              '   [eV Å$^{-1}$]')
ax.set_title('The same point, two surfaces', loc='left', pad=10)

ax.text(med * 1.10, len(rows) - 0.55, 'median %.3f' % med,
        fontsize=8, color=C_UN)
ax.text(STAT * 0.90, len(rows) - 0.55, 'stationarity threshold 0.15',
        fontsize=8, rotation=90, ha='right', va='top')
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.005), ncol=1)

n_rks = int((frks < STAT).sum())
n_bs = int((fbs < STAT).sum())
n_cross = int((fbs < frks).sum())
ax.text(0.5, -0.115,
        '%d of %d stationary on RKS      ·      %d of %d stationary on BS'
        '      ·      %d lines cross back'
        % (n_rks, len(rows), n_bs, len(rows), n_cross),
        transform=ax.transAxes, ha='center', va='top', fontsize=8.5,
        fontweight='bold')

fig.text(0.0, -0.20,
         'One row is one multireference reaction. Both markers sit at the '
         'identical nuclear geometry — the transition state of the reference\n'
         'NEB, `~/orca_neb_results/<rxn>/transition_state.xyz` — and differ '
         'only in which electronic solution the force is evaluated on. The\n'
         'geometry was optimised on the RKS surface, so a small force there is '
         'expected; the broken-symmetry solution is the ground state at\n'
         'that point, and the force it carries is what the geometry would '
         'have to relax against.\n'
         'wB97M-V/def2-TZVP, PySCF, grids 3, conv_tol 1e-10 '
         '(pipeline/stability_pipeline.py). This panel is still on the '
         'def2-TZVP level, not on the\ndef2-TZVPD audit level of the other '
         'figures. Ratios on the right are F_BS / F_RKS. Data: '
         'results/hinge_rows.csv.',
         fontsize=7.5, color=GREY, ha='left')

os.makedirs(FIG, exist_ok=True)
p = os.path.join(FIG, 'fig4a_hinge_en.png')
fig.savefig(p, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  ', os.path.relpath(p, HERE))
print('   %d Zeilen, F_RKS %.4f-%.4f, F_BS %.4f-%.4f, Verhaeltnis %.1f-%.1f'
      % (len(rows), frks.min(), frks.max(), fbs.min(), fbs.max(),
         fac.min(), fac.max()))
