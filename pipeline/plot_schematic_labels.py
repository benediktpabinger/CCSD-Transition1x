# -*- coding: utf-8 -*-
"""Schema: ein Transition1x-Pfad, die OMol25-Labels darauf, und die Restkraefte.

Keine Daten -- eine Skizze entlang einer Reaktionskoordinate, fuer eine
Reaktion mit gebrochener Grundzustandsloesung am Sattel. Drei Kurven:

    A  wB97x/6-31G(d), restringiert       -- die Flaeche, auf der Transition1x
                                            den Pfad relaxiert hat
    B  wB97M-V/def2-TZVPD, restringiert   -- dieselbe Flaechenart auf dem
                                            OMol25-Niveau (der Niveauwechsel)
    C  wB97M-V/def2-TZVPD, Grundzustand   -- die gebrochene Loesung, auf der
                                            OMol25 tatsaechlich labelt

Der Transition1x-Sattel liegt auf A. OMol25 liest dort Energie und Kraft von
C ab. Auf B und C ist der Punkt kein Sattel mehr: die Pfeile sind die
Restkraefte, die offenen Marker zeigen, wo ein relaxierter Pfad auf B bzw. C
seinen Sattel haette.

Die Mediane in der Textbox stammen aus results/hinge_t1x.csv (instabile
Zeilen); die Kurvenformen sind frei gewaehlt und nicht massstaeblich.

figures/fig_schematic_labels.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, 'figures')

C_ST = '#2a6f7f'
C_UN = '#c2542a'
GREY = '#6b6b6b'
LIGHT = '#9a9a9a'

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

x = np.linspace(0.0, 1.0, 600)
g = lambda x0, w: np.exp(-((x - x0) / w) ** 2)

E_A = 1.00 * g(0.50, 0.16)                    # Transition1x-Flaeche
E_B = 1.18 * g(0.58, 0.16) + 0.18             # OMol25-Niveau, restringiert
E_C = E_B - 0.38 * g(0.56, 0.13)              # OMol25-Niveau, Grundzustand

x0 = 0.50                                      # der Transition1x-Sattel
i0 = int(np.argmin(np.abs(x - x0)))
xB, xC = x[np.argmax(E_B)], x[np.argmax(E_C)]
slope = lambda E: np.gradient(E, x)[i0]

fig, ax = plt.subplots(figsize=(7.4, 4.3))
fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.14)

ax.plot(x, E_A, color=LIGHT, lw=1.6, ls='--', zorder=2,
        label=r'$\omega$B97x/6-31G(d), restricted  —  where Transition1x relaxed the path')
ax.plot(x, E_B, color=C_ST, lw=2.0, zorder=3,
        label=r'$\omega$B97M-V/def2-TZVPD, restricted  —  the level change')
ax.plot(x, E_C, color=C_UN, lw=2.0, zorder=3,
        label=r'$\omega$B97M-V/def2-TZVPD, ground state  —  where OMol25 labels')

# der geerbte Punkt und die vertikale Linie, auf der OMol25 abliest
ax.vlines(x0, -0.05, E_B[i0] + 0.02, color=GREY, lw=0.8, ls=':', zorder=1)
ax.plot([x0], [E_A[i0]], 'o', ms=8, color=LIGHT, mec='white', mew=1.2, zorder=6)
ax.plot([x0], [E_B[i0]], 'o', ms=8, color=C_ST, mec='white', mew=1.2, zorder=6)
ax.plot([x0], [E_C[i0]], 'o', ms=8, color=C_UN, mec='white', mew=1.2, zorder=6)
ax.annotate('Transition1x transition state\n(a saddle here, $f_\\mathrm{ref}\\approx 0$)',
            xy=(x0, E_A[i0]), xytext=(0.06, 1.28), fontsize=8, color=GREY,
            arrowprops=dict(arrowstyle='-', color=GREY, lw=0.7))
ax.annotate('OMol25 reads $E$ and $F$ here',
            xy=(x0, E_C[i0]), xytext=(0.04, 0.62), fontsize=8, color=C_UN,
            arrowprops=dict(arrowstyle='-', color=C_UN, lw=0.7))

# Restkraefte als waagerechte Pfeile entlang der Reaktionskoordinate. Am
# geerbten Punkt steigen B und C noch an (ihre Saettel liegen rechts), die
# Kraft zeigt also nach links, zurueck zum Edukt. Laengen im gemessenen
# Medianverhaeltnis f_BS/f_RKS = 2.8; nicht massstaeblich.
for E, c, name, L, dy in ((E_B, C_ST, r'$f_\mathrm{RKS}$', 0.07, 0.07),
                          (E_C, C_UN, r'$f_\mathrm{BS}$', 0.196, -0.08)):
    y = E[i0]
    ax.annotate('', xy=(x0 - L, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=2.4,
                                mutation_scale=14, shrinkA=4, shrinkB=0),
                zorder=7)
    ax.text(x0 - L / 2, y + dy, name, color=c, fontsize=10, fontweight='bold',
            va='center', ha='center', zorder=8)

# wo ein relaxierter Pfad seinen Sattel haette
for xs, E, c, txt in ((xB, E_B, C_ST, 'RKS saddle\nat OMol25 level'),
                      (xC, E_C, C_UN, 'ground-state\nsaddle')):
    ax.plot([xs], [E.max()], 'o', ms=9, mfc='white', mec=c, mew=1.8, zorder=6)
    ax.annotate(txt, xy=(xs, E.max()), xytext=(xs + 0.10, E.max() + 0.12),
                fontsize=8, color=c, ha='left',
                arrowprops=dict(arrowstyle='-', color=c, lw=0.7))

# Brechungstiefe
xd = 0.70
ax.annotate('', xy=(xd, E_C[np.argmin(np.abs(x - xd))]),
            xytext=(xd, E_B[np.argmin(np.abs(x - xd))]),
            arrowprops=dict(arrowstyle='<->', color=GREY, lw=0.9,
                            shrinkA=0, shrinkB=0))
ax.text(xd + 0.012, 0.5 * (E_B[np.argmin(np.abs(x - xd))]
                           + E_C[np.argmin(np.abs(x - xd))]),
        'breaking\ndepth', fontsize=8, color=GREY, va='center')

ax.text(0.985, 0.03,
        'medians over the 18 unstable reactions,\nat the Transition1x geometry:\n'
        r'$f_\mathrm{ref}$ 0.014   $f_\mathrm{RKS}$ 0.59   $f_\mathrm{BS}$ 1.64  eV Å$^{-1}$',
        transform=ax.transAxes, fontsize=7.5, color=GREY, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ddd', lw=0.6))

ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1.55)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('reaction coordinate')
ax.set_ylabel('energy')
ax.set_title('One inherited geometry, three surfaces  —  an RKS-unstable reaction',
             loc='left', fontsize=10, fontweight='bold', pad=8)
ax.legend(loc='lower left', fontsize=7.4, frameon=True, framealpha=0.95,
          edgecolor='#ddd', borderaxespad=0.3)

os.makedirs(FIG, exist_ok=True)
p = os.path.join(FIG, 'fig_schematic_labels.png')
fig.savefig(p, bbox_inches='tight', facecolor='white')
print('geschrieben:', os.path.relpath(p, HERE))
