# -*- coding: utf-8 -*-
"""Schema in 3D: die Grundzustandsflaeche auf OMol25-Niveau, und wo darauf die
geerbten Punkte liegen.

Keine Daten -- eine Skizze. Eine einzige Flaeche, schraeg von oben betrachtet:
ein Sattel entlang der Reaktionskoordinate x, ein Tal quer dazu in y. Darauf:

    grauer Pfad      der Transition1x-Pfad, auf einer anderen Flaeche
                     relaxiert; sein Uebergangszustand liegt hier am Hang
    tealer Marker    wo eine Nachoptimierung auf der RESTRINGIERTEN Flaeche
                     desselben Niveaus landet -- naeher, aber immer noch am Hang
    oranger Marker   der Sattel dieser Flaeche, wo ein relaxierter Pfad laege
    Pfeile           die Restkraft an den zwei geerbten Punkten, bergab

Die Mediane in der Textbox stammen aus results/hinge_t1x.csv und
results/hinge_omol25.csv; Flaechenform und Lagen sind frei gewaehlt.

figures/fig_schematic_3d.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, 'figures')

C_ST = '#2a6f7f'
C_UN = '#c2542a'
GREY = '#6b6b6b'
LIGHT = '#8c8c8c'

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9})

# ------------------------------------------------------------ die Flaeche
XS, YS = 0.62, 0.0                      # Sattel der Grundzustandsflaeche
def E(x, y):
    return 1.0 * np.exp(-((x - XS) / 0.20) ** 2) + 1.6 * (y - YS) ** 2

def grad(x, y, h=1e-4):
    return np.array([(E(x + h, y) - E(x - h, y)) / (2 * h),
                     (E(x, y + h) - E(x, y - h)) / (2 * h)])

xg = np.linspace(0.0, 1.2, 160)
yg = np.linspace(-0.45, 0.45, 120)
X, Y = np.meshgrid(xg, yg)
Z = E(X, Y)

# ------------------------------------------------------- die drei Punkte
P_T1X = (0.40, -0.24)                   # Transition1x-Uebergangszustand
P_RKS = (0.54, -0.11)                   # nach Nachoptimierung, restringiert
P_BS = (XS, YS)                         # Sattel dieser Flaeche

# der geerbte Pfad: leicht versetzt und gebogen, ueber den Hang
tp = np.linspace(0.05, 1.10, 200)
path_y = -0.24 - 0.06 * np.sin(np.pi * (tp - 0.05) / 1.05)
path_z = E(tp, path_y) + 0.03          # minimal ueber der Flaeche, sichtbar

# ------------------------------------------------------------------ Bild
fig = plt.figure(figsize=(8.2, 5.6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=50, azim=-52)
ax.computed_zorder = False      # sonst verschwinden Pfeile hinter der Flaeche

cmap = LinearSegmentedColormap.from_list('pes', ['#f4f1ec', '#d9d3c7', '#b9b0a0'])
ax.plot_surface(X, Y, Z, cmap=cmap, rstride=2, cstride=2, linewidth=0.15,
                edgecolor='#a09888', alpha=0.70, antialiased=True, zorder=1)
ax.contour(X, Y, Z, levels=12, zdir='z', offset=-0.55, colors='#bbb',
           linewidths=0.6)

# Pfad
ax.plot(tp, path_y, path_z, color='#5a5a5a', lw=2.4, ls='--', zorder=5)

# Punkte
for (px, py), c, mfc, ms in ((P_T1X, '#5a5a5a', '#5a5a5a', 8), (P_RKS, C_ST, 'white', 9),
                             (P_BS, C_UN, 'white', 9)):
    ax.plot([px], [py], [E(px, py) + 0.04], 'o', ms=ms, mfc=mfc, mec=c,
            mew=1.8, zorder=10)
    ax.plot([px, px], [py, py], [-0.55, E(px, py)], color=c, lw=0.7, ls=':', zorder=4)

# Restkraft-Pfeile, bergab entlang der Flaeche; Laengen im Verhaeltnis der
# gemessenen Mediane (1.64 an T1X, 1.87 am RKS-Sattel), nicht massstaeblich
for (px, py), c, L in ((P_T1X, '#222222', 0.22), (P_RKS, C_ST, 0.22)):
    g = -grad(px, py); g /= np.linalg.norm(g)
    qx, qy = px + g[0] * L, py + g[1] * L
    ax.quiver(px, py, E(px, py) + 0.10, qx - px, qy - py,
              E(qx, qy) - E(px, py), color=c, lw=3.0, arrow_length_ratio=0.3,
              zorder=11)

# Beschriftungen, weit auseinander
ax.text(0.02, -0.44, 1.10,
        'Transition1x transition state\n(inherited; a saddle on another surface)',
        color='#5a5a5a', fontsize=8, ha='left', zorder=12)
ax.text(0.68, -0.40, 0.40,
        'RKS saddle at the OMol25 level\n(level fixed, surface not)',
        color=C_ST, fontsize=8, ha='left', zorder=12)
ax.text(0.80, 0.10, 1.20,
        'ground-state saddle\n(where a relaxed path would be)',
        color=C_UN, fontsize=8, ha='left', zorder=12)
ax.text(0.08, -0.22, 0.42, r'$f_\mathrm{BS}$', color='#222222', fontsize=10,
        fontweight='bold', zorder=12)
ax.text(0.27, -0.02, 1.08, r'$f_\mathrm{BS}$', color=C_ST, fontsize=10,
        fontweight='bold', zorder=12)

ax.text2D(0.99, 0.97,
          'ground-state surface at $\\omega$B97M-V/def2-TZVPD, RKS-unstable reaction\n'
          'residual force on this surface, medians:  at the Transition1x point '
          '1.64,  at the RKS saddle 1.87 eV Å$^{-1}$',
          transform=ax.transAxes, fontsize=7.5, color=GREY, va='top', ha='right',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ddd', lw=0.6))

ax.set_xlim(0.0, 1.2); ax.set_ylim(-0.45, 0.45); ax.set_zlim(-0.55, 1.35)
ax.set_xlabel('reaction coordinate', labelpad=4)
ax.set_ylabel('orthogonal coordinate', labelpad=6)
ax.set_zlabel('energy', labelpad=-8)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_ticks([])
    a.pane.fill = False
    a.pane.set_edgecolor('white')
ax.grid(False)
ax.set_title('One surface, three points: none of the inherited saddles is a '
             'saddle here', loc='left', fontsize=10, fontweight='bold', pad=0)
ax.set_box_aspect((1.6, 1.0, 0.75), zoom=0.95)

os.makedirs(FIG, exist_ok=True)
p = os.path.join(FIG, 'fig_schematic_3d.png')
fig.savefig(p, bbox_inches='tight', facecolor='white')
print('geschrieben:', os.path.relpath(p, HERE))
