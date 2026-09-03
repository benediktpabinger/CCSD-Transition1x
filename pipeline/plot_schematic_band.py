# -*- coding: utf-8 -*-
"""Schema: eine spingebrochene PES, darauf ein Transition1x-Band mit zehn
Bildern, das auf dieser Flaeche nicht relaxiert ist, und die NEB-Restkraefte,
die es zum eigentlichen Sattel ziehen.

Keine Daten -- eine Skizze. Eine Flaeche (Grundzustand auf OMol25-Niveau),
schraeg von oben. Das Band kommt aus Transition1x, wurde also auf einer anderen
Flaeche relaxiert: hier laeuft es versetzt neben dem Minimum-Energie-Pfad
ueber den Hang, und sein hoechstes Bild liegt nicht am Sattel. Die Pfeile sind
die zum Pfad senkrechten Kraftanteile -- das, was ein NEB an jedem Bild sieht
-- und sie zeigen alle auf den relaxierten Pfad und seinen Sattel.

figures/fig_schematic_band.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, 'figures')

C_UN = '#c2542a'
DARK = '#333333'
GREY = '#6b6b6b'

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9})

# ------------------------------------------------------------ die Flaeche
XS = 0.62                                       # Sattel auf dieser Flaeche
def E(x, y):
    return 1.0 * np.exp(-((x - XS) / 0.20) ** 2) + 1.6 * y ** 2

def grad(x, y, h=1e-4):
    return np.array([(E(x + h, y) - E(x - h, y)) / (2 * h),
                     (E(x, y + h) - E(x, y - h)) / (2 * h)])

xg = np.linspace(0.0, 1.2, 160)
yg = np.linspace(-0.45, 0.45, 120)
X, Y = np.meshgrid(xg, yg)
Z = E(X, Y)

# ------------------------------------------- das geerbte Band, zehn Bilder
N = 10
bx = np.linspace(0.06, 1.06, N)
by = -0.22 - 0.06 * np.sin(np.pi * (bx - 0.06) / 1.00)
bz = E(bx, by)

# NEB-Restkraft je Bild: wahre Kraft minus Anteil entlang der Bandtangente
tang = np.gradient(np.c_[bx, by], axis=0)
tang /= np.linalg.norm(tang, axis=1)[:, None]
F = np.array([-grad(x, y) for x, y in zip(bx, by)])
Fperp = F - (np.sum(F * tang, axis=1))[:, None] * tang

# ------------------------------------------------------------------ Bild
fig = plt.figure(figsize=(8.2, 5.6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=50, azim=-52)
ax.computed_zorder = False

cmap = LinearSegmentedColormap.from_list('pes', ['#f4f1ec', '#d9d3c7', '#b9b0a0'])
ax.plot_surface(X, Y, Z, cmap=cmap, rstride=2, cstride=2, linewidth=0.15,
                edgecolor='#a09888', alpha=0.70, antialiased=True, zorder=1)
ax.contour(X, Y, Z, levels=12, zdir='z', offset=-0.55, colors='#bbb',
           linewidths=0.6, zorder=0)

# der relaxierte Pfad auf dieser Flaeche und sein Sattel
mx = np.linspace(0.02, 1.15, 200)
ax.plot(mx, np.zeros_like(mx), E(mx, 0.0) + 0.02, color=C_UN, lw=1.6,
        ls=':', zorder=5)
ax.plot([XS], [0.0], [E(XS, 0.0) + 0.04], 'o', ms=10, mfc='white', mec=C_UN,
        mew=2.0, zorder=10)

# das Band
ax.plot(bx, by, bz + 0.03, color=DARK, lw=1.8, ls='--', zorder=6)
for k in range(N):
    hi = k == int(np.argmax(bz))
    ax.plot([bx[k]], [by[k]], [bz[k] + 0.04], 'o', ms=8 if hi else 6,
            mfc=DARK if hi else 'white', mec=DARK, mew=1.6, zorder=9)

# die senkrechten Restkraefte, eine gemeinsame Skala
scale = 0.21 / np.abs(Fperp).max()
for k in range(N):
    d = Fperp[k] * scale
    if np.linalg.norm(d) < 0.01:
        continue
    qx, qy = bx[k] + d[0], by[k] + d[1]
    ax.quiver(bx[k], by[k], bz[k] + 0.08, d[0], d[1], E(qx, qy) - bz[k],
              color=DARK, lw=2.4, arrow_length_ratio=0.4, zorder=11)

# Beschriftungen
ax.text(0.02, -0.44, 1.25,
        'Transition1x band, ten images\n(relaxed on another surface; not here)',
        color=DARK, fontsize=8, ha='left', zorder=12)
ax.text(0.66, 0.14, 1.30,
        'saddle of this surface\n(where a relaxed band ends up)',
        color=C_UN, fontsize=8, ha='left', zorder=12)
ax.text2D(0.02, 0.04,
          'arrows: force component perpendicular to the band at each image\n'
          '(what a NEB acts on); common scale',
          transform=ax.transAxes, color=DARK, fontsize=7.5, ha='left',
          va='bottom', zorder=12)

ax.text2D(0.99, 0.97,
          'ground-state surface at $\\omega$B97M-V/def2-TZVPD, '
          'RKS-unstable reaction\nsketch, not to scale',
          transform=ax.transAxes, fontsize=7.5, color=GREY, va='top',
          ha='right',
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
ax.set_title('An inherited band on the surface it was labelled on',
             loc='left', fontsize=10, fontweight='bold', pad=0)
ax.set_box_aspect((1.6, 1.0, 0.75), zoom=0.95)

os.makedirs(FIG, exist_ok=True)
p = os.path.join(FIG, 'fig_schematic_band.png')
fig.savefig(p, bbox_inches='tight', facecolor='white')
print('geschrieben:', os.path.relpath(p, HERE))
