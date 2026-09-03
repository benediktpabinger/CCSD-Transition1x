# -*- coding: utf-8 -*-
"""Schema, zweite Fassung: zwei Flaechen uebereinander.

Oben, durchscheinend, die RESTRINGIERTE Flaeche -- darauf wurde das Band
relaxiert, dort folgt es dem Minimum-Energie-Pfad und sein hoechstes Bild ist
der RKS-Sattel. Darunter, gefuellt, die spingebrochene Grundzustandsflaeche:
sie faellt mit der oberen zusammen, wo die geschlossenschalige Loesung stabil
ist (Edukt, Produkt), und sinkt in der Sattelregion darunter ab. Ihr Sattel
liegt woanders.

Dieselben zehn Geometrien werden auf beiden Flaechen gezeichnet. Die
senkrechten Verbindungen sind die Brechungstiefe je Bild. Auf der unteren
Flaeche -- der, auf der OMol25 labelt -- sind die Pfeile der zum Band
senkrechte Kraftanteil: er zieht das Band zum Sattel der unteren Flaeche.

Keine Daten, nicht massstaeblich.

figures/fig_schematic_band2.png
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
DARK = '#333333'
GREY = '#6b6b6b'

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9})

# ----------------------------------------------------------- die Flaechen
XR = 0.55                                       # RKS-Sattel
def E_R(x, y):                                  # restringiert
    return 1.3 * np.exp(-((x - XR) / 0.20) ** 2) + 1.6 * y ** 2

def E_B(x, y):                                  # Grundzustand, gebrochen
    dip = 0.75 * np.exp(-((x - 0.68) / 0.22) ** 2 - ((y - 0.30) / 0.38) ** 2)
    return E_R(x, y) - dip

def grad_B(x, y, h=1e-4):
    return np.array([(E_B(x + h, y) - E_B(x - h, y)) / (2 * h),
                     (E_B(x, y + h) - E_B(x, y - h)) / (2 * h)])

xg = np.linspace(0.0, 1.2, 150)
yg = np.linspace(-0.40, 0.50, 110)
X, Y = np.meshgrid(xg, yg)
ZR, ZB = E_R(X, Y), E_B(X, Y)

# Minimum-Energie-Pfad der unteren Flaeche: je x das y mit kleinster Energie
yfine = np.linspace(-0.40, 0.50, 700)
mx = np.linspace(0.02, 1.15, 220)
my = np.array([yfine[np.argmin(E_B(x, yfine))] for x in mx])
mz = E_B(mx, my)
kB = int(np.argmax(mz))                         # Sattel der unteren Flaeche

# ---------------------------------- das Band: zehn Bilder, relaxiert auf RKS
N = 10
bx = np.linspace(0.06, 1.06, N)
by = np.zeros(N)                                # der RKS-MEP liegt bei y = 0
zR, zB = E_R(bx, by), E_B(bx, by)

tang = np.gradient(np.c_[bx, by], axis=0)
tang /= np.linalg.norm(tang, axis=1)[:, None]
F = np.array([-grad_B(x, y) for x, y in zip(bx, by)])
Fperp = F - (np.sum(F * tang, axis=1))[:, None] * tang

# ------------------------------------------------------------------ Bild
fig = plt.figure(figsize=(8.8, 6.0))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=32, azim=-46)
ax.computed_zorder = False

cm_b = LinearSegmentedColormap.from_list('bs', ['#f7e6db', '#e8bfa6', '#cf9678'])
cm_r = LinearSegmentedColormap.from_list('rks', ['#dfeff1', '#b7d6db', '#8fbdc4'])
ax.contour(X, Y, ZB, levels=12, zdir='z', offset=-0.55, colors='#ccc',
           linewidths=0.6, zorder=0)
ax.plot_surface(X, Y, ZB, cmap=cm_b, rstride=2, cstride=2, linewidth=0.1,
                edgecolor='#c9a08a', alpha=0.90, antialiased=True, zorder=1)
ax.plot_surface(X, Y, ZR, cmap=cm_r, rstride=3, cstride=3, linewidth=0.15,
                edgecolor=C_ST, alpha=0.30, antialiased=True, zorder=2)

# MEP und Sattel der unteren Flaeche
ax.plot(mx, my, mz + 0.02, color=C_UN, lw=1.8, ls=':', zorder=5)
ax.plot([mx[kB]], [my[kB]], [mz[kB] + 0.04], 'o', ms=7, mfc=C_UN,
        mec=C_UN, mew=0, zorder=10)

# das Band oben (wo es relaxiert wurde) und unten (wo es gelabelt wird)
ax.plot(bx, by, zR + 0.02, color=C_ST, lw=1.4, ls='--', alpha=0.9, zorder=6)
ax.plot(bx, by, zB + 0.03, color=DARK, lw=1.8, ls='--', zorder=7)
kR = int(np.argmax(zR))
for k in range(N):
    ax.plot([bx[k], bx[k]], [by[k], by[k]], [zB[k], zR[k]], color=GREY,
            lw=0.9, ls=':', zorder=6)
    ax.plot([bx[k]], [by[k]], [zR[k] + 0.03], 'o', ms=3.5, mfc=C_ST,
            mec=C_ST, mew=0, zorder=9)
    ax.plot([bx[k]], [by[k]], [zB[k] + 0.04], 'o', ms=5 if k == kR else 3.5,
            mfc=DARK, mec=DARK, mew=0, zorder=9)
ax.plot([bx[kR]], [by[kR]], [zR[kR] + 0.04], 'o', ms=6, mfc=C_ST,
        mec=C_ST, mew=0, zorder=10)

# senkrechte Restkraefte auf der unteren Flaeche, gemeinsame Skala
scale = 0.30 / np.abs(Fperp).max()
for k in range(N):
    d = Fperp[k] * scale
    if np.linalg.norm(d) < 0.012:
        continue
    qx, qy = bx[k] + d[0], by[k] + d[1]
    ax.quiver(bx[k], by[k], zB[k] + 0.08, d[0], d[1], E_B(qx, qy) - zB[k],
              color=DARK, lw=2.4, arrow_length_ratio=0.4, zorder=11)

# Beschriftungen
ax.text2D(0.01, 0.74,
          'restricted surface (translucent):\nthe band was relaxed here; '
          'its top image is the RKS saddle',
          transform=ax.transAxes, color=C_ST, fontsize=8, ha='left', va='top')
ax.text2D(0.01, 0.62,
          'ground-state surface (filled):\nsame ten geometries, labelled here',
          transform=ax.transAxes, color=DARK, fontsize=8, ha='left', va='top')
ax.text(mx[kB] + 0.06, my[kB] + 0.04, mz[kB] + 0.16,
        'saddle of the\nground-state surface',
        color=C_UN, fontsize=8, ha='left', zorder=12)
ax.text2D(0.99, 0.97,
          'RKS-unstable reaction; sketch, not to scale\n\n'
          'vertical ties: breaking depth per image\n'
          'arrows: force perpendicular to the band on the\n'
          'ground-state surface, common scale -- zero where\n'
          'the surfaces coincide, pulling toward the lower\n'
          'saddle near the barrier',
          transform=ax.transAxes, fontsize=7.3, color=GREY, va='top', ha='right',
          bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='#ddd', lw=0.6))

ax.set_xlim(0.0, 1.2); ax.set_ylim(-0.40, 0.50); ax.set_zlim(-0.55, 1.80)
ax.set_xlabel('reaction coordinate', labelpad=2)
ax.set_ylabel('orthogonal coord.', labelpad=2)
ax.set_zlabel('energy', labelpad=-8)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_ticks([])
    a.pane.fill = False
    a.pane.set_edgecolor('white')
ax.grid(False)
ax.set_title('Relaxed on the restricted surface, labelled on the ground-state '
             'surface', loc='left', fontsize=10, fontweight='bold', pad=0)
ax.set_box_aspect((1.6, 1.0, 1.05), zoom=0.92)

os.makedirs(FIG, exist_ok=True)
p = os.path.join(FIG, 'fig_schematic_band2.png')
fig.savefig(p, bbox_inches='tight', facecolor='white')
print('geschrieben:', os.path.relpath(p, HERE))
