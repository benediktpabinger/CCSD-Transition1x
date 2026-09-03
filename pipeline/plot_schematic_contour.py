# -*- coding: utf-8 -*-
"""Schema als Konturkarte plus Energieprofil: zwei Pfade.

Links die Grundzustandsflaeche von oben, als Hoehenlinien. Darauf das
Transition1x-Band (zehn Bilder, auf der restringierten Flaeche relaxiert,
deren Minimum-Energie-Pfad hier bei y = 0 liegt) und der Pfad, auf dem es auf
dieser Flaeche laege. Die Pfeile sind an jedem Bild der zum Band senkrechte
Kraftanteil auf der Grundzustandsflaeche.

Rechts die Energie entlang der Reaktionskoordinate: die zehn Geometrien auf
der restringierten Flaeche (wo relaxiert), dieselben zehn auf der
Grundzustandsflaeche (wo gelabelt), und der relaxierte Pfad der
Grundzustandsflaeche. Die senkrechten Verbindungen sind die Brechungstiefe je
Bild.

Keine Daten, nicht massstaeblich.

figures/fig_schematic_contour.png
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
DARK = '#2b2b2b'
GREY = '#6b6b6b'

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ----------------------------------------------------------- die Flaechen
XR = 0.55
def E_R(x, y):                                  # restringiert
    return 1.3 * np.exp(-((x - XR) / 0.20) ** 2) + 1.6 * y ** 2

def E_B(x, y):                                  # Grundzustand
    # Senke AUF dem Grat, nicht daneben -- sonst entsteht ein Loch unter dem
    # Eduktniveau statt eines abgesenkten Sattels
    dip = 0.65 * np.exp(-((x - 0.60) / 0.20) ** 2 - ((y - 0.28) / 0.34) ** 2)
    return E_R(x, y) - dip

def grad_B(x, y, h=1e-4):
    return np.array([(E_B(x + h, y) - E_B(x - h, y)) / (2 * h),
                     (E_B(x, y + h) - E_B(x, y - h)) / (2 * h)])

xg = np.linspace(0.0, 1.2, 300)
yg = np.linspace(-0.40, 0.60, 260)
X, Y = np.meshgrid(xg, yg)
ZB = E_B(X, Y)

# relaxierter Pfad der Grundzustandsflaeche: je x das y kleinster Energie
yfine = np.linspace(-0.40, 0.60, 900)
mx = np.linspace(0.02, 1.15, 240)
my = np.array([yfine[np.argmin(E_B(x, yfine))] for x in mx])
mz = E_B(mx, my)
kB = int(np.argmax(mz))

# das Band: zehn Bilder auf dem RKS-Pfad, y = 0
N = 10
bx = np.linspace(0.06, 1.06, N)
by = np.zeros(N)
zR, zB = E_R(bx, by), E_B(bx, by)
kR = int(np.argmax(zR))

tang = np.gradient(np.c_[bx, by], axis=0)
tang /= np.linalg.norm(tang, axis=1)[:, None]
F = np.array([-grad_B(x, y) for x, y in zip(bx, by)])
Fperp = F - (np.sum(F * tang, axis=1))[:, None] * tang

# ------------------------------------------------------------------ Bild
fig, (ax, bx_) = plt.subplots(1, 2, figsize=(11.6, 4.6),
                              gridspec_kw={'width_ratios': [1.25, 1.0],
                                           'wspace': 0.22})
fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.14)

# ---- links: Konturkarte
levels = np.linspace(ZB.min(), ZB.max(), 22)
ax.contourf(X, Y, ZB, levels=levels, cmap='Greys', alpha=0.28)
ax.contour(X, Y, ZB, levels=levels, colors='#8a8a8a', linewidths=0.5)

ax.plot(mx, my, color=C_UN, lw=2.2, zorder=4)
ax.plot([mx[kB]], [my[kB]], 'o', ms=9, mfc=C_UN, mec='white', mew=1.4,
        zorder=6)

ax.plot(bx, by, color=DARK, lw=1.8, ls='--', zorder=5)
ax.plot(bx, by, 'o', ms=5, mfc=DARK, mec='white', mew=0.8, zorder=6)
ax.plot([bx[kR]], [by[kR]], 'o', ms=9, mfc=C_ST, mec='white', mew=1.4,
        zorder=7)

scale = 0.16 / np.abs(Fperp).max()
for k in range(N):
    d = Fperp[k] * scale
    if np.linalg.norm(d) < 0.008:
        continue
    ax.annotate('', xy=(bx[k] + d[0], by[k] + d[1]), xytext=(bx[k], by[k]),
                arrowprops=dict(arrowstyle='-|>', color=DARK, lw=1.8,
                                mutation_scale=13, shrinkA=3, shrinkB=0),
                zorder=8)

ax.annotate('Transition1x band, ten images\nrelaxed on the restricted surface',
            xy=(bx[2], by[2]), xytext=(0.04, -0.30), fontsize=8, color=DARK,
            arrowprops=dict(arrowstyle='-', color=DARK, lw=0.7))
ax.annotate('its top image:\nthe RKS saddle', xy=(bx[kR], by[kR]),
            xytext=(0.30, 0.45), fontsize=8, color=C_ST,
            arrowprops=dict(arrowstyle='-', color=C_ST, lw=0.7))
ax.annotate('where the path lies on\nthe ground-state surface',
            xy=(mx[kB + 40], my[kB + 40]), xytext=(0.78, 0.50), fontsize=8,
            color=C_UN, arrowprops=dict(arrowstyle='-', color=C_UN, lw=0.7))
ax.annotate('its saddle', xy=(mx[kB], my[kB]), xytext=(0.84, 0.18),
            fontsize=8, color=C_UN,
            arrowprops=dict(arrowstyle='-', color=C_UN, lw=0.7))
ax.text(0.62, -0.34, 'arrows: force perpendicular to the band\n'
        'on the ground-state surface, common scale', fontsize=7.5,
        color=DARK, ha='left')

ax.set_xlim(0.0, 1.2); ax.set_ylim(-0.40, 0.60)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('reaction coordinate')
ax.set_ylabel('orthogonal coordinate')
ax.set_title('Ground-state surface from above', loc='left', fontsize=10,
             fontweight='bold')

# ---- rechts: Energieprofil
for k in range(N):
    bx_.plot([bx[k], bx[k]], [zB[k], zR[k]], color=GREY, lw=0.9, ls=':',
             zorder=2)
bx_.plot(bx, zR, color=C_ST, lw=1.8, ls='--', zorder=3)
bx_.plot(bx, zR, 'o', ms=4.5, mfc=C_ST, mec='white', mew=0.8, zorder=4)
bx_.plot(bx, zB, color=DARK, lw=1.8, ls='--', zorder=3)
bx_.plot(bx, zB, 'o', ms=4.5, mfc=DARK, mec='white', mew=0.8, zorder=4)
bx_.plot(mx, mz, color=C_UN, lw=2.2, zorder=3)
bx_.plot([mx[kB]], [mz[kB]], 'o', ms=8, mfc=C_UN, mec='white', mew=1.4,
         zorder=5)
bx_.plot([bx[kR]], [zR[kR]], 'o', ms=8, mfc=C_ST, mec='white', mew=1.4,
         zorder=5)

kD = int(np.argmax(zR - zB))                   # Bild mit groesster Tiefe
bx_.annotate('', xy=(bx[kD] + 0.03, zB[kD]), xytext=(bx[kD] + 0.03, zR[kD]),
             arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.0,
                             shrinkA=0, shrinkB=0))
bx_.text(bx[kD] + 0.12, 0.5 * (zB[kD] + zR[kD]), 'breaking\ndepth',
         fontsize=8, color=GREY, va='center')

bx_.text(0.99, 0.97, 'the ten geometries on the restricted surface\n'
         '(where the band was relaxed)', transform=bx_.transAxes,
         color=C_ST, fontsize=8, va='top', ha='right')
bx_.text(0.99, 0.84, 'the same ten on the ground-state surface\n'
         '(the energies OMol25 labels)', transform=bx_.transAxes,
         color=DARK, fontsize=8, va='top', ha='right')
bx_.text(0.99, 0.71, 'the relaxed path of the ground-state surface',
         transform=bx_.transAxes, color=C_UN, fontsize=8, va='top',
         ha='right')

bx_.set_xlim(0.0, 1.2)
bx_.set_ylim(-0.08, 2.05)
bx_.set_xticks([]); bx_.set_yticks([])
bx_.set_xlabel('reaction coordinate')
bx_.set_ylabel('energy')
bx_.set_title('Energy along the two paths', loc='left', fontsize=10,
              fontweight='bold')

fig.suptitle('Relaxed on the restricted surface, labelled on the ground-state '
             'surface  —  RKS-unstable reaction, sketch',
             x=0.05, ha='left', fontsize=11, fontweight='bold', y=0.98)

os.makedirs(FIG, exist_ok=True)
p = os.path.join(FIG, 'fig_schematic_contour.png')
fig.savefig(p, bbox_inches='tight', facecolor='white')
print('geschrieben:', os.path.relpath(p, HERE))
