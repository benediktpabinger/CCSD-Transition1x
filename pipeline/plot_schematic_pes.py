# -*- coding: utf-8 -*-
"""Schema im Stil einer PES-Karte: links die UKS-Grundzustandsflaeche von
oben, rechts der Schnitt von der Seite mit RKS- und UKS-Flaeche.

Links: farbige Hoehenkarte der Grundzustandsflaeche. Darauf das
Transition1x-Band (zehn Bilder, auf der restringierten Flaeche relaxiert,
deren Pfad hier geradeaus ueber den Grat laeuft) und der Minimum-Energie-Pfad
der Grundzustandsflaeche, der seitlich ausweicht. Pfeile: der zum Band
senkrechte Kraftanteil auf der Grundzustandsflaeche.

Rechts: Energie entlang des Bandes. Zwei Flaechen im selben Schnitt --
restringiert und Grundzustand. Sie fallen an Edukt und Produkt zusammen; um
den Uebergangszustand sinkt die Grundzustandsflaeche darunter ab. Die
schraffierte Flaeche ist die Brechungstiefe.

Keine Daten, nicht massstaeblich.

figures/fig_schematic_pes.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, 'figures')

C_RKS = '#2a6f7f'
C_UKS = '#c2542a'
BAND = '#111111'

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ----------------------------------------------------------- die Flaechen
XR, XT = 0.62, 0.53                              # Saettel: RKS (OMol25-Niveau), Transition1x
XC, WX = 0.60, 0.28                              # Fenster, in dem UKS unter RKS liegt;
                                                 # symmetrisch zum Band (0.12 .. 1.04)
def wells(x):
    return (- 0.35 * np.exp(-((x - 0.12) / 0.16) ** 2)
            - 0.35 * np.exp(-((x - 1.04) / 0.16) ** 2))

def E_R(x, y):                                  # restringiert, OMol25-Niveau
    return 1.3 * np.exp(-((x - XR) / 0.20) ** 2) + 1.6 * y ** 2 + wells(x)

def E_B(x, y):                                  # Grundzustand: Senke auf dem Grat
    # In x ein Fenster mit hartem Rand (cos^2-Buckel), nicht eine Gauss-Glocke:
    # ausserhalb |x - 0.66| < 0.30 ist die UKS-Flaeche EXAKT die RKS-Flaeche,
    # die beiden Kurven liegen dort aufeinander und trennen sich erst innen.
    t = (np.asarray(x) - XC) / WX
    win = np.where(np.abs(t) < 1.0, np.cos(0.5 * np.pi * np.clip(t, -1, 1)) ** 2,
                   0.0)
    dip = 0.75 * win * np.exp(-((y - 0.28) / 0.34) ** 2)
    return E_R(x, y) - dip

def E_T(x):                                     # Transition1x-Flaeche (billigeres
    return 1.12 * np.exp(-((x - XT) / 0.19) ** 2) + wells(x)   # Niveau), Schnitt y = 0

def grad_B(x, y, h=1e-4):
    return np.array([(E_B(x + h, y) - E_B(x - h, y)) / (2 * h),
                     (E_B(x, y + h) - E_B(x, y - h)) / (2 * h)])

xg = np.linspace(0.0, 1.18, 320)
yg = np.linspace(-0.42, 0.62, 280)
X, Y = np.meshgrid(xg, yg)
ZB = E_B(X, Y)

# Minimum-Energie-Pfad der Grundzustandsflaeche: je x das y kleinster Energie
yfine = np.linspace(-0.42, 0.62, 900)
mx = np.linspace(0.12, 1.04, 260)
my = np.array([yfine[np.argmin(E_B(x, yfine))] for x in mx])
mz = E_B(mx, my)
kB = int(np.argmax(mz))

# das Band: zehn Bilder auf dem RKS-Pfad (y = 0), Edukt bis Produkt
N = 10
bx = np.linspace(0.12, 1.04, N)
by = np.zeros(N)
zR, zB, zT = E_R(bx, by), E_B(bx, by), E_T(bx)
kR = int(np.argmax(zT))        # hoechstes Bild = Sattel der Transition1x-Flaeche

tang = np.gradient(np.c_[bx, by], axis=0)
tang /= np.linalg.norm(tang, axis=1)[:, None]
F = np.array([-grad_B(x, y) for x, y in zip(bx, by)])
Fperp = F - (np.sum(F * tang, axis=1))[:, None] * tang


def tag(ax, x, y, text, dx=8, dy=8, ha='left', va='bottom', fs=8):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points',
                fontsize=fs, color='white', ha=ha, va=va, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc='black', ec='none',
                          alpha=0.85), zorder=12)


# ------------------------------------------------------------------ Bild
fig, (ax, sx) = plt.subplots(1, 2, figsize=(12.0, 4.8),
                             gridspec_kw={'width_ratios': [1.25, 1.0],
                                          'wspace': 0.18})
fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.12)

# ---- links: Karte der Grundzustandsflaeche
levels = np.linspace(ZB.min(), ZB.max(), 40)
ax.contourf(X, Y, ZB, levels=levels, cmap='turbo')
ax.contour(X, Y, ZB, levels=levels[::2], colors='white', linewidths=0.5,
           linestyles='--', alpha=0.7)

ax.plot(mx, my, color='white', lw=3.4, zorder=4)
ax.plot(mx, my, color=C_UKS, lw=2.0, zorder=5)
ax.plot(bx, by, color='white', lw=3.2, ls='-', zorder=4)
ax.plot(bx, by, color=BAND, lw=1.8, ls='--', zorder=5)

scale = 0.15 / np.abs(Fperp).max()
for k in range(N):
    d = Fperp[k] * scale
    if np.linalg.norm(d) < 0.025:      # sonst bleiben weisse Stummel stehen
        continue
    ax.annotate('', xy=(bx[k] + d[0], by[k] + d[1]), xytext=(bx[k], by[k]),
                arrowprops=dict(arrowstyle='-|>', color='white', lw=3.2,
                                mutation_scale=15, shrinkA=3, shrinkB=0),
                zorder=7)
    ax.annotate('', xy=(bx[k] + d[0], by[k] + d[1]), xytext=(bx[k], by[k]),
                arrowprops=dict(arrowstyle='-|>', color=BAND, lw=1.6,
                                mutation_scale=13, shrinkA=3, shrinkB=0),
                zorder=8)

ax.plot(bx, by, 'o', ms=5, mfc='white', mec=BAND, mew=1.0, zorder=9)
ax.plot([bx[0], bx[-1]], [0, 0], 'o', ms=10, mfc='white', mec='black',
        mew=1.2, zorder=10)
ax.plot([bx[kR]], [0], '*', ms=15, mfc='white', mec='black', mew=1.0,
        zorder=10)
ax.plot([mx[kB]], [my[kB]], '*', ms=15, mfc='white', mec='black', mew=1.0,
        zorder=10)

tag(ax, bx[0], 0, 'Reactant', dx=-6, dy=10, ha='left')
tag(ax, bx[-1], 0, 'Product', dx=-6, dy=-14, ha='right', va='top')
tag(ax, bx[kR], 0, 'Transition1x transition state',
    dx=-10, dy=-14, ha='right', va='top')
tag(ax, mx[kB], my[kB], 'UKS saddle', dx=10, dy=6)
tag(ax, mx[kB + 30], my[kB + 30], 'MEP on the UKS surface', dx=12, dy=10)
tag(ax, bx[3], 0, 'Transition1x band', dx=-4, dy=-40, ha='center', va='top')
ax.text(0.99, 0.02, 'arrows: force perpendicular to the band\non the UKS '
        'surface, common scale', transform=ax.transAxes, fontsize=7.5,
        color='white', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.7))

ax.set_xlim(0.0, 1.18); ax.set_ylim(-0.42, 0.62)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('reaction coordinate')
ax.set_ylabel('orthogonal coordinate')
ax.set_title('UKS ground-state surface, from above', loc='left', fontsize=10,
             fontweight='bold')

# ---- rechts: Schnitt entlang des Bandes, beide Flaechen
xs = np.linspace(bx[0], bx[-1], 400)
eR, eB, eT = E_R(xs, 0.0), E_B(xs, 0.0), E_T(xs)
sx.fill_between(xs, eB, eR, color=C_UKS, alpha=0.15, lw=0, zorder=1)
sx.plot(xs, eT, color='#8c8c8c', lw=1.8, ls='--', zorder=2,
        label='Transition1x surface  (ωB97x/6-31G(d), restricted)')
sx.plot(xs, eR, color=C_RKS, lw=2.6, zorder=3,
        label='RKS surface  (ωB97M-V/def2-TZVPD)')
sx.plot(xs, eB, color=C_UKS, lw=2.6, zorder=3,
        label='UKS ground-state surface  (same level)')
sx.plot(bx, zT, 'o', ms=4, mfc='white', mec='#8c8c8c', mew=1.1, zorder=5)
sx.plot(bx, zR, 'o', ms=4.5, mfc='white', mec=C_RKS, mew=1.2, zorder=5)
sx.plot(bx, zB, 'o', ms=4.5, mfc='white', mec=C_UKS, mew=1.2, zorder=5)
for k in range(N):
    sx.plot([bx[k], bx[k]], [zB[k], zR[k]], color='#888', lw=0.8, ls=':',
            zorder=2)

# der geerbte Punkt: Sattel der Transition1x-Flaeche, auf den anderen ein Hang
sx.plot([bx[kR]], [zT[kR]], '*', ms=16, mfc='white', mec='black', mew=1.0,
        zorder=8)
sx.plot([bx[kR], bx[kR]], [zT[kR], zR[kR]], color='black', lw=0.8, ls=':',
        zorder=6)
sx.annotate('Transition1x transition state:\nthe saddle of the Transition1x '
            'surface,\nnot of either surface at the OMol25 level',
            (bx[kR], zT[kR]), xytext=(bx[kR] + 0.22, zR.max() + 0.16),
            textcoords='data', fontsize=8, ha='left', va='bottom',
            arrowprops=dict(arrowstyle='-', color='black', lw=0.7,
                            shrinkB=6))

kD = int(np.argmax(zR - zB))
sx.annotate('', xy=(bx[kD], zB[kD]), xytext=(bx[kD], zR[kD]),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.1,
                            shrinkA=0, shrinkB=0), zorder=7)
sx.annotate('breaking depth', (bx[kD], 0.5 * (zB[kD] + zR[kD])),
            xytext=(14, 0), textcoords='offset points', fontsize=8,
            va='center')

# wo RKS und UKS dieselbe Flaeche sind
yb = E_R(bx[0], 0) - 0.05
# Klammern exakt dort, wo das Fenster die Flaechen identisch laesst
for x0, x1, ha in ((bx[0], XC - WX, 'left'), (XC + WX, bx[-1], 'right')):
    sx.annotate('', xy=(x1, yb), xytext=(x0, yb),
                arrowprops=dict(arrowstyle='|-|', color='#444', lw=0.9,
                                mutation_scale=3, shrinkA=0, shrinkB=0))
    sx.text(0.5 * (x0 + x1), yb - 0.035, 'RKS = UKS', fontsize=7.5,
            color='#444', ha='center', va='top')
sx.text(0.84, E_R(bx[0], 0) + 0.10,
        'the UKS saddle lies off this cut (left panel)',
        fontsize=7.5, color='#444', ha='right', va='bottom')

sx.legend(loc='upper left', fontsize=7.5, frameon=False)
sx.set_xlim(bx[0] - 0.04, bx[-1] + 0.04)
sx.set_ylim(E_R(bx[0], 0) - 0.20, zR.max() + 0.50)
sx.set_xticks([]); sx.set_yticks([])
sx.set_xlabel('reaction coordinate, along the Transition1x band')
sx.set_ylabel('energy')
sx.set_title('Both surfaces, side view along the band', loc='left',
             fontsize=10, fontweight='bold')

fig.suptitle('Relaxed on the restricted surface, labelled on the ground-state '
             'surface  —  RKS-unstable reaction, sketch',
             x=0.04, ha='left', fontsize=11, fontweight='bold', y=0.985)

os.makedirs(FIG, exist_ok=True)
p = os.path.join(FIG, 'fig_schematic_pes.png')
fig.savefig(p, bbox_inches='tight', facecolor='white')
print('geschrieben:', os.path.relpath(p, HERE))
