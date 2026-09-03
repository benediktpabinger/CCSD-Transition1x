# -*- coding: utf-8 -*-
"""Schema in zwei Zeilen: oben das Transition1x-Niveau, unten das OMol25-Niveau.

Oben links   die Transition1x-Flaeche (wB97x/6-31G(d), restringiert) von
             oben. Das Band liegt auf ihrem Minimum-Energie-Pfad, sein
             hoechstes Bild ist ihr Sattel -- hier wurde es relaxiert.
Oben rechts  Energie entlang des Bandes auf dieser Flaeche. Eine Kurve.
Unten links  die UKS-Grundzustandsflaeche auf OMol25-Niveau (wB97M-V/
             def2-TZVPD) von oben. Dasselbe Band; der Pfad dieser Flaeche
             weicht seitlich aus, die Pfeile sind der zum Band senkrechte
             Kraftanteil.
Unten rechts Energie entlang des Bandes: Transition1x-Flaeche zum Vergleich,
             dazu RKS und UKS auf OMol25-Niveau. RKS = UKS an den Enden, in
             der Mitte faellt UKS ab.

Keine Daten, nicht massstaeblich.

figures/fig_schematic_pes2.png
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
C_T1X = '#555555'
BAND = '#111111'

LVL_T1X = r'$\omega$B97x/6-31G(d), restricted'
LVL_OM = r'$\omega$B97M-V/def2-TZVPD'

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 200, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ----------------------------------------------------------- die Flaechen
XR, XT = 0.62, 0.53                              # Saettel: RKS (OMol25), Transition1x
XC, WX = 0.60, 0.28                              # Fenster, in dem UKS unter RKS liegt

def wells(x):
    return (- 0.35 * np.exp(-((x - 0.12) / 0.16) ** 2)
            - 0.35 * np.exp(-((x - 1.04) / 0.16) ** 2))

def E_T(x, y=0.0):                               # Transition1x-Flaeche
    return 1.12 * np.exp(-((x - XT) / 0.19) ** 2) + 1.6 * y ** 2 + wells(x)

def E_R(x, y):                                   # restringiert, OMol25-Niveau
    return 1.3 * np.exp(-((x - XR) / 0.20) ** 2) + 1.6 * y ** 2 + wells(x)

def E_B(x, y):                                   # Grundzustand, OMol25-Niveau
    t = (np.asarray(x) - XC) / WX
    win = np.where(np.abs(t) < 1.0,
                   np.cos(0.5 * np.pi * np.clip(t, -1, 1)) ** 2, 0.0)
    dip = 0.75 * win * np.exp(-((y - 0.28) / 0.34) ** 2)
    return E_R(x, y) - dip

def grad_B(x, y, h=1e-4):
    return np.array([(E_B(x + h, y) - E_B(x - h, y)) / (2 * h),
                     (E_B(x, y + h) - E_B(x, y - h)) / (2 * h)])

xg = np.linspace(0.0, 1.18, 320)
yg = np.linspace(-0.42, 0.62, 280)
X, Y = np.meshgrid(xg, yg)
ZT, ZB = E_T(X, Y), E_B(X, Y)

# Pfad der UKS-Flaeche: je x das y kleinster Energie
yfine = np.linspace(-0.42, 0.62, 900)
mx = np.linspace(0.12, 1.04, 260)
my = np.array([yfine[np.argmin(E_B(x, yfine))] for x in mx])
mz = E_B(mx, my)
kB = int(np.argmax(mz))

# das Band: zehn Bilder auf y = 0, Edukt bis Produkt
N = 10
bx = np.linspace(0.12, 1.04, N)
by = np.zeros(N)
zT, zR, zB = E_T(bx), E_R(bx, by), E_B(bx, by)
kS = int(np.argmax(zT))                          # hoechstes Bild = T1x-Sattel

tang = np.gradient(np.c_[bx, by], axis=0)
tang /= np.linalg.norm(tang, axis=1)[:, None]
F = np.array([-grad_B(x, y) for x, y in zip(bx, by)])
Fperp = F - (np.sum(F * tang, axis=1))[:, None] * tang

xs = np.linspace(bx[0], bx[-1], 400)


def tag(ax, x, y, text, dx=8, dy=8, ha='left', va='bottom', fs=8):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points',
                fontsize=fs, color='white', ha=ha, va=va, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', fc='black', ec='none',
                          alpha=0.85), zorder=12)


def pes_map(ax, Z, title):
    levels = np.linspace(Z.min(), Z.max(), 40)
    ax.contourf(X, Y, Z, levels=levels, cmap='turbo')
    ax.contour(X, Y, Z, levels=levels[::2], colors='white', linewidths=0.5,
               linestyles='--', alpha=0.7)
    ax.plot(bx, by, color='white', lw=3.2, zorder=4)
    ax.plot(bx, by, color=BAND, lw=1.8, ls='--', zorder=5)
    ax.plot(bx, by, 'o', ms=5, mfc='white', mec=BAND, mew=1.0, zorder=9)
    ax.plot([bx[0], bx[-1]], [0, 0], 'o', ms=10, mfc='white', mec='black',
            mew=1.2, zorder=10)
    ax.plot([bx[kS]], [0], '*', ms=15, mfc='white', mec='black', mew=1.0,
            zorder=10)
    tag(ax, bx[0], 0, 'Reactant', dx=-6, dy=10)
    tag(ax, bx[-1], 0, 'Product', dx=-6, dy=-14, ha='right', va='top')
    ax.set_xlim(0.0, 1.18); ax.set_ylim(-0.42, 0.62)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('reaction coordinate')
    ax.set_ylabel('orthogonal coordinate')
    ax.set_title(title, loc='left', fontsize=10, fontweight='bold')


# ------------------------------------------------------------------ Bild
fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.2),
                        gridspec_kw={'width_ratios': [1.25, 1.0],
                                     'wspace': 0.18, 'hspace': 0.42})
(aT, sT), (aB, sB) = axs
fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.06)

# ================================================ oben: Transition1x-Niveau
pes_map(aT, ZT, 'Transition1x surface, from above  —  ' + LVL_T1X)
tag(aT, bx[kS], 0, 'Transition1x transition state\n= saddle of this surface',
    dx=-10, dy=-14, ha='right', va='top')
tag(aT, bx[3], 0, 'Transition1x band, relaxed here:\nit lies on the '
    'minimum-energy path', dx=-4, dy=-44, ha='center', va='top')
aT.text(0.99, 0.02, 'residual force on this surface: ≈ 0  (converged here)',
        transform=aT.transAxes, fontsize=7.5, color='white', ha='right',
        va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='black',
                               ec='none', alpha=0.7))

sT.plot(xs, E_T(xs), color=C_T1X, lw=2.6, zorder=3,
        label='Transition1x surface  (' + LVL_T1X + ')')
sT.plot(bx, zT, 'o', ms=4.5, mfc='white', mec=C_T1X, mew=1.2, zorder=5)
sT.plot([bx[kS]], [zT[kS]], '*', ms=16, mfc='white', mec='black', mew=1.0,
        zorder=8)
sT.annotate('Transition1x transition state:\nthe top image sits at the saddle',
            (bx[kS], zT[kS]), xytext=(bx[kS] + 0.16, zT[kS] + 0.22),
            textcoords='data', fontsize=8, ha='left', va='bottom',
            arrowprops=dict(arrowstyle='-', color='black', lw=0.7, shrinkB=6))
sT.legend(loc='upper left', fontsize=7.5, frameon=False)
sT.set_xlim(bx[0] - 0.04, bx[-1] + 0.04)
sT.set_ylim(E_T(bx[0]) - 0.20, zR.max() + 0.50)
sT.set_xticks([]); sT.set_yticks([])
sT.set_xlabel('reaction coordinate, along the Transition1x band')
sT.set_ylabel('energy')
sT.set_title('Energy along the band  —  Transition1x level', loc='left',
             fontsize=10, fontweight='bold')

# ================================================== unten: OMol25-Niveau
pes_map(aB, ZB, 'UKS ground-state surface, from above  —  ' + LVL_OM)
aB.plot(mx, my, color='white', lw=3.4, zorder=4)
aB.plot(mx, my, color=C_UKS, lw=2.0, zorder=5)
aB.plot([mx[kB]], [my[kB]], '*', ms=15, mfc='white', mec='black', mew=1.0,
        zorder=10)
scale = 0.15 / np.abs(Fperp).max()
for k in range(N):
    d = Fperp[k] * scale
    if np.linalg.norm(d) < 0.025:
        continue
    for c, lw, ms_ in (('white', 3.2, 15), (BAND, 1.6, 13)):
        aB.annotate('', xy=(bx[k] + d[0], by[k] + d[1]), xytext=(bx[k], by[k]),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                                    mutation_scale=ms_, shrinkA=3, shrinkB=0),
                    zorder=7 if c == 'white' else 8)
tag(aB, bx[kS], 0, 'Transition1x transition state\n(same geometry as above)',
    dx=-10, dy=-14, ha='right', va='top')
tag(aB, mx[kB], my[kB], 'UKS saddle', dx=10, dy=6)
tag(aB, mx[kB + 30], my[kB + 30], 'MEP on the UKS surface', dx=12, dy=10)
tag(aB, bx[3], 0, 'same Transition1x band', dx=-4, dy=-40, ha='center',
    va='top')
aB.text(0.99, 0.02, 'arrows: force perpendicular to the band\non the UKS '
        'surface, common scale', transform=aB.transAxes, fontsize=7.5,
        color='white', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.7))

eR, eB = E_R(xs, 0.0), E_B(xs, 0.0)
sB.fill_between(xs, eB, eR, color=C_UKS, alpha=0.15, lw=0, zorder=1)
sB.plot(xs, E_T(xs), color='#9a9a9a', lw=1.6, ls='--', zorder=2,
        label='Transition1x surface  (' + LVL_T1X + ')')
sB.plot(xs, eR, color=C_RKS, lw=2.6, zorder=3,
        label='RKS surface  (' + LVL_OM + ', restricted)')
sB.plot(xs, eB, color=C_UKS, lw=2.6, zorder=3,
        label='UKS ground-state surface  (' + LVL_OM + ', unrestricted)')
sB.plot(bx, zT, 'o', ms=4, mfc='white', mec='#9a9a9a', mew=1.1, zorder=5)
sB.plot(bx, zR, 'o', ms=4.5, mfc='white', mec=C_RKS, mew=1.2, zorder=5)
sB.plot(bx, zB, 'o', ms=4.5, mfc='white', mec=C_UKS, mew=1.2, zorder=5)
for k in range(N):
    sB.plot([bx[k], bx[k]], [zB[k], zR[k]], color='#888', lw=0.8, ls=':',
            zorder=2)
sB.plot([bx[kS]], [zT[kS]], '*', ms=16, mfc='white', mec='black', mew=1.0,
        zorder=8)
sB.plot([bx[kS], bx[kS]], [zT[kS], zR[kS]], color='black', lw=0.8, ls=':',
        zorder=6)
sB.annotate('Transition1x transition state:\na saddle above, a slope on both\n'
            'surfaces at this level',
            (bx[kS], zT[kS]), xytext=(bx[kS] + 0.34, zR.max() - 0.40),
            textcoords='data', fontsize=8, ha='left', va='top',
            arrowprops=dict(arrowstyle='-', color='black', lw=0.7, shrinkB=6))
kD = int(np.argmax(zR - zB))
sB.annotate('', xy=(bx[kD], zB[kD]), xytext=(bx[kD], zR[kD]),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.1,
                            shrinkA=0, shrinkB=0), zorder=7)
sB.annotate('breaking depth', (bx[kD], 0.5 * (zB[kD] + zR[kD])),
            xytext=(14, 0), textcoords='offset points', fontsize=8,
            va='center')
yb = E_R(bx[0], 0) - 0.05
for x0, x1 in ((bx[0], XC - WX), (XC + WX, bx[-1])):
    sB.annotate('', xy=(x1, yb), xytext=(x0, yb),
                arrowprops=dict(arrowstyle='|-|', color='#444', lw=0.9,
                                mutation_scale=3, shrinkA=0, shrinkB=0))
    sB.text(0.5 * (x0 + x1), yb - 0.035, 'RKS = UKS', fontsize=7.5,
            color='#444', ha='center', va='top')
sB.text(0.84, E_R(bx[0], 0) + 0.10,
        'the UKS saddle lies off this cut (left panel)',
        fontsize=7.5, color='#444', ha='right', va='bottom')
sB.legend(loc='upper left', fontsize=7.5, frameon=False)
sB.set_xlim(bx[0] - 0.04, bx[-1] + 0.04)
sB.set_ylim(E_R(bx[0], 0) - 0.20, zR.max() + 0.50)
sB.set_xticks([]); sB.set_yticks([])
sB.set_xlabel('reaction coordinate, along the Transition1x band')
sB.set_ylabel('energy')
sB.set_title('Energy along the band  —  OMol25 level, both surfaces',
             loc='left', fontsize=10, fontweight='bold')

fig.suptitle('The same ten geometries at two levels of theory  —  '
             'RKS-unstable reaction, sketch',
             x=0.04, ha='left', fontsize=12, fontweight='bold', y=0.975)
fig.text(0.04, 0.935, 'top: the level the band was relaxed at.   '
         'bottom: the level OMol25 labelled it at.', fontsize=9, color='#444')

os.makedirs(FIG, exist_ok=True)
p = os.path.join(FIG, 'fig_schematic_pes2.png')
fig.savefig(p, bbox_inches='tight', facecolor='white')
print('geschrieben:', os.path.relpath(p, HERE))
