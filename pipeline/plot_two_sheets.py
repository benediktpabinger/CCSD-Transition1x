"""Why two valid saddles can exist: the ground state is two surfaces, not one.

A schematic. Nothing here is computed from the benchmark -- the two sheets are
analytic model functions chosen to show the topology, and the numbers on the
axes are arbitrary. What it illustrates is a real feature of the data:

  at our rxn1147 structure   <S^2> 0.456   the broken solution is the ground
                                           state and lies lower
  at the UMA-S structure     <S^2> 0.000   no broken solution exists; the
                                           restricted one is externally stable

Two structures, both first-order saddles, both of the ground-state surface --
because "the ground-state surface" is the lower envelope of two solutions of
the SCF equations, and each saddle sits on a different piece of it.

Three panels:
  a  the two sheets, drawn apart so the crossing is visible
  b  the lower envelope, which is what a geometry optimisation actually walks
     on, with its crease and its two passes
  c  a cut through both at the barrier, where the crossing is a plain line
     crossing and the kink in the envelope is obvious
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H = '/home/energy/s242862'

SURF = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
INK3 = '#8a8985'
C_RKS = '#2a78d6'      # categorical slot 1
C_BS = '#eb6834'       # categorical slot 2
C_SEAM = '#4a3aa7'

# The model. E_R is the restricted sheet: a barrier along x with minima at
# x = +-1, the ordinary picture of a bond breaking with the electrons paired.
# E_B is the broken sheet: its valley runs at a different value of the second
# coordinate, it is slightly lower at the barrier, and it does not exist once
# the bond is short again -- past the Coulson-Fischer point it merges into the
# restricted solution rather than staying as a separate state.
YB = 0.85
XCF = 1.35


def E_R(x, y):
    return (x ** 2 - 1) ** 2 + 0.70 * y ** 2


def E_B(x, y):
    return 0.90 * (x ** 2 - 1) ** 2 + 0.70 * (y - YB) ** 2 + 0.02


n = 150
x = np.linspace(-1.75, 1.75, n)
y = np.linspace(-1.05, 1.95, n)
X, Y = np.meshgrid(x, y)
ZR = E_R(X, Y)
ZB = E_B(X, Y)
ZB_vis = np.where(np.abs(X) <= XCF, ZB, np.nan)   # where it exists at all
ZG = np.where(np.isnan(ZB_vis), ZR, np.minimum(ZR, ZB_vis))

# the two passes: (0, 0) on the restricted sheet, (0, YB) on the broken one
P_R = (0.0, 0.0, E_R(0.0, 0.0))
P_B = (0.0, YB, E_B(0.0, YB))

fig = plt.figure(figsize=(15.2, 6.4), facecolor=SURF)

# The seam has a closed form here: setting E_R = E_B and solving for y gives
#   y = [0.1 (x^2 - 1)^2 + 0.52575] / 1.19
# which is cleaner to draw than a scatter of near-equal grid points.
xs = np.linspace(-XCF, XCF, 300)
ys_seam = (0.10 * (xs ** 2 - 1) ** 2 + 0.52575) / 1.19
zs_seam = E_R(xs, ys_seam)

# ---------------------------------------------------------------- a
axa = fig.add_subplot(1, 3, 1, projection='3d', facecolor=SURF)
# One sheet as a mesh and one solid: two semi-transparent solids on top of one
# another read as a single muddy volume and the crossing disappears.
axa.plot_wireframe(X, Y, ZR, color=C_RKS, linewidth=0.45, rstride=7,
                   cstride=7, alpha=0.85)
axa.plot_surface(X, Y, ZB_vis, color=C_BS, alpha=0.80, linewidth=0,
                 antialiased=True, rstride=2, cstride=2, shade=True)
axa.plot(xs, ys_seam, zs_seam, color=C_SEAM, lw=2.2, zorder=20)
axa.set_title('a   two solutions of the same SCF equations',
              fontsize=11.5, color=INK, pad=2, loc='left')

# ---------------------------------------------------------------- b
axb = fig.add_subplot(1, 3, 2, projection='3d', facecolor=SURF)
axb.plot_surface(X, Y, ZG, color='#aab8c8', alpha=1.0, linewidth=0,
                 antialiased=True, rstride=2, cstride=2, shade=True)
axb.plot(xs, ys_seam, zs_seam + 0.012, color=C_SEAM, lw=2.2, zorder=20)
# Markers get a stem down to the surface: a bare 3D scatter is occluded by the
# surface it sits on, and two text labels in the same corner overlapped, so the
# naming moved to the legend below the panel.
for p, col in ((P_R, C_RKS), (P_B, C_BS)):
    axb.plot([p[0], p[0]], [p[1], p[1]], [p[2], p[2] + 0.30],
             color=col, lw=1.3, zorder=29)
    axb.scatter([p[0]], [p[1]], [p[2] + 0.30], s=95, color=col,
                edgecolor='white', linewidth=1.6, depthshade=False, zorder=30)
axb.set_title('b   what an optimisation actually walks on',
              fontsize=11.5, color=INK, pad=2, loc='left')
axb.legend(handles=[
    Line2D([], [], marker='o', ls='', markersize=8, markerfacecolor=C_RKS,
           markeredgecolor='white', label='pass on the restricted sheet'),
    Line2D([], [], marker='o', ls='', markersize=8, markerfacecolor=C_BS,
           markeredgecolor='white', label='pass on the broken sheet'),
    Line2D([], [], color=C_SEAM, lw=2.2, label='the crease where they cross')],
    loc='upper center', bbox_to_anchor=(0.5, 0.10), frameon=False,
    fontsize=8.8, handletextpad=0.5, labelspacing=0.35)

for ax in (axa, axb):
    ax.set_xlabel('reaction coordinate', fontsize=9, color=INK2, labelpad=-4)
    ax.set_ylabel('a second coordinate', fontsize=9, color=INK2, labelpad=-4)
    ax.set_zlabel('energy', fontsize=9, color=INK2, labelpad=-6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=26, azim=-58)
    ax.set_box_aspect((1, 1, 0.62))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(SURF)
        pane.pane.set_edgecolor('#dedcd6')
        pane.pane.set_alpha(1.0)
    ax.grid(False)

# ---------------------------------------------------------------- c
axc = fig.add_subplot(1, 3, 3, facecolor=SURF)
yy = np.linspace(-1.05, 1.95, 400)
axc.plot(yy, E_R(0.0, yy), color=C_RKS, lw=2.0, label='restricted (RKS)')
axc.plot(yy, E_B(0.0, yy), color=C_BS, lw=2.0, label='broken symmetry (BS)')
env = np.minimum(E_R(0.0, yy), E_B(0.0, yy))
axc.plot(yy, env, color='#111111', lw=4.0, alpha=0.18, zorder=0,
         label='the ground state — the lower of the two')
cross = yy[np.argmin(np.abs(E_R(0.0, yy) - E_B(0.0, yy)))]
axc.axvline(cross, color=C_SEAM, lw=1.2, ls=(0, (1, 2)))
axc.annotate('the sheets cross here', xy=(cross, E_R(0.0, cross)),
             xytext=(10, 34), textcoords='offset points', fontsize=9,
             color=C_SEAM)
axc.scatter([0.0], [E_R(0.0, 0.0)], s=62, color=C_RKS, zorder=5,
            edgecolor='white', linewidth=1.2)
axc.scatter([YB], [E_B(0.0, YB)], s=62, color=C_BS, zorder=5,
            edgecolor='white', linewidth=1.2)
axc.annotate('RKS pass', xy=(0.0, E_R(0.0, 0.0)), xytext=(-6, 12),
             textcoords='offset points', fontsize=9, color=C_RKS, ha='right')
axc.annotate('BS pass', xy=(YB, E_B(0.0, YB)), xytext=(8, -16),
             textcoords='offset points', fontsize=9, color=C_BS)
axc.set_xlabel('a second coordinate  (cut across the barrier)', fontsize=9.5,
               color=INK2)
axc.set_ylabel('energy', fontsize=9.5, color=INK2)
axc.set_xticks([]); axc.set_yticks([])
for sp in ('top', 'right'):
    axc.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axc.spines[sp].set_color('#dedcd6')
axc.legend(frameon=False, fontsize=9.0, loc='upper center',
           bbox_to_anchor=(0.5, 1.02), labelspacing=0.35)
axc.set_title('c   the same thing as a cut', fontsize=11.5, color=INK,
              pad=8, loc='left')

# ---------------------------------------------------------------- text
fig.subplots_adjust(left=0.015, right=0.985, top=0.775, bottom=0.135,
                    wspace=0.10)
fig.text(0.015, 0.955,
         'Why one reaction can have two valid transition states',
         fontsize=16.5, color=INK, weight='bold', ha='left')
for k, line in enumerate([
        'SCHEMATIC — model functions, not computed data. At a fixed geometry '
        'the SCF equations can have more than one self-consistent solution: '
        'the restricted one pairs the two',
        'electrons in the same spatial orbital, the broken one puts them in '
        'different ones. The ground state is whichever lies lower, so it is '
        'the lower envelope of the two —',
        'and each sheet can carry its own pass. Both are then first-order '
        'saddles of the ground-state surface, sitting in regions where '
        'different sheets win.']):
    fig.text(0.015, 0.913 - 0.0295 * k, line, fontsize=9.6, color=INK2,
             ha='left')

fig.text(0.015, 0.052,
         'The broken sheet stops short of the ends on purpose: once the bond is '
         'short again the broken solution merges into the restricted one and '
         'no longer exists as a separate state — the',
         fontsize=9.3, color=INK3, ha='left')
fig.text(0.015, 0.024,
         'Coulson-Fischer point. That is why ⟨S²⟩ goes to zero at a relaxed '
         'reactant and why that is correct rather than a failed calculation. '
         'In our data all 45 reactants sit there.',
         fontsize=9.3, color=INK3, ha='left')

fig.savefig(f'{H}/two_sheets.png', dpi=200, facecolor=SURF)
print('written two_sheets.png')
print(f'RKS pass at z={P_R[2]:.3f}, BS pass at z={P_B[2]:.3f}')
