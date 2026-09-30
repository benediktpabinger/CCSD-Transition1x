"""Reactant, transition state and product of one Transition1x test reaction
(rxn8551, C2H3N3O2, a proton transfer from N to O), as flat ball-and-stick
drawings in the style of the other defence pictures.

Run from the repo root:  python docs/defence/draw_molecules.py
Output: docs/defence/pics/mol_reactant.png, mol_ts.png, mol_product.png, mol_strip.png
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = "docs/defence/pics/"
FORMULA, RXN = "C2H3N3O2", "rxn8551"
DARK, NAVY, MID = "#3a3a40", "#030f4f", "#7f7f86"
RC = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
RAD = {1: 0.22, 6: 0.34, 7: 0.34, 8: 0.34}
FILL = {1: "#f2f2f4", 6: "#8c8c92", 7: "#b9bfd6", 8: "#d9d9de"}


def bonds(Z, X, fac=1.25):
    return {(i, j) for i in range(len(Z)) for j in range(i + 1, len(Z))
            if np.linalg.norm(X[i] - X[j]) < fac * (RC[Z[i]] + RC[Z[j]])}


def kabsch(P, Q):
    """rotate P onto Q (both centred)"""
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    return P @ (Vt.T @ D @ U.T).T


with h5py.File("data/Transition1x.h5", "r") as f:
    rx = f["test"][FORMULA][RXN]
    Z = rx["atomic_numbers"][:]
    R = rx["reactant"]["positions"][0]
    T = rx["transition_state"]["positions"][0]
    P = rx["product"]["positions"][0]

Tc = T - T.mean(0)
Rc = kabsch(R - R.mean(0), Tc)
Pc = kabsch(P - P.mean(0), Tc)
# project everything on the plane of the transition state
_, _, Vt = np.linalg.svd(Tc)
proj = lambda X: X @ Vt[:2].T
bR, bP = bonds(Z, R), bonds(Z, P)
changing = (bR - bP) | (bP - bR)
moving = {k for b in changing for k in b if Z[k] == 1}


def draw(ax, X, solid, dashed, title):
    xy = proj(X)
    for i, j in solid:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="#6f6f75", lw=4, solid_capstyle="round", zorder=1)
    for i, j in dashed:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=NAVY, lw=2.6, ls=(0, (2, 1.6)), zorder=1)
    for k, z in enumerate(Z):
        hot = k in moving
        ax.add_patch(Circle(xy[k], RAD[z], fc="#e3e6f3" if hot else FILL[z], ec=NAVY if hot else DARK,
                            lw=2.0 if hot else 1.3, zorder=2))
        ax.text(xy[k, 0], xy[k, 1], SYM[z], ha="center", va="center", fontsize=11 if z != 1 else 9,
                color="white" if z == 6 else (NAVY if hot else DARK), fontweight="bold", zorder=3)
    ax.set_aspect("equal"); ax.axis("off")
    allxy = np.vstack([proj(Rc), proj(Tc), proj(Pc)])
    ax.set_xlim(allxy[:, 0].min() - 0.6, allxy[:, 0].max() + 0.6)
    ax.set_ylim(allxy[:, 1].min() - 1.1, allxy[:, 1].max() + 0.6)
    if title:
        ax.text(0.5, 0.02, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=13, fontweight="bold", color=MID)


panels = [("mol_reactant.png", Rc, bR, set(), "reactant"),
          ("mol_ts.png", Tc, bR & bP, changing, "transition state"),
          ("mol_product.png", Pc, bP, set(), "product")]
for name, X, solid, dashed, title in panels:
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    draw(ax, X, solid, dashed, title)
    fig.savefig(OUT + name, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("written", OUT + name)
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), gridspec_kw={"wspace": 0.05})
for ax, (name, X, solid, dashed, title) in zip(axes, panels):
    draw(ax, X, solid, dashed, title)
fig.savefig(OUT + "mol_strip.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "mol_strip.png")
print("changing bonds:", [(SYM[Z[i]] + str(i), SYM[Z[j]] + str(j)) for i, j in changing])
