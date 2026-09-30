"""Three small drawings for the introduction of the defence talk.

pic_dataset.png : a reaction path, ten geometries, a label (E, F) at each.
pic_rq1.png     : the same, a few labels swapped for better ones.
pic_rq2.png     : the same, all labels from another surface whose transition
                  state sits elsewhere; the geometries are not moved.

Run from the repo root:  python docs/defence/draw_dataset_pics.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "docs/defence/pics/"
INK, GREY, TAG, NEW, NAVY = "#222222", "#9a9a9a", "#d9d9de", "#e0731f", "#030f4f"


def profile(x):
    # reactant at 0, transition state at 0.5, product at 1 (lower)
    return 1.0 * np.exp(-((x - 0.5) ** 2) / 0.045) - 0.35 * x


def unrestricted(x):
    # lies below the restricted profile around the barrier, maximum shifted
    return profile(x) - 0.75 * np.exp(-((x - 0.60) ** 2) / 0.035)


def base(ax, show_curve=True):
    x = np.linspace(0, 1, 400)
    if show_curve:
        ax.plot(x, profile(x), color=INK, lw=2.2, zorder=2)
    xs = np.linspace(0.02, 0.98, 10)
    ax.scatter(xs, profile(xs), s=70, color="white", edgecolor=INK, lw=1.8, zorder=4)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.75, 1.55)
    ax.axis("off")
    return xs


def tag(ax, x, y, color, text="E, F", dy=0.25, fc=None):
    ax.add_patch(FancyBboxPatch((x - 0.045, y + dy - 0.06), 0.09, 0.12,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc or color, ec=color, lw=1.2, zorder=5))
    ax.text(x, y + dy, text, ha="center", va="center", fontsize=8.5,
            color="white" if fc is None and color != TAG else INK, zorder=6)
    ax.plot([x, x], [y + 0.04, y + dy - 0.07], color=color, lw=1, zorder=3)


def finish(fig, name):
    fig.savefig(OUT + name, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("written", OUT + name)


# 1. dataset = geometries + labels
fig, ax = plt.subplots(figsize=(6.4, 3.6))
xs = base(ax)
for x in xs:
    tag(ax, x, profile(x), TAG)
ax.text(0.5, -0.62, "geometries: where the path search put them",
        ha="center", fontsize=11, color=INK)
ax.text(0.5, 1.45, "labels: energy and force at each geometry",
        ha="center", fontsize=11, color=INK)
finish(fig, "pic_dataset.png")

# 2. RQ1: a few labels swapped for better ones
fig, ax = plt.subplots(figsize=(6.4, 3.6))
xs = base(ax)
better = {3, 4, 6}
for i, x in enumerate(xs):
    if i in better:
        tag(ax, x, profile(x), NEW)
    else:
        tag(ax, x, profile(x), TAG)
ax.text(0.5, 1.45, "a few labels recomputed at a better level",
        ha="center", fontsize=11, color=NEW)
ax.text(0.5, -0.62, "geometries unchanged", ha="center", fontsize=11, color=INK)
finish(fig, "pic_rq1.png")

# 3. RQ2: all labels from another surface, geometries not moved
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.linspace(0, 1, 400)
xs = base(ax, show_curve=False)
ax.plot(x, profile(x), color=GREY, lw=1.6, ls="--", zorder=1)
ax.plot(x, unrestricted(x), color=NEW, lw=2.2, zorder=2)
for xi in xs:
    tag(ax, xi, profile(xi), NEW)
xm = x[np.argmax(unrestricted(x))]
ax.scatter([xm], [unrestricted(xm)], marker="*", s=180, color=NEW, zorder=6)
ax.annotate("transition state of\nthe new surface", (xm, unrestricted(xm)),
            xytext=(0.18, 1.1), fontsize=10, color=NEW, ha="center",
            arrowprops=dict(arrowstyle="-", color=NEW, lw=1))
ax.text(0.5, 1.45, "all labels from another surface",
        ha="center", fontsize=11, color=NEW)
ax.text(0.5, -0.62, "geometries unchanged, placed by the old surface",
        ha="center", fontsize=11, color=INK)
finish(fig, "pic_rq2.png")


# 4. relabelling: same geometries, labels swapped, one single point each
from matplotlib.patches import FancyArrowPatch
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"wspace": 0.55})
for ax, color, head, sub in [
        (axes[0], TAG, "Transition1x", "geometries from the path search,\nlabels at " + chr(969) + "B97X/6-31G(d)"),
        (axes[1], NEW, "relabelled", "same geometries,\nlabels at " + chr(969) + "B97M-V/def2-TZVP")]:
    xs = base(ax)
    for x in xs:
        tag(ax, x, profile(x), color)
    ax.text(0.5, 1.45, head, ha="center", fontsize=13, color=INK, fontweight="bold")
    ax.text(0.5, -0.68, sub, ha="center", va="top", fontsize=10.5, color=INK, linespacing=1.4)
    ax.set_ylim(-1.15, 1.6)
# arrow between the panels, in figure coordinates
arr = FancyArrowPatch((0.435, 0.52), (0.565, 0.52), transform=fig.transFigure,
                      arrowstyle="-|>", mutation_scale=28, lw=3, color=NEW)
fig.patches.append(arr)
fig.text(0.5, 0.60, "one single point\nper geometry", ha="center", va="bottom",
         fontsize=10.5, color=NEW, linespacing=1.3)
fig.text(0.5, 0.45, "no new" + chr(10) + "path search", ha="center", va="top", fontsize=10.5, color="#9a9a9a", linespacing=1.3)
finish(fig, "pic_relabelling.png")


# 5. relabelling seen in configuration space: contour map of the surface,
#    the NEB path as points; right panel the other level, same points
def pes(x, y, level="cheap"):
    e = (-np.exp(-((x - 1.0) ** 2 + (y - 0.9) ** 2) / 0.55)
         - np.exp(-((x + 1.0) ** 2 + (y + 0.9) ** 2) / 0.55)
         + 0.05 * (x ** 2 + y ** 2))
    if level == "expensive":
        e = e + 0.10 * np.exp(-((x - 0.3) ** 2 + (y + 0.4) ** 2) / 0.4) - 0.04 * x
    return e


def path_points(n=10):
    t = np.linspace(-1.05, 1.05, n)
    return t, 0.85 * t + 0.25 * np.sin(np.pi * t)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"wspace": 0.45})
xx, yy = np.meshgrid(np.linspace(-2.1, 2.1, 300), np.linspace(-2.1, 2.1, 300))
for ax, level, color, head, sub in [
        (axes[0], "cheap", "#b8b8bd", "Transition1x", "path relaxed and labelled at " + chr(969) + "B97X/6-31G(d)"),
        (axes[1], "expensive", NEW, "relabelled", "same geometries, labels at " + chr(969) + "B97M-V/def2-TZVP")]:
    zz = pes(xx, yy, level)
    ax.contour(xx, yy, zz, levels=np.linspace(-0.95, 0.35, 14), colors=GREY if level == "cheap" else NEW,
               linewidths=0.8, alpha=0.55, linestyles="solid")
    px, py = path_points()
    ax.plot(px, py, color=INK, lw=1.4, zorder=3)
    ax.scatter(px, py, s=80, color=color, edgecolor=INK, lw=1.4, zorder=4)
    # one example label
    k = 3
    ax.annotate("E, F", (px[k], py[k]), xytext=(px[k] + 0.75, py[k] - 0.35), fontsize=9.5,
                color=INK if level == "cheap" else NEW, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.2),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2), zorder=6)
    ax.text(-1.0, -1.55, "reactant", fontsize=10, color=INK, ha="center")
    ax.text(1.0, 1.45, "product", fontsize=10, color=INK, ha="center")
    ax.text(px[5] - 0.28, py[5] + 0.22, chr(8225), fontsize=13, color=INK, ha="center", va="center")
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.1, 2.1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb")
    ax.set_xlabel("configuration space", fontsize=10, color=GREY)
    ax.set_title(head, fontsize=13, fontweight="bold", color=INK)
    ax.text(0.5, -0.14, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=INK)
arr = FancyArrowPatch((0.455, 0.52), (0.545, 0.52), transform=fig.transFigure,
                      arrowstyle="-|>", mutation_scale=28, lw=3, color=NEW)
fig.patches.append(arr)
fig.text(0.5, 0.58, "one single point" + chr(10) + "per geometry", ha="center", va="bottom",
         fontsize=10.5, color=NEW, linespacing=1.3)
fig.text(0.5, 0.46, "no new" + chr(10) + "path search", ha="center", va="top", fontsize=10.5, color="#9a9a9a", linespacing=1.3)
finish(fig, "pic_relabelling_config.png")


# 6. the same in 3D: surface height and the floor contours, path points
#    lifted to their label energy, stems show the geometry is unchanged
fig = plt.figure(figsize=(13, 5.2))
xx3, yy3 = np.meshgrid(np.linspace(-1.9, 1.9, 160), np.linspace(-1.9, 1.9, 160))
for k, (level, color, cmap_col, head, sub) in enumerate([
        ("cheap", "#8c8c92", "#bfbfc4", "Transition1x", "path relaxed and labelled at " + chr(969) + "B97X/6-31G(d)"),
        ("expensive", NEW, "#f0b88a", "relabelled", "same geometries, labels at " + chr(969) + "B97M-V/def2-TZVP")]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    zz = pes(xx3, yy3, level)
    floor = -1.55
    ax.plot_surface(xx3, yy3, zz, color=cmap_col, alpha=0.18, linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(xx3, yy3, zz, rstride=20, cstride=20, color=color, linewidth=0.5, alpha=0.45)
    ax.contour(xx3, yy3, zz, levels=np.linspace(-0.95, 0.35, 14), zdir="z", offset=floor,
               colors=color, linewidths=0.7, alpha=0.6)
    px, py = path_points()
    pz = pes(px, py, level)
    ax.plot(px, py, pz, color=INK, lw=1.4, zorder=10)
    ax.scatter(px, py, pz, s=55, color=color if level == "expensive" else "#b8b8bd", edgecolor=INK, lw=1.2, zorder=11, depthshade=False)
    ax.scatter(px, py, [floor] * len(px), s=22, color=INK, zorder=11, depthshade=False)
    for x, y, z in zip(px, py, pz):
        ax.plot([x, x], [y, y], [floor, z], color=INK, lw=0.6, alpha=0.45)
    ax.set_zlim(floor, 0.45)
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9)
    ax.view_init(elev=26, azim=-38)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_zlabel("energy", fontsize=10, color=GREY, labelpad=-8)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#dddddd"); ax.yaxis.pane.set_edgecolor("#dddddd"); ax.zaxis.pane.set_edgecolor("#dddddd")
    ax.grid(False)
    ax.set_title(head, fontsize=13, fontweight="bold", color=INK, pad=2)
    ax.text2D(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=INK)
    ax.text2D(0.5, -0.07, "floor: contour lines of the surface, black dots: the stored geometries" if k == 0
              else "same dots on the floor, new heights", transform=ax.transAxes, ha="center", va="top",
              fontsize=9.5, color=GREY)
arr = FancyArrowPatch((0.47, 0.5), (0.53, 0.5), transform=fig.transFigure,
                      arrowstyle="-|>", mutation_scale=26, lw=3, color=NEW)
fig.patches.append(arr)
fig.text(0.5, 0.56, "one single point" + chr(10) + "per geometry", ha="center", va="bottom", fontsize=10.5, color=NEW, linespacing=1.3)
fig.text(0.5, 0.44, "no new" + chr(10) + "path search", ha="center", va="top", fontsize=10.5, color="#9a9a9a", linespacing=1.3)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.14, wspace=0.05)
fig.savefig(OUT + "pic_relabelling_3d.png", dpi=220, transparent=True)
plt.close(fig)
print("written", OUT + "pic_relabelling_3d.png")


# 7. one panel, both surfaces: old (grey) and new (orange) slightly different,
#    same floor geometries, stems up to the label heights on both surfaces
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1, projection="3d")
floor = -1.55
z_old = pes(xx3, yy3, "cheap")
z_new = pes(xx3, yy3, "expensive") + 0.32
ax.plot_surface(xx3, yy3, z_old, color="#bfbfc4", alpha=0.10, linewidth=0, antialiased=True, shade=True)
ax.plot_wireframe(xx3, yy3, z_old, rstride=20, cstride=20, color="#8c8c92", linewidth=0.5, alpha=0.5)
ax.plot_surface(xx3, yy3, z_new, color="#f0b88a", alpha=0.10, linewidth=0, antialiased=True, shade=True)
ax.plot_wireframe(xx3, yy3, z_new, rstride=20, cstride=20, color=NEW, linewidth=0.5, alpha=0.5)
ax.contour(xx3, yy3, z_old, levels=np.linspace(-0.95, 0.35, 14), zdir="z", offset=floor,
           colors="#8c8c92", linewidths=0.7, alpha=0.6)
px, py = path_points()
pz_old = pes(px, py, "cheap")
pz_new = pes(px, py, "expensive") + 0.32
ax.plot(px, py, pz_old, color="#6f6f75", lw=1.3)
ax.plot(px, py, pz_new, color=NEW, lw=1.3)
for x, y, zo, zn in zip(px, py, pz_old, pz_new):
    ax.plot([x, x], [y, y], [floor, max(zo, zn)], color=INK, lw=0.7, alpha=0.5)
ax.scatter(px, py, [floor] * len(px), s=26, color=INK, depthshade=False)
ax.scatter(px, py, pz_old, s=55, color="#b8b8bd", edgecolor=INK, lw=1.1, depthshade=False)
ax.scatter(px, py, pz_new, s=55, color=NEW, edgecolor=INK, lw=1.1, depthshade=False)
ax.set_zlim(floor, 0.8); ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9)
ax.view_init(elev=26, azim=-38)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_zlabel("energy", fontsize=10, color=GREY, labelpad=-8)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.set_edgecolor("#dddddd")
ax.grid(False)
# legend as text
ax.text2D(0.02, 1.10, "grey: " + chr(969) + "B97X/6-31G(d), the surface the path was relaxed on",
          transform=ax.transAxes, fontsize=10.5, color="#6f6f75", va="top")
ax.text2D(0.02, 1.05, "orange: " + chr(969) + "B97M-V/def2-TZVP, the surface of the new labels",
          transform=ax.transAxes, fontsize=10.5, color=NEW, va="top")
ax.text2D(0.02, 1.00, "black dots: the stored geometries, unchanged",
          transform=ax.transAxes, fontsize=10.5, color=INK, va="top")
ax.text2D(0.5, 0.0, "relabelling: one single point per geometry on the new surface, no new path search",
          transform=ax.transAxes, fontsize=10.5, color=INK, ha="center", va="top")
fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.05)
fig.savefig(OUT + "pic_relabelling_3d_v2.png", dpi=220, transparent=True)
plt.close(fig)
print("written", OUT + "pic_relabelling_3d_v2.png")


# 8. two panels: left the cheap surface alone, arrow, right both surfaces
def panel3d(ax, both, bands=False):
    floor = -1.55
    z_old = pes(xx3, yy3, "cheap")
    ax.plot_surface(xx3, yy3, z_old, color="#bfbfc4", alpha=0.10, linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(xx3, yy3, z_old, rstride=20, cstride=20, color="#8c8c92", linewidth=0.5, alpha=0.5)
    ax.contour(xx3, yy3, z_old, levels=np.linspace(-0.95, 0.35, 14), zdir="z", offset=floor,
               colors="#8c8c92", linewidths=0.7, alpha=0.6)
    px, py = path_points()
    pz_old = pes(px, py, "cheap")
    if bands:
        # the intermediate NEB bands, from the straight interpolation to the final path
        t = np.linspace(-1.05, 1.05, 10)
        y_start = py[-1] / px[-1] * t - 0.55 * np.sin(np.pi * t)   # first band, bowed the other way
        for w, alpha in [(0.0, 0.35), (0.4, 0.45), (0.72, 0.55)]:
            bx, by = t, (1 - w) * y_start + w * py
            bz = pes(bx, by, "cheap")
            ax.plot(bx, by, bz, color="#8c8c92", lw=0.9, alpha=alpha)
            ax.scatter(bx, by, [floor] * len(bx), s=10, color=INK, alpha=alpha, depthshade=False)
            ax.scatter(bx, by, bz, s=22, color="#d0d0d4", edgecolor=INK, lw=0.6, alpha=alpha + 0.2, depthshade=False)
            if both:
                bzn = pes(bx, by, "expensive") + 0.32
                ax.scatter(bx, by, bzn, s=22, color=NEW, edgecolor=INK, lw=0.6, alpha=alpha + 0.2, depthshade=False)
            for x, y, z in zip(bx, by, bz):
                ax.plot([x, x], [y, y], [floor, z], color=INK, lw=0.3, alpha=0.18)
    ax.plot(px, py, pz_old, color="#6f6f75", lw=1.3)
    top = pz_old
    if both:
        z_new = pes(xx3, yy3, "expensive") + 0.32
        ax.plot_surface(xx3, yy3, z_new, color="#f0b88a", alpha=0.10, linewidth=0, antialiased=True, shade=True)
        ax.plot_wireframe(xx3, yy3, z_new, rstride=20, cstride=20, color=NEW, linewidth=0.5, alpha=0.5)
        pz_new = pes(px, py, "expensive") + 0.32
        ax.plot(px, py, pz_new, color=NEW, lw=1.3)
        top = np.maximum(pz_old, pz_new)
    for x, y, zt in zip(px, py, top):
        ax.plot([x, x], [y, y], [floor, zt], color=INK, lw=0.7, alpha=0.5)
    ax.scatter(px, py, [floor] * len(px), s=26, color=INK, depthshade=False)
    ax.scatter(px, py, pz_old, s=55, color="#b8b8bd", edgecolor=INK, lw=1.1, depthshade=False)
    if both:
        ax.scatter(px, py, pz_new, s=55, color=NEW, edgecolor=INK, lw=1.1, depthshade=False)
    ax.set_zlim(floor, 0.8); ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9)
    ax.view_init(elev=26, azim=-38)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_zlabel("energy", fontsize=10, color=GREY, labelpad=-8)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor("#dddddd")
    ax.grid(False)


fig = plt.figure(figsize=(14, 5.6))
axl = fig.add_subplot(1, 2, 1, projection="3d"); panel3d(axl, both=False)
axr = fig.add_subplot(1, 2, 2, projection="3d"); panel3d(axr, both=True)
axl.set_title("Transition1x", fontsize=13, fontweight="bold", color=INK, pad=14)
axr.set_title("relabelled", fontsize=13, fontweight="bold", color=INK, pad=14)
axl.text2D(0.5, 0.0, "path relaxed and labelled at " + chr(969) + "B97X/6-31G(d)", transform=axl.transAxes,
           ha="center", va="top", fontsize=10.5, color="#8c8c92")
axr.text2D(0.5, 0.0, "same geometries, new labels at " + chr(969) + "B97M-V/def2-TZVP", transform=axr.transAxes,
           ha="center", va="top", fontsize=10.5, color=NEW)
arr = FancyArrowPatch((0.47, 0.5), (0.53, 0.5), transform=fig.transFigure,
                      arrowstyle="-|>", mutation_scale=26, lw=3, color=NEW)
fig.patches.append(arr)
fig.text(0.5, 0.56, "one single point" + chr(10) + "per geometry", ha="center", va="bottom", fontsize=10.5, color=NEW, linespacing=1.3)
fig.text(0.5, 0.44, "no new" + chr(10) + "path search", ha="center", va="top", fontsize=10.5, color="#9a9a9a", linespacing=1.3)
fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.08, wspace=0.02)
fig.savefig(OUT + "pic_relabelling_3d_v3.png", dpi=220, transparent=True)
plt.close(fig)
print("written", OUT + "pic_relabelling_3d_v3.png")


# 9. as 8, but with the intermediate NEB bands that Transition1x also keeps
fig = plt.figure(figsize=(14, 5.6))
axl = fig.add_subplot(1, 2, 1, projection="3d"); panel3d(axl, both=False, bands=True)
axr = fig.add_subplot(1, 2, 2, projection="3d"); panel3d(axr, both=True, bands=True)
axl.set_title("Transition1x", fontsize=13, fontweight="bold", color=INK, pad=14)
axr.set_title("relabelled", fontsize=13, fontweight="bold", color=INK, pad=14)
axl.text2D(0.5, 0.0, "every NEB iteration kept: paths relaxed and labelled at " + chr(969) + "B97X/6-31G(d)",
           transform=axl.transAxes, ha="center", va="top", fontsize=10.5, color="#8c8c92")
axr.text2D(0.5, 0.0, "same geometries, new labels at " + chr(969) + "B97M-V/def2-TZVP", transform=axr.transAxes,
           ha="center", va="top", fontsize=10.5, color=NEW)
arr = FancyArrowPatch((0.47, 0.5), (0.53, 0.5), transform=fig.transFigure,
                      arrowstyle="-|>", mutation_scale=26, lw=3, color=NEW)
fig.patches.append(arr)
fig.text(0.5, 0.56, "one single point" + chr(10) + "per geometry", ha="center", va="bottom", fontsize=10.5, color=NEW, linespacing=1.3)
fig.text(0.5, 0.44, "no new" + chr(10) + "path search", ha="center", va="top", fontsize=10.5, color="#9a9a9a", linespacing=1.3)
fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.08, wspace=0.02)
fig.savefig(OUT + "pic_relabelling_3d_v4.png", dpi=220, transparent=True)
plt.close(fig)
print("written", OUT + "pic_relabelling_3d_v4.png")


# 10. what Transition1x changed: left, equilibrium datasets sample around the
#     minima only; right, Transition1x samples the reaction paths
def panel_equilibrium(ax):
    floor = -1.55
    z_old = pes(xx3, yy3, "cheap")
    ax.plot_surface(xx3, yy3, z_old, color="#bfbfc4", alpha=0.10, linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(xx3, yy3, z_old, rstride=20, cstride=20, color="#8c8c92", linewidth=0.5, alpha=0.5)
    ax.contour(xx3, yy3, z_old, levels=np.linspace(-0.95, 0.35, 14), zdir="z", offset=floor,
               colors="#8c8c92", linewidths=0.7, alpha=0.6)
    rng = np.random.default_rng(3)
    for cx, cy in [(-1.0, -0.9), (1.0, 0.9)]:
        n = 22
        r = np.clip(np.abs(rng.normal(0, 0.22, n)), 0, 0.42); th = rng.uniform(0, 2 * np.pi, n)
        qx, qy = cx + r * np.cos(th), cy + r * np.sin(th)
        qz = pes(qx, qy, "cheap")
        for x, y, z in zip(qx, qy, qz):
            ax.plot([x, x], [y, y], [floor, z], color=INK, lw=0.4, alpha=0.3)
        ax.scatter(qx, qy, [floor] * n, s=14, color=INK, alpha=0.7, depthshade=False)
        ax.scatter(qx, qy, qz, s=30, color="#b8b8bd", edgecolor=INK, lw=0.8, depthshade=False)
    ax.set_zlim(floor, 0.8); ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9)
    ax.view_init(elev=26, azim=-38)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_zlabel("energy", fontsize=10, color=GREY, labelpad=-8)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor("#dddddd")
    ax.grid(False)


fig = plt.figure(figsize=(14, 5.6))
axl = fig.add_subplot(1, 2, 1, projection="3d"); panel_equilibrium(axl)
axr = fig.add_subplot(1, 2, 2, projection="3d"); panel3d(axr, both=False, bands=True)
axl.set_title("equilibrium datasets (QM9, ANI-1x)", fontsize=13, fontweight="bold", color="#7f7f86", pad=14)
axr.set_title("Transition1x", fontsize=13, fontweight="bold", color="#7f7f86", pad=14)
axl.text2D(0.5, 0.0, "structures at and near the minima; the transition-state region is empty",
           transform=axl.transAxes, ha="center", va="top", fontsize=10.5, color="#7f7f86")
axr.text2D(0.5, 0.0, "reaction paths, every NEB iteration kept; the transition-state region is sampled",
           transform=axr.transAxes, ha="center", va="top", fontsize=10.5, color="#7f7f86")
arr = FancyArrowPatch((0.47, 0.5), (0.53, 0.5), transform=fig.transFigure,
                      arrowstyle="-|>", mutation_scale=26, lw=3, color="#9a9a9a")
fig.patches.append(arr)
fig.text(0.5, 0.56, "sampling the" + chr(10) + "reaction paths", ha="center", va="bottom", fontsize=10.5, color="#7f7f86", linespacing=1.3)
fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.08, wspace=0.02)
fig.savefig(OUT + "pic_t1x_vs_equilibrium.png", dpi=220, transparent=True)
plt.close(fig)
print("written", OUT + "pic_t1x_vs_equilibrium.png")


# 11. the two questions as two edits of the relabelling picture
def pes_bs(x, y):
    """the surface of the other spin formalism: a lowered channel beside the
    restricted path, so the lowest saddle sits beside the path, not on it"""
    u = (x + y) / np.sqrt(2.0)          # along the path
    v = (x - y) / np.sqrt(2.0)          # across the path
    # trough across the path direction, with a small bump at u = 0 inside it,
    # so the trough's highest point (the new saddle) sits at (u, v) = (0, 0.75)
    across = np.exp(-((v - 1.1) ** 2) / 0.35)
    along = 0.45 + 0.55 * (1.0 - np.exp(-(u ** 2) / 0.35))
    ridge = 0.65 * np.exp(-(v ** 2) / 0.16) * np.exp(-(u ** 2) / 0.7)   # the restricted saddle region is raised
    return pes(x, y, "cheap") + 1.9 - 1.0 * across * along + ridge


def saddle_of(f):
    u, v = 0.0, 1.1
    x = (u + v) / np.sqrt(2.0); y = (u - v) / np.sqrt(2.0)
    return x, y, f(x, y)


def panel_rq(ax, mode):
    floor = -1.55
    z_old = pes(xx3, yy3, "cheap")
    ax.plot_surface(xx3, yy3, z_old, color="#bfbfc4", alpha=0.10, linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(xx3, yy3, z_old, rstride=20, cstride=20, color="#8c8c92", linewidth=0.5, alpha=0.5)
    ax.contour(xx3, yy3, z_old, levels=np.linspace(-0.95, 0.35, 14), zdir="z", offset=floor,
               colors="#8c8c92", linewidths=0.7, alpha=0.6)
    px, py = path_points()
    pz_old = pes(px, py, "cheap")
    if mode == "rq1":
        z_new = pes(xx3, yy3, "expensive") + 0.32
        pz_new = pes(px, py, "expensive") + 0.32
        sel = [4, 6]
        # the many stored geometries of the earlier NEB bands, cheap labels only
        t = np.linspace(-1.05, 1.05, 10)
        y_start = py[-1] / px[-1] * t - 0.55 * np.sin(np.pi * t)
        for w, alpha in [(0.0, 0.35), (0.4, 0.45), (0.72, 0.55)]:
            bx, by = t, (1 - w) * y_start + w * py
            bz = pes(bx, by, "cheap")
            ax.plot(bx, by, bz, color="#8c8c92", lw=0.9, alpha=alpha)
            ax.scatter(bx, by, [floor] * len(bx), s=10, color=INK, alpha=alpha, depthshade=False)
            ax.scatter(bx, by, bz, s=22, color="#d0d0d4", edgecolor=INK, lw=0.6, alpha=alpha + 0.2, depthshade=False)
            for x, y, z in zip(bx, by, bz):
                ax.plot([x, x], [y, y], [floor, z], color=INK, lw=0.3, alpha=0.18)
    else:
        z_new = pes_bs(xx3, yy3)
        pz_new = pes_bs(px, py)
        sel = list(range(len(px)))
    ax.plot_surface(xx3, yy3, z_new, color="#f0b88a", alpha=0.10, linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(xx3, yy3, z_new, rstride=20, cstride=20, color=NEW, linewidth=0.5, alpha=0.5)
    ax.plot(px, py, pz_old, color="#6f6f75", lw=1.3)
    for i, (x, y) in enumerate(zip(px, py)):
        top = max(pz_old[i], pz_new[i]) if i in sel else pz_old[i]
        ax.plot([x, x], [y, y], [floor, top], color="#6e6e74", lw=0.7, alpha=0.8)
    ax.scatter(px, py, [floor] * len(px), s=22, color="#6e6e74", depthshade=False)
    ax.scatter(px, py, pz_old, s=55, color="#b8b8bd", edgecolor=INK, lw=1.1, depthshade=False)
    ax.scatter(px[sel], py[sel], pz_new[sel], s=55, color=NEW, edgecolor=INK, lw=1.1, depthshade=False)
    if mode == "rq2":
        # the minimum energy path of the unrestricted surface, through its own saddle
        uu = np.linspace(-1.34, 1.34, 200); vv = 1.1 * np.exp(-(uu ** 2) / 0.45)
        mx, my = (uu + vv) / np.sqrt(2.0), (uu - vv) / np.sqrt(2.0)
        ax.plot(mx, my, pes_bs(mx, my), color=NEW, lw=1.6, ls="--")
        ax.plot(mx, my, [floor] * len(mx), color=NEW, lw=1.2, ls="--", alpha=0.8)
        sx, sy, sz = saddle_of(pes_bs)
        ax.scatter([sx], [sy], [sz], s=170, marker="*", color=NEW, edgecolor=INK, lw=0.8, depthshade=False)
        ax.scatter([sx], [sy], [floor], s=70, marker="*", color=NEW, edgecolor=INK, lw=0.6, depthshade=False)
        ax.plot([sx, sx], [sy, sy], [floor, sz], color=NEW, lw=0.9, ls="--", alpha=0.8)
    ax.set_zlim(floor, 2.5 if mode == "rq2" else 0.8); ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9)
    if mode == "rq2":
        ax.set_box_aspect((1, 1, 1.35))
    ax.view_init(elev=20 if mode == "rq2" else 26, azim=-38)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_zlabel("energy", fontsize=10, color=GREY, labelpad=-8)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor("#dddddd")
    ax.grid(False)


MID = "#7f7f86"
for mode, name, head, sub in [
        ("rq1", "pic_rq1_3d.png", "relabel a few geometries, not all",
         "two single points at the better level; the many others keep their old labels"),
        ("rq2", "pic_rq2_3d_v2.png", "relabel on a surface with a different shape",
         "labels from the unrestricted surface; its transition state (star) lies beside the path")]:
    fig = plt.figure(figsize=(7, 5.6))
    ax = fig.add_subplot(1, 1, 1, projection="3d"); panel_rq(ax, mode)
    ax.set_title(head, fontsize=13, fontweight="bold", color=MID, pad=14)
    ax.text2D(0.5, 0.0, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=MID)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.08)
    fig.savefig(OUT + name, dpi=220, transparent=True)
    plt.close(fig)
    print("written", OUT + name)


# 12. restricted vs unrestricted: the H2 dissociation curve
r = np.linspace(0.45, 3.4, 500)
r0, D, a = 0.74, 1.0, 1.9
e_u = D * (1 - np.exp(-a * (r - r0))) ** 2 - D          # unrestricted: goes to the atoms
r_cf = 1.25                                               # where the restricted solution becomes unstable
e_r = e_u + np.where(r > r_cf, 0.62 * (1 - np.exp(-1.1 * (r - r_cf))) ** 2, 0.0)   # restricted: same up to r_cf, then above, levelling off higher
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(r, e_r, color="#8c8c92", lw=2.4, label="restricted (RKS)")
ax.plot(r, e_u, color=NEW, lw=2.4, label="unrestricted (UKS)")
ax.plot(r[r <= r_cf], e_u[r <= r_cf], color="#8c8c92", lw=2.4)
ax.axvline(r_cf, color="#bbbbbb", lw=1, ls=":")
ax.text(r_cf - 0.05, 0.05, "all electrons paired:\nsame energy", ha="right", va="bottom", fontsize=10, color="#6f6f75")
ax.text(r_cf + 0.08, 0.05, "bond half broken:\nunrestricted lies lower,\nbroken-symmetry solution", ha="left", va="bottom", fontsize=10, color=NEW)
ax.annotate("", xy=(2.6, e_u[np.searchsorted(r, 2.6)]), xytext=(2.6, e_r[np.searchsorted(r, 2.6)]),
            arrowprops=dict(arrowstyle="<->", color=MID, lw=1))
ax.text(2.66, 0.5 * (e_u[np.searchsorted(r, 2.6)] + e_r[np.searchsorted(r, 2.6)]), "breaking depth", fontsize=9.5, color=MID, va="center")
ax.set_xlabel("bond length", fontsize=11, color=MID)
ax.set_ylabel("energy", fontsize=11, color=MID)
ax.set_xticks([]); ax.set_yticks([])
ax.set_ylim(-1.1, 0.55)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#bbbbbb")
ax.legend(loc="upper right", frameon=False, fontsize=10.5, labelcolor=[ "#6f6f75", NEW])
ax.set_title("two spin formalisms, two surfaces", fontsize=13, fontweight="bold", color=MID)
fig.savefig(OUT + "pic_rks_uks.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_rks_uks.png")


# 13. energy along a reaction path, restricted and unrestricted surface, schematic
x = np.linspace(0, 1, 500)
e_rks = np.exp(-((x - 0.5) ** 2) / 0.04) - 0.3 * x
depth = 0.42 * np.exp(-((x - 0.5) ** 2) / 0.018)        # unstable region around the barrier
e_uks = e_rks - depth
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.fill_between(x, e_uks, e_rks, color=NEW, alpha=0.12, lw=0)
ax.plot(x, e_rks, color="#8c8c92", lw=2.4, label="restricted surface")
ax.plot(x, e_uks, color=NEW, lw=2.4, label="unrestricted surface")
i_ts = np.argmax(e_rks)
ax.scatter([x[i_ts]], [e_rks[i_ts]], s=90, color="white", edgecolor="#6f6f75", lw=1.8, zorder=5)
ax.annotate("", xy=(x[i_ts], e_uks[i_ts] + 0.01), xytext=(x[i_ts], e_rks[i_ts] - 0.01),
            arrowprops=dict(arrowstyle="<->", color=MID, lw=1))
ax.text(x[i_ts] + 0.02, 0.5 * (e_uks[i_ts] + e_rks[i_ts]), "breaking depth", fontsize=9.5, color=MID, va="center")
ax.annotate("restricted transition state:\na saddle on the restricted surface,\na slope on the unrestricted one",
            (x[i_ts], e_rks[i_ts]), xytext=(0.66, 0.95), fontsize=9.5, color="#6f6f75", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.9))
for xa, xb in [(0.02, 0.24), (0.76, 0.98)]:
    y = -0.42
    ax.plot([xa, xb], [y, y], color="#bbbbbb", lw=1)
    ax.text(0.5 * (xa + xb), y - 0.05, "stable: both surfaces coincide", ha="center", va="top", fontsize=9, color=MID)
ax.text(0.5, -0.47, "unstable: unrestricted lies below", ha="center", va="top", fontsize=9, color=NEW)
ax.text(0.5, 0.06, "the unrestricted transition state\nlies off this path", ha="center", va="center", fontsize=9.5, color=NEW, style="italic")
ax.set_xlabel("reaction coordinate, along the restricted path", fontsize=11, color=MID)
ax.set_ylabel("energy", fontsize=11, color=MID)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, 1); ax.set_ylim(-0.62, 1.12)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("#bbbbbb")
ax.legend(loc="upper left", frameon=False, fontsize=10.5, labelcolor=["#6f6f75", NEW])
ax.set_title("energy along a reaction path, both surfaces", fontsize=13, fontweight="bold", color=MID)
fig.savefig(OUT + "pic_path_two_surfaces.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_path_two_surfaces.png")


# 14. level of theory: functional ladder and basis-set size
fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.35})
rungs = ["LDA", "GGA", "meta-GGA", "hybrid", "range-separated\nhybrid", "range-separated\nhybrid meta-GGA"]
for k, name in enumerate(rungs):
    y = k
    col = "#8c8c92" if k < 4 else ("#8c8c92" if k == 4 else NEW)
    axl.plot([0.25, 0.75], [y, y], color="#bbbbbb", lw=2.2)
    axl.text(0.78, y, name, fontsize=9.5, color=MID, va="center", ha="left")
axl.plot([0.25, 0.25], [-0.3, len(rungs) - 0.7], color="#bbbbbb", lw=2.2)
axl.plot([0.75, 0.75], [-0.3, len(rungs) - 0.7], color="#bbbbbb", lw=2.2)
axl.scatter([0.5], [4], s=90, color="#b8b8bd", edgecolor=INK, lw=1.2, zorder=5)
axl.text(0.5, 3.62, chr(969) + "B97X", ha="center", va="top", fontsize=10, color="#6f6f75", fontweight="bold")
axl.scatter([0.5], [5], s=90, color=NEW, edgecolor=INK, lw=1.2, zorder=5)
axl.text(0.5, 5.32, chr(969) + "B97M-V", ha="center", fontsize=10, color=NEW, fontweight="bold")
axl.annotate("", xy=(0.5, 4.85), xytext=(0.5, 4.15), arrowprops=dict(arrowstyle="-|>", color=NEW, lw=1.6))
axl.set_xlim(0, 2.2); axl.set_ylim(-0.6, 5.9); axl.axis("off")
axl.set_title("functional", fontsize=12, fontweight="bold", color=MID)
axl.text(0.5, -0.55, "Jacob's ladder", ha="center", fontsize=9.5, color=MID, style="italic")
# basis set: functions per carbon atom as bars
names = ["6-31G(d)", "def2-TZVP"]; nfn = [15, 31]; cols = ["#b8b8bd", NEW]
axr.bar([0, 1], nfn, width=0.55, color=cols, edgecolor=INK, lw=1.0)
for i, (n, nm) in enumerate(zip(nfn, names)):
    axr.text(i, n + 1.2, str(n), ha="center", fontsize=10.5, color=MID, fontweight="bold")
    axr.text(i, -2.5, nm, ha="center", va="top", fontsize=10, color="#6f6f75" if i == 0 else NEW, fontweight="bold")
axr.text(0.5, -8.5, "double zeta  " + chr(8594) + "  triple zeta,\npolarisation on all atoms", ha="center", va="top", fontsize=9.5, color=MID)
axr.annotate("", xy=(0.72, 22), xytext=(0.28, 22), arrowprops=dict(arrowstyle="-|>", color=NEW, lw=1.6))
axr.set_xlim(-0.6, 1.6); axr.set_ylim(-14, 38); axr.axis("off")
axr.set_title("basis set", fontsize=12, fontweight="bold", color=MID)
axr.text(0.5, 35.5, "basis functions per carbon atom", ha="center", fontsize=9.5, color=MID, style="italic")
fig.savefig(OUT + "pic_level_of_theory.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_level_of_theory.png")


# 15. ways to raise the fidelity: three rows, Transition1x -> upgrade
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
rows = [
    (3.55, "Functional", "the approximation for exchange\nand correlation, Jacob's ladder",
     chr(969) + "B97X", "range-separated hybrid", chr(969) + "B97M-V", "one rung higher, with dispersion"),
    (2.35, "Basis set", "the functions the orbitals\nare built from",
     "6-31G(d)", "double zeta, 15 functions per C", "def2-TZVP", "triple zeta, 31 functions per C"),
    (1.15, "Spin formalism", "whether spin-up and spin-down\nelectrons share the same orbitals",
     "restricted", "shared orbitals, the default", "unrestricted", "own orbitals, lower where a bond\nis half broken: broken symmetry"),
]
def cell(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
# column headers
ax.text(4.55, 4.35, "Transition1x", ha="center", fontsize=12.5, color="#6f6f75", fontweight="bold")
ax.text(9.1, 4.35, "upgrade", ha="center", fontsize=12.5, color=NEW, fontweight="bold")
for y, name, desc, a, adesc, b, bdesc in rows:
    ax.text(0.25, y + 0.16, name, fontsize=12.5, color=MID, fontweight="bold", va="center")
    ax.text(0.25, y - 0.24, desc, fontsize=9, color=MID, va="center", linespacing=1.25)
    cell(3.2, y, 2.7, 0.92, "#ececef", "#b8b8bd")
    ax.text(4.55, y + 0.16, a, ha="center", va="center", fontsize=12, color="#6f6f75", fontweight="bold")
    ax.text(4.55, y - 0.24, adesc, ha="center", va="center", fontsize=8.8, color="#6f6f75")
    cell(7.75, y, 2.7, 0.92, "#fbe7d6", NEW)
    ax.text(9.1, y + 0.16, b, ha="center", va="center", fontsize=12, color=NEW, fontweight="bold")
    ax.text(9.1, y - 0.24, bdesc, ha="center", va="center", fontsize=8.8, color=NEW, linespacing=1.2)
    ax.add_patch(FancyArrowPatch((6.05, y), (7.6, y), arrowstyle="-|>", mutation_scale=18, lw=2, color="#9a9a9a"))
# brace: level of theory
ax.plot([10.75, 10.9, 10.9, 10.75], [4.05, 4.05, 1.85, 1.85], color="#9a9a9a", lw=1.4)
ax.plot([10.9, 11.05], [2.95, 2.95], color="#9a9a9a", lw=1.4)
ax.text(11.15, 2.95, "level of theory", fontsize=11, color=MID, va="center")
ax.plot([10.75, 10.9, 10.9, 10.75], [1.6, 1.6, 0.7, 0.7], color=NEW, lw=1.4)
ax.plot([10.9, 11.05], [1.15, 1.15], color=NEW, lw=1.4)
ax.text(11.15, 1.15, "a different surface\nwhere a bond breaks", fontsize=10, color=NEW, va="center", linespacing=1.25)
ax.text(6.85, 0.25, "each step: higher accuracy, higher cost", ha="center", fontsize=10.5, color=MID, style="italic")
fig.savefig(OUT + "pic_fidelity_table.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_fidelity_table.png")


# 16. compact fidelity tables for the two questions
def compact_table(name, rows, note):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.set_xlim(0, 6.4); ax.set_ylim(-0.25, 2.4); ax.axis("off")
    ax.text(2.35, 2.2, "Transition1x", ha="center", fontsize=10.5, color="#6f6f75", fontweight="bold")
    ax.text(5.05, 2.2, "relabelled", ha="center", fontsize=10.5, color=NEW, fontweight="bold")
    for y, label, a, b, changed in rows:
        col_b = NEW if changed else "#8c8c92"
        fc_b = "#fbe7d6" if changed else "#ececef"
        ec_b = NEW if changed else "#b8b8bd"
        ax.text(0.05, y, label, fontsize=10.5, color=MID, va="center", fontweight="bold")
        ax.add_patch(FancyBboxPatch((1.45, y - 0.22), 1.8, 0.44, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#ececef", ec="#b8b8bd", lw=1))
        ax.text(2.35, y, a, ha="center", va="center", fontsize=10, color="#6f6f75", fontweight="bold")
        ax.add_patch(FancyBboxPatch((4.15, y - 0.22), 1.8, 0.44, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc_b, ec=ec_b, lw=1))
        ax.text(5.05, y, b, ha="center", va="center", fontsize=10, color=col_b, fontweight="bold")
        ax.add_patch(FancyArrowPatch((3.35, y), (4.05, y), arrowstyle="-|>", mutation_scale=14, lw=1.6,
                                     color=NEW if changed else "#c8c8cc"))
        if not changed:
            ax.text(5.05, y - 0.36, "unchanged", ha="center", va="center", fontsize=8, color="#9a9a9a", style="italic")
    ax.text(3.2, -0.08, note, ha="center", va="center", fontsize=9.5, color=MID, style="italic")
    fig.savefig(OUT + name, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("written", OUT + name)


compact_table("pic_table_rq1.png", [
    (1.75, "Functional", chr(969) + "B97X", chr(969) + "B97M-V", True),
    (1.2, "Basis set", "6-31G(d)", "def2-TZVP", True),
    (0.6, "Spin formalism", "restricted", "restricted", False),
], "only the level of theory changes, on a subset of the data")
compact_table("pic_table_rq2.png", [
    (1.75, "Functional", chr(969) + "B97X", chr(969) + "B97M-V", True),
    (1.2, "Basis set", "6-31G(d)", "def2-TZVPD", True),
    (0.6, "Spin formalism", "restricted", "unrestricted", True),
], "level and spin formalism change, on all of the data (OMol25)")


# 17. the mountain-pass metaphor: finding the pass vs. a better altimeter
def terrain(x, y):
    return (np.exp(-((x) ** 2 + (y - 1.1) ** 2) / 0.6) + np.exp(-((x) ** 2 + (y + 1.1) ** 2) / 0.6)
            + 0.35 * np.exp(-(x ** 2) / 0.5))


gx, gy = np.meshgrid(np.linspace(-2.2, 2.2, 300), np.linspace(-2.2, 2.2, 300))
gz = terrain(gx, gy)
tx = np.linspace(-2.0, 2.0, 10)
ty = 0.18 * np.sin(np.pi * tx / 2.0)          # the trail over the pass
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.3})
for ax in axes:
    ax.contour(gx, gy, gz, levels=np.linspace(0.15, 1.3, 12), colors="#9a9a9a", linewidths=0.8, alpha=0.7)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb")
    ax.text(-1.9, -1.95, "valley A", fontsize=10, color=MID)
    ax.text(1.25, 1.85, "valley B", fontsize=10, color=MID)
    ax.text(0, 1.1, "peak", fontsize=9, color=MID, ha="center", va="center")
    ax.text(0, -1.1, "peak", fontsize=9, color=MID, ha="center", va="center")
# left: the search, many tentative steps converging on the pass
rng = np.random.default_rng(7)
ax = axes[0]
for k, w in enumerate([1.0, 0.6, 0.3, 0.0]):
    # earlier bands bow over the shoulder of the peak and converge on the pass
    wig = 0.85 * w * np.sin(np.pi * (tx + 2) / 4)
    px_ = tx; py_ = ty + wig
    ax.plot(px_, py_, color="#8c8c92", lw=1.0, ls="--" if w > 0 else "-", alpha=0.35 + 0.16 * k)
    ax.scatter(px_, py_, s=12 if w > 0 else 40, color="#6f6f75" if w > 0 else "white", edgecolor="#6f6f75",
               lw=0.8, alpha=0.5 + 0.12 * k, zorder=4)
    if w > 0:
        for x, y in zip(px_[1:-1], py_[1:-1]):
            # a small slope measurement at each tentative step
            ax.annotate("", xy=(x, y - 0.16), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color="#8c8c92", lw=0.6, mutation_scale=7))
ax.scatter([0], [0], s=160, marker="*", color="white", edgecolor="#6f6f75", lw=1.2, zorder=6)
ax.set_title("finding the pass", fontsize=13, fontweight="bold", color=MID)
ax.text(0.5, -0.06, "walk, measure the slope, correct, walk again:\nhundreds of measurements for one pass", transform=ax.transAxes,
        ha="center", va="top", fontsize=10.5, color=MID, linespacing=1.3)
# right: the trail is marked, one reading per marker
ax = axes[1]
ax.plot(tx, ty, color="#6f6f75", lw=1.4)
ax.scatter(tx, ty, s=40, color="white", edgecolor="#6f6f75", lw=1.2, zorder=4)
ax.scatter([0], [0], s=160, marker="*", color="white", edgecolor="#6f6f75", lw=1.2, zorder=6)
for x, y in zip(tx, ty):
    ax.add_patch(FancyBboxPatch((x - 0.16, y + 0.22), 0.32, 0.2, boxstyle="round,pad=0.01,rounding_size=0.04", fc=NEW, ec=NEW, lw=1, zorder=5))
    ax.text(x, y + 0.32, "h", ha="center", va="center", fontsize=8, color="white", zorder=6, style="italic")
    ax.plot([x, x], [y + 0.05, y + 0.22], color=NEW, lw=0.8)
ax.set_title("a better altimeter", fontsize=13, fontweight="bold", color=NEW)
ax.text(0.5, -0.06, "the trail is marked: stand on each marker once\nand read the height again. one reading per marker", transform=ax.transAxes,
        ha="center", va="top", fontsize=10.5, color=NEW, linespacing=1.3)
arr = FancyArrowPatch((0.475, 0.52), (0.525, 0.52), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=26, lw=3, color=NEW)
fig.patches.append(arr)
fig.savefig(OUT + "pic_mountain_pass.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_mountain_pass.png")


# 18. the mountain pass as an illustration, seen from the valley: the trail
#     zigzags up the face to the notch between the two peaks
from matplotlib.patches import Polygon
def mountains(ax):
    xs = np.linspace(0, 10, 600)
    back = 4.2 + 0.9 * np.sin(xs * 1.1 + 0.4) + 0.5 * np.sin(xs * 2.7) + 0.25 * np.sin(xs * 5.3 + 1)
    ax.fill_between(xs, 0, back, color="#dcdce0", lw=0)
    front = (1.6 + 3.0 * np.exp(-((xs - 3.0) ** 2) / 1.6) + 3.3 * np.exp(-((xs - 7.2) ** 2) / 1.8)
             + 0.3 * np.sin(xs * 4.1) * np.exp(-((xs - 5.1) ** 2) / 8))
    ax.fill_between(xs, 0, front, color="#b3b3ba", lw=0)
    ax.plot(xs, front, color="#8c8c92", lw=1.2)
    for cx in (3.0, 7.2):
        m = np.abs(xs - cx) < 0.6
        ax.fill_between(xs[m], front[m] - 0.32, front[m], color="#f2f2f4", lw=0)
    ip = np.argmin(np.where(np.abs(xs - 5.1) < 1.4, front, 99))
    return xs, front, xs[ip], front[ip]


def zigzag(x0, y0, x1, y1, n=10, legs=4, width=1.4):
    # clean switchbacks: 'legs' straight legs alternating left/right, then resampled to n markers
    knots_x = [x0]; knots_y = [y0]
    for L in range(1, legs + 1):
        f = L / legs
        kx = x0 + (x1 - x0) * f + (width * (1 - f) if L % 2 else -width * (1 - f) * 0.9)
        ky = y0 + (y1 - y0) * f
        knots_x.append(kx); knots_y.append(ky)
    knots_x[-1], knots_y[-1] = x1, y1
    seg = np.cumsum([0] + [np.hypot(knots_x[k + 1] - knots_x[k], knots_y[k + 1] - knots_y[k]) for k in range(legs)])
    t = np.linspace(0, seg[-1], n)
    return np.interp(t, seg, knots_x), np.interp(t, seg, knots_y)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.12})
for k, ax in enumerate(axes):
    xs, front, px, py = mountains(ax)
    ax.set_xlim(0, 10); ax.set_ylim(-1.6, 6.2); ax.axis("off")
    ax.scatter([px], [py + 0.1], s=220, marker="*", color="white", edgecolor="#6f6f75", lw=1.2, zorder=6)
    ax.text(px, py + 0.5, "the pass", ha="center", fontsize=10, color=MID)
    tx, ty = zigzag(1.4, 0.9, px, py - 0.05, legs=3, width=2.4)
    if k == 0:
        # dead ends: tentative routes that climb towards the peaks and stop
        for (xa, ya, xb, yb) in [(1.6, 0.9, 2.9, 4.1), (1.6, 0.9, 6.6, 3.9)]:
            dx, dy = zigzag(xa, ya, xb, yb, n=7, legs=3, width=0.7)
            ax.plot(dx, dy, color="#6f6f75", lw=0.9, ls="--", alpha=0.45, zorder=4)
            ax.scatter(dx, dy, s=12, color="#6f6f75", alpha=0.5, zorder=5)
            ax.text(xb, yb + 0.12, chr(10005), ha="center", va="bottom", fontsize=9, color="#6f6f75", alpha=0.8)
        ax.plot(tx, ty, color="#6f6f75", lw=1.2, zorder=4)
        ax.scatter(tx, ty, s=30, color="white", edgecolor="#6f6f75", lw=0.9, zorder=5)
        ax.set_title("finding the pass", fontsize=13, fontweight="bold", color=MID)
        ax.text(5, -0.95, "walk, measure the slope, correct, walk again:" + chr(10) + "hundreds of measurements for one pass", ha="center", va="bottom", fontsize=10.5, color=MID, linespacing=1.3)
    else:
        ax.plot(tx, ty, color="#6f6f75", lw=1.2, zorder=4)
        for x, y in zip(tx, ty):
            ax.plot([x, x], [y, y + 0.42], color="#6f6f75", lw=1.2, zorder=5)
            ax.add_patch(FancyBboxPatch((x - 0.17, y + 0.42), 0.34, 0.24, boxstyle="round,pad=0.01,rounding_size=0.04", fc=NEW, ec=NEW, zorder=6))
            ax.text(x, y + 0.54, "h", ha="center", va="center", fontsize=8, color="white", style="italic", zorder=7)
        ax.set_title("a better altimeter", fontsize=13, fontweight="bold", color=NEW)
        ax.text(5, -0.95, "the trail is marked: stand on each marker once" + chr(10) + "and read the height again. one reading per marker", ha="center", va="bottom", fontsize=10.5, color=NEW, linespacing=1.3)
fig.savefig(OUT + "pic_mountain_pass_illustration.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_mountain_pass_illustration.png")


# 19. v2: the map changed, the pass is somewhere else and lower; the markers stayed
def mountains_v2(ax):
    xs = np.linspace(0, 10, 600)
    back = 4.2 + 0.9 * np.sin(xs * 1.1 + 0.4) + 0.5 * np.sin(xs * 2.7) + 0.25 * np.sin(xs * 5.3 + 1)
    ax.fill_between(xs, 0, back, color="#dcdce0", lw=0)
    front = (1.6 + 3.0 * np.exp(-((xs - 3.0) ** 2) / 1.6) + 3.3 * np.exp(-((xs - 8.2) ** 2) / 1.6)
             + 0.3 * np.sin(xs * 4.1) * np.exp(-((xs - 5.1) ** 2) / 8))
    # the old notch is filled in, a new, lower notch opens well to the right
    front = front + 1.0 * np.exp(-((xs - 5.1) ** 2) / 0.6) - 1.6 * np.exp(-((xs - 7.05) ** 2) / 0.22)
    ax.fill_between(xs, 0, front, color="#b3b3ba", lw=0)
    ax.plot(xs, front, color=NEW, lw=1.4)
    for cx in (3.0, 8.2):
        m = np.abs(xs - cx) < 0.6
        ax.fill_between(xs[m], front[m] - 0.32, front[m], color="#f2f2f4", lw=0)
    i_old = np.argmin(np.where(np.abs(xs - 5.1) < 1.4, front, 99))
    i_new = np.argmin(np.where(np.abs(xs - 7.05) < 0.6, front, 99))
    return xs, front, xs[i_old], front[i_old], xs[i_new], front[i_new]


fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.12})
# left: the marked trail on the old map
ax = axes[0]
xs, front, px, py = mountains(ax)
ax.set_xlim(0, 10); ax.set_ylim(-1.6, 6.2); ax.axis("off")
ax.scatter([px], [py + 0.1], s=220, marker="*", color="white", edgecolor="#6f6f75", lw=1.2, zorder=6)
ax.text(px, py + 0.5, "the pass", ha="center", fontsize=10, color=MID)
tx, ty = zigzag(1.4, 0.9, px, py - 0.05, legs=3, width=2.4)
ax.plot(tx, ty, color="#6f6f75", lw=1.2, zorder=4)
for x, y in zip(tx, ty):
    ax.plot([x, x], [y, y + 0.42], color="#6f6f75", lw=1.2, zorder=5)
    ax.add_patch(FancyBboxPatch((x - 0.17, y + 0.42), 0.34, 0.24, boxstyle="round,pad=0.01,rounding_size=0.04", fc="#8c8c92", ec="#8c8c92", zorder=6))
    ax.text(x, y + 0.54, "h", ha="center", va="center", fontsize=8, color="white", style="italic", zorder=7)
ax.set_title("the marked trail", fontsize=13, fontweight="bold", color=MID)
ax.text(5, -0.95, "the markers were placed on this map:" + chr(10) + "the trail leads to the pass", ha="center", va="bottom", fontsize=10.5, color=MID, linespacing=1.3)
# right: a different map
ax = axes[1]
xs2, front2, ox, oy, nx, ny = mountains_v2(ax)
ax.set_xlim(0, 10); ax.set_ylim(-1.6, 6.2); ax.axis("off")
# the old trail and its markers, unchanged in position; the old notch is now a ridge
tx2, ty2 = zigzag(1.4, 0.9, px, py - 0.05, legs=3, width=2.4)
ty2 = np.minimum(ty2, np.interp(tx2, xs2, front2) + 0.05)
ax.plot(tx2, ty2, color="#6f6f75", lw=1.2, zorder=4)
for x, y in zip(tx2, ty2):
    ax.plot([x, x], [y, y + 0.42], color="#6f6f75", lw=1.2, zorder=5)
    ax.add_patch(FancyBboxPatch((x - 0.17, y + 0.42), 0.34, 0.24, boxstyle="round,pad=0.01,rounding_size=0.04", fc=NEW, ec=NEW, zorder=6))
    ax.text(x, y + 0.54, "h", ha="center", va="center", fontsize=8, color="white", style="italic", zorder=7)
ax.scatter([px], [np.interp(px, xs2, front2) + 0.1], s=160, marker="x", color="#6f6f75", lw=1.6, zorder=6)
ax.text(px - 0.9, np.interp(px, xs2, front2) + 0.6, "not a pass" + chr(10) + "any more", ha="center", va="bottom", fontsize=9.5, color=MID, linespacing=1.2)
ax.scatter([nx], [ny + 0.1], s=260, marker="*", color=NEW, edgecolor="#6f6f75", lw=1.0, zorder=6)
ax.text(nx + 0.45, ny + 0.75, "the pass:" + chr(10) + "lower, elsewhere," + chr(10) + "no marker there", ha="left", va="bottom", fontsize=9.5, color=NEW, linespacing=1.2)
ax.set_title("a different map", fontsize=13, fontweight="bold", color=NEW)
ax.text(5, -0.95, "the new readings are right at every marker," + chr(10) + "but the pass moved and no marker leads there", ha="center", va="bottom", fontsize=10.5, color=NEW, linespacing=1.3)
fig.savefig(OUT + "pic_mountain_pass_illustration_v3.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_mountain_pass_illustration_v3.png")


# 20. the middle case: only the level of theory changes, the map is almost the
#     same and the pass moves by a step
def mountains_lot(ax):
    xs = np.linspace(0, 10, 600)
    back = 4.2 + 0.9 * np.sin(xs * 1.1 + 0.4) + 0.5 * np.sin(xs * 2.7) + 0.25 * np.sin(xs * 5.3 + 1)
    ax.fill_between(xs, 0, back, color="#dcdce0", lw=0)
    front = (1.6 + 3.0 * np.exp(-((xs - 3.0) ** 2) / 1.6) + 3.3 * np.exp(-((xs - 7.2) ** 2) / 1.8)
             + 0.3 * np.sin(xs * 4.1) * np.exp(-((xs - 5.1) ** 2) / 8))
    # slightly different: a little higher overall, a little different in shape, notch a step to the right
    front = front + 0.25 + 0.12 * np.sin(xs * 2.3 + 1) - 0.18 * np.exp(-((xs - 5.45) ** 2) / 0.15)
    ax.fill_between(xs, 0, front, color="#b3b3ba", lw=0)
    ax.plot(xs, front, color=NEW, lw=1.4)
    for cx in (3.0, 7.2):
        m = np.abs(xs - cx) < 0.6
        ax.fill_between(xs[m], front[m] - 0.32, front[m], color="#f2f2f4", lw=0)
    i_new = np.argmin(np.where(np.abs(xs - 5.3) < 1.0, front, 99))
    return xs, front, xs[i_new], front[i_new]


fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.12})
ax = axes[0]
xs, front, px, py = mountains(ax)
ax.set_xlim(0, 10); ax.set_ylim(-1.6, 6.2); ax.axis("off")
ax.scatter([px], [py + 0.1], s=220, marker="*", color="white", edgecolor="#6f6f75", lw=1.2, zorder=6)
ax.text(px, py + 0.5, "the pass", ha="center", fontsize=10, color=MID)
tx, ty = zigzag(1.4, 0.9, px, py - 0.05, legs=3, width=2.4)
ax.plot(tx, ty, color="#6f6f75", lw=1.2, zorder=4)
for x, y in zip(tx, ty):
    ax.plot([x, x], [y, y + 0.42], color="#6f6f75", lw=1.2, zorder=5)
    ax.add_patch(FancyBboxPatch((x - 0.17, y + 0.42), 0.34, 0.24, boxstyle="round,pad=0.01,rounding_size=0.04", fc="#8c8c92", ec="#8c8c92", zorder=6))
    ax.text(x, y + 0.54, "h", ha="center", va="center", fontsize=8, color="white", style="italic", zorder=7)
ax.set_title("the marked trail", fontsize=13, fontweight="bold", color=MID)
ax.text(5, -0.95, "the markers were placed on this map:" + chr(10) + "the trail leads to the pass", ha="center", va="bottom", fontsize=10.5, color=MID, linespacing=1.3)
ax = axes[1]
xs2, front2, nx, ny = mountains_lot(ax)
ax.set_xlim(0, 10); ax.set_ylim(-1.6, 6.2); ax.axis("off")
tx2, ty2 = zigzag(1.4, 0.9, px, py - 0.05, legs=3, width=2.4)
ty2 = ty2 + 0.22   # the same markers, the map sits a little higher
ax.plot(tx2, ty2, color="#6f6f75", lw=1.2, zorder=4)
for x, y in zip(tx2, ty2):
    ax.plot([x, x], [y, y + 0.42], color="#6f6f75", lw=1.2, zorder=5)
    ax.add_patch(FancyBboxPatch((x - 0.17, y + 0.42), 0.34, 0.24, boxstyle="round,pad=0.01,rounding_size=0.04", fc=NEW, ec=NEW, zorder=6))
    ax.text(x, y + 0.54, "h", ha="center", va="center", fontsize=8, color="white", style="italic", zorder=7)
ax.scatter([nx], [ny + 0.1], s=240, marker="*", color=NEW, edgecolor="#6f6f75", lw=1.0, zorder=6)
ax.scatter([px], [np.interp(px, xs2, front2) + 0.1], s=90, marker="*", color="white", edgecolor="#9a9a9a", lw=1.0, zorder=5)
ax.text(nx + 0.05, ny + 1.15, "the pass: a step away," + chr(10) + "the last marker still" + chr(10) + "stands on it", ha="center", va="bottom", fontsize=9.5, color=NEW, linespacing=1.2)
ax.set_title("a slightly different map", fontsize=13, fontweight="bold", color=NEW)
ax.text(5, -0.95, "the heights change, the mountain hardly does:" + chr(10) + "new readings at the old markers are enough", ha="center", va="bottom", fontsize=10.5, color=NEW, linespacing=1.3)
fig.savefig(OUT + "pic_mountain_pass_illustration_v4.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_mountain_pass_illustration_v4.png")


# 21. v5: all three next to each other
def draw_markers(ax, tx, ty, color):
    ax.plot(tx, ty, color="#6f6f75", lw=1.2, zorder=4)
    for x, y in zip(tx, ty):
        ax.plot([x, x], [y, y + 0.42], color="#6f6f75", lw=1.2, zorder=5)
        ax.add_patch(FancyBboxPatch((x - 0.17, y + 0.42), 0.34, 0.24, boxstyle="round,pad=0.01,rounding_size=0.04", fc=color, ec=color, zorder=6))
        ax.text(x, y + 0.54, "h", ha="center", va="center", fontsize=8, color="white", style="italic", zorder=7)


def mountains_lot5(ax):
    xs = np.linspace(0, 10, 600)
    back = 4.2 + 0.9 * np.sin(xs * 1.1 + 0.4) + 0.5 * np.sin(xs * 2.7) + 0.25 * np.sin(xs * 5.3 + 1)
    ax.fill_between(xs, 0, back, color="#dcdce0", lw=0)
    front = (1.6 + 3.0 * np.exp(-((xs - 3.0) ** 2) / 1.6) + 3.3 * np.exp(-((xs - 7.2) ** 2) / 1.8)
             + 0.3 * np.sin(xs * 4.1) * np.exp(-((xs - 5.1) ** 2) / 8))
    front = front + 0.25 + 0.12 * np.sin(xs * 2.3 + 1) - 0.55 * np.exp(-((xs - 5.75) ** 2) / 0.14)
    ax.fill_between(xs, 0, front, color="#b3b3ba", lw=0)
    ax.plot(xs, front, color=NEW, lw=1.4)
    for cx in (3.0, 7.2):
        m = np.abs(xs - cx) < 0.6
        ax.fill_between(xs[m], front[m] - 0.32, front[m], color="#f2f2f4", lw=0)
    i_new = np.argmin(np.where(np.abs(xs - 5.6) < 0.8, front, 99))
    return xs, front, xs[i_new], front[i_new]


fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw={"wspace": 0.08})
for ax in axes:
    ax.set_xlim(0, 10); ax.set_ylim(-1.8, 6.4); ax.axis("off")
# 1: the marked trail
ax = axes[0]
xs, front, px, py = mountains(ax)
ax.scatter([px], [py + 0.1], s=220, marker="*", color="white", edgecolor="#6f6f75", lw=1.2, zorder=6)
ax.text(px, py + 0.5, "the pass", ha="center", fontsize=10, color=MID)
tx, ty = zigzag(1.4, 0.9, px, py - 0.05, legs=3, width=2.4)
draw_markers(ax, tx, ty, "#8c8c92")
ax.set_title("the marked trail", fontsize=13, fontweight="bold", color=MID)
ax.text(5, -1.1, "the markers were placed on this map:" + chr(10) + "the trail leads to the pass", ha="center", va="bottom", fontsize=10.5, color=MID, linespacing=1.3)
# 2: slightly different map
ax = axes[1]
xs2, front2, nx, ny = mountains_lot5(ax)
draw_markers(ax, tx, ty + 0.22, NEW)
ax.scatter([nx], [ny + 0.1], s=240, marker="*", color=NEW, edgecolor="#6f6f75", lw=1.0, zorder=6)
ax.text(nx - 0.9, ny + 1.35, "the pass: a step away," + chr(10) + "the last marker is" + chr(10) + "next to it", ha="center", va="bottom", fontsize=9.5, color=NEW, linespacing=1.2)
ax.set_title("a slightly different map", fontsize=13, fontweight="bold", color=NEW)
ax.text(5, -1.1, "the heights change, the mountain hardly does:" + chr(10) + "new readings at the old markers are enough", ha="center", va="bottom", fontsize=10.5, color=NEW, linespacing=1.3)
# 3: a different map
ax = axes[2]
xs3, front3, ox, oy, nx3, ny3 = mountains_v2(ax)
ty3 = np.minimum(ty, np.interp(tx, xs3, front3) + 0.05)
draw_markers(ax, tx, ty3, NEW)
ax.scatter([px], [np.interp(px, xs3, front3) + 0.1], s=160, marker="x", color="#6f6f75", lw=1.6, zorder=6)
ax.text(px - 0.9, np.interp(px, xs3, front3) + 0.6, "not a pass" + chr(10) + "any more", ha="center", va="bottom", fontsize=9.5, color=MID, linespacing=1.2)
ax.scatter([nx3], [ny3 + 0.1], s=260, marker="*", color=NEW, edgecolor="#6f6f75", lw=1.0, zorder=6)
ax.text(nx3 + 0.45, ny3 + 0.75, "the pass:" + chr(10) + "lower, elsewhere," + chr(10) + "no marker there", ha="left", va="bottom", fontsize=9.5, color=NEW, linespacing=1.2)
ax.set_title("a different map", fontsize=13, fontweight="bold", color=NEW)
ax.text(5, -1.1, "the pass moved and no marker leads there:" + chr(10) + "new readings at the old markers are not enough", ha="center", va="bottom", fontsize=10.5, color=NEW, linespacing=1.3)
fig.savefig(OUT + "pic_mountain_pass_illustration_v5.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_mountain_pass_illustration_v5.png")


# 22. RQ2 workflow: search with the OMol25 models, check with DFT, sort, compare
fig, ax = plt.subplots(figsize=(12.5, 6.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
def wbox(x, y, w, h, title, lines, fc="#ececef", ec="#8c8c92", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=11, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + (h - 0.5) / 2 - 0.02, lines, ha="center", va="center", fontsize=8.6, color=tc, linespacing=1.35)
def warrow(x0, y0, x1, y1, color="#6f6f75"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=color))
# row 1: reactions -> models search
wbox(0.3, 4.4, 3.0, 1.5, "45 reactions", "Transition1x test split\nranked by N$_{FOD}$: low, mid, high MR\nno unrestricted reference exists")
warrow(3.3, 5.15, 3.9, 5.15)
wbox(3.9, 4.4, 3.2, 1.5, "3 OMol25 models search", "UMA-S, UMA-M, eSEN, as released\nrelax endpoints, CI-NEB\nsame settings as the DFT reference")
warrow(7.1, 5.15, 7.7, 5.15)
wbox(7.7, 4.4, 3.0, 1.5, "135 transition states", "one per model and reaction\nthe workflow reports success")
# row 2: DFT check
warrow(9.2, 4.4, 9.2, 3.75)
wbox(3.9, 2.25, 6.8, 1.5, "one DFT single point at each, OMol25 protocol", "ωB97M-V/def2-TZVPD, unrestricted, plus a stability analysis\nno optimisation: the structure is judged where the model left it", fc="#fbe7d6", ec=NEW, tc=INK)
# row 3: three outputs
for k, (title, lines) in enumerate([
        ("residual force", "largest force component\non the unrestricted surface\nis it a stationary point?"),
        ("energy", "the barrier the model found,\nand the model's own energy\nand force error at the point"),
        ("<S²>", "0: closed-shell\n> 0: broken-symmetry\n82 / 53 structures")]):
    x = 3.9 + k * 2.3
    warrow(x + 1.05, 2.25, x + 1.05, 1.75)
    wbox(x, 0.35, 2.1, 1.4, title, lines, fc="#f5f5f7", ec="#b8b8bd")
# left: the comparison
ax.add_patch(FancyBboxPatch((0.3, 0.35), 3.0, 3.4, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=NEW, lw=1.6, ls="--"))
ax.text(1.8, 3.4, "the test", ha="center", va="center", fontsize=11, fontweight="bold", color=NEW)
ax.text(1.8, 2.05, "same metrics on both groups\n\nclosed-shell: the surfaces coincide,\nthe models are on home ground\n\nbroken-symmetry: the unrestricted\nsurface has its own transition state,\nno training geometry sat on it\n\ndo the models do as well there?",
        ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.35)
warrow(3.9, 1.05, 3.35, 1.05, color=NEW)
fig.savefig(OUT + "pic_rq2_workflow.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_rq2_workflow.png")


# 23. fidelity table v2: larger, darker text for projection, no footer line
DARK = "#3a3a40"
fig, ax = plt.subplots(figsize=(12, 4.3))
ax.set_xlim(0, 12); ax.set_ylim(0.45, 4.6); ax.axis("off")
rows = [
    (3.55, "Functional", "approximation for exchange" + chr(10) + "and correlation",
     chr(969) + "B97X", "range-separated hybrid", chr(969) + "B97M-V", "a rung higher, with dispersion"),
    (2.35, "Basis set", "the functions the orbitals" + chr(10) + "are built from",
     "6-31G(d)", "double zeta, 15 functions per C", "def2-TZVP", "triple zeta, 31 functions per C"),
    (1.15, "Spin formalism", "do spin-up and spin-down" + chr(10) + "electrons share orbitals?",
     "restricted", "shared orbitals, the default", "unrestricted", "own orbitals: lower energy" + chr(10) + "where a bond is half broken"),
]
ax.text(4.55, 4.35, "Transition1x", ha="center", fontsize=14, color=DARK, fontweight="bold")
ax.text(9.0, 4.35, "higher fidelity", ha="center", fontsize=14, color=NEW, fontweight="bold")
for y, name, desc, a, adesc, b_, bdesc in rows:
    ax.text(0.2, y + 0.2, name, fontsize=14, color=DARK, fontweight="bold", va="center")
    ax.text(0.2, y - 0.24, desc, fontsize=10.5, color=DARK, va="center", linespacing=1.25)
    ax.add_patch(FancyBboxPatch((3.0, y - 0.48), 3.1, 0.96, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#ececef", ec="#8c8c92", lw=1.4))
    ax.text(4.55, y + 0.18, a, ha="center", va="center", fontsize=13.5, color=DARK, fontweight="bold")
    ax.text(4.55, y - 0.24, adesc, ha="center", va="center", fontsize=10, color=DARK)
    ax.add_patch(FancyBboxPatch((7.45, y - 0.48), 3.1, 0.96, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fbe7d6", ec=NEW, lw=1.4))
    ax.text(9.0, y + 0.18, b_, ha="center", va="center", fontsize=13.5, color="#b8560f", fontweight="bold")
    ax.text(9.0, y - 0.24, bdesc, ha="center", va="center", fontsize=10, color="#b8560f", linespacing=1.2)
    ax.add_patch(FancyArrowPatch((6.2, y), (7.35, y), arrowstyle="-|>", mutation_scale=20, lw=2.2, color="#6f6f75"))
ax.plot([10.7, 10.85, 10.85, 10.7], [4.05, 4.05, 1.85, 1.85], color="#6f6f75", lw=1.6)
ax.plot([10.85, 11.0], [2.95, 2.95], color="#6f6f75", lw=1.6)
ax.text(11.1, 2.95, "level of" + chr(10) + "theory", fontsize=12.5, color=DARK, va="center", fontweight="bold", linespacing=1.2)
ax.plot([10.7, 10.85, 10.85, 10.7], [1.63, 1.63, 0.67, 0.67], color=NEW, lw=1.6)
ax.plot([10.85, 11.0], [1.15, 1.15], color=NEW, lw=1.6)
ax.text(11.1, 1.15, "a different" + chr(10) + "surface", fontsize=12.5, color="#b8560f", va="center", fontweight="bold", linespacing=1.2)
fig.savefig(OUT + "pic_fidelity_table_v2.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_fidelity_table_v2.png")


# 24. what a calculation does: geometry in, energy and forces out
from matplotlib.patches import Circle
DARK = "#3a3a40"
atoms = [("C", 0.0, 0.0), ("C", 1.25, 0.45), ("O", 2.45, -0.15),
         ("H", -0.75, 0.75), ("H", -0.65, -0.85), ("H", 0.35, -0.95),
         ("H", 1.2, 1.55), ("H", 1.75, -0.45 + 1.45), ("H", 3.15, 0.45)]
atoms[7] = ("H", 2.0, 1.15)
bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8)]
forces = [(0.25, -0.3), (-0.2, 0.35), (0.4, 0.1), (-0.35, -0.05), (-0.15, -0.3), (0.2, -0.25), (0.05, 0.35), (0.3, 0.2), (0.3, 0.25)]
RADIUS = {"C": 0.30, "O": 0.30, "H": 0.19}
FILL = {"C": "#8c8c92", "O": "#c9c9ce", "H": "#f2f2f4"}


def molecule(ax, ox, oy, with_forces=False):
    for i, j in bonds:
        ax.plot([ox + atoms[i][1], ox + atoms[j][1]], [oy + atoms[i][2], oy + atoms[j][2]], color="#6f6f75", lw=3, zorder=1, solid_capstyle="round")
    for k, (el, x, y) in enumerate(atoms):
        ax.add_patch(Circle((ox + x, oy + y), RADIUS[el], fc=FILL[el], ec=DARK, lw=1.3, zorder=2))
        ax.text(ox + x, oy + y, el, ha="center", va="center", fontsize=10 if el != "H" else 8, color="white" if el == "C" else DARK, fontweight="bold", zorder=3)
        if with_forces:
            fx, fy = forces[k]
            ax.add_patch(FancyArrowPatch((ox + x, oy + y), (ox + x + 1.7 * fx, oy + y + 1.7 * fy), arrowstyle="-|>", mutation_scale=14, lw=2.2, color="#030f4f", zorder=4))


fig, ax = plt.subplots(figsize=(12.5, 4.2))
ax.set_xlim(-1.6, 14.4); ax.set_ylim(-2.4, 2.9); ax.set_aspect("equal"); ax.axis("off")
# in: the geometry
molecule(ax, 0.0, 0.2)
ax.text(1.2, 2.55, "geometry", ha="center", fontsize=14, fontweight="bold", color=DARK)
ax.text(1.2, -1.75, "the positions of all atoms", ha="center", fontsize=11, color=DARK)
# the calculation
ax.add_patch(FancyArrowPatch((3.9, 0.35), (5.0, 0.35), arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#6f6f75"))
ax.add_patch(FancyBboxPatch((5.1, -0.75), 3.0, 2.2, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#ececef", ec="#8c8c92", lw=1.5))
ax.text(6.6, 0.85, "DFT", ha="center", va="center", fontsize=17, fontweight="bold", color=DARK)
ax.text(6.6, -0.05, "functional" + chr(10) + "basis set" + chr(10) + "spin formalism", ha="center", va="center", fontsize=10.5, color=DARK, linespacing=1.35)
ax.text(6.6, 2.55, "calculation", ha="center", fontsize=14, fontweight="bold", color=DARK)
ax.text(6.6, -1.75, "one calculation per geometry", ha="center", fontsize=11, color=DARK)
ax.add_patch(FancyArrowPatch((8.2, 0.35), (9.3, 0.35), arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#6f6f75"))
# out: energy and forces
molecule(ax, 10.6, 0.2, with_forces=True)
ax.text(11.8, 2.55, "energy and forces", ha="center", fontsize=14, fontweight="bold", color="#030f4f")
ax.text(11.8, -1.75, "one energy E for the molecule," + chr(10) + "one force F on every atom", ha="center", va="center", fontsize=11, color="#030f4f", linespacing=1.3)
ax.add_patch(FancyBboxPatch((9.2, 1.5), 0.85, 0.65, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#e3e6f3", ec="#030f4f", lw=1.4))
ax.text(9.625, 1.825, "E", ha="center", va="center", fontsize=15, fontweight="bold", color="#030f4f", style="italic")
fig.savefig(OUT + "pic_geometry_to_labels.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_geometry_to_labels.png")


# 25. the same, stacked top to bottom
NAVY_C = "#030f4f"
_forces_backup = None
fig, ax = plt.subplots(figsize=(7.6, 6.9))
ax.set_xlim(-1.9, 10.4); ax.set_ylim(-3.5, 7.6); ax.set_aspect("equal"); ax.axis("off")
# top: geometry
molecule(ax, 0.0, 5.6)
ax.text(4.5, 6.15, "geometry", ha="left", va="center", fontsize=14, fontweight="bold", color=DARK)
ax.text(4.5, 5.55, "the positions of all atoms", ha="left", va="center", fontsize=11, color=DARK)
ax.add_patch(FancyArrowPatch((1.2, 4.35), (1.2, 3.55), arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#6f6f75"))
# middle: the calculation
ax.add_patch(FancyBboxPatch((-0.3, 1.35), 3.0, 2.1, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#ececef", ec="#8c8c92", lw=1.5))
ax.text(1.2, 2.95, "DFT", ha="center", va="center", fontsize=17, fontweight="bold", color=DARK)
ax.text(1.2, 2.05, "functional" + chr(10) + "basis set" + chr(10) + "spin formalism", ha="center", va="center", fontsize=10.5, color=DARK, linespacing=1.35)
ax.text(4.5, 2.7, "calculation", ha="left", va="center", fontsize=14, fontweight="bold", color=DARK)
ax.text(4.5, 2.1, "one calculation per geometry", ha="left", va="center", fontsize=11, color=DARK)
ax.add_patch(FancyArrowPatch((1.2, 1.25), (1.2, 0.45), arrowstyle="-|>", mutation_scale=22, lw=2.4, color="#6f6f75"))
# bottom: energy and forces
for i_, j_ in bonds:
    ax.plot([atoms[i_][1], atoms[j_][1]], [-2.0 + atoms[i_][2], -2.0 + atoms[j_][2]], color="#6f6f75", lw=3, zorder=1, solid_capstyle="round")
for k_, (el, x, y) in enumerate(atoms):
    ax.add_patch(Circle((x, -2.0 + y), RADIUS[el], fc=FILL[el], ec=DARK, lw=1.3, zorder=2))
    ax.text(x, -2.0 + y, el, ha="center", va="center", fontsize=10 if el != "H" else 8, color="white" if el == "C" else DARK, fontweight="bold", zorder=3)
    fx, fy = forces[k_]
    ax.add_patch(FancyArrowPatch((x, -2.0 + y), (x + 1.7 * fx, -2.0 + y + 1.7 * fy), arrowstyle="-|>", mutation_scale=14, lw=2.2, color=NAVY_C, zorder=4))
ax.add_patch(FancyBboxPatch((-1.6, -0.75), 0.85, 0.65, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#e3e6f3", ec=NAVY_C, lw=1.4))
ax.text(-1.175, -0.425, "E", ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY_C, style="italic")
ax.text(4.5, -1.45, "energy and forces", ha="left", va="center", fontsize=14, fontweight="bold", color=NAVY_C)
ax.text(4.5, -2.2, "one energy E for the molecule," + chr(10) + "one force F on every atom", ha="left", va="center", fontsize=11, color=NAVY_C, linespacing=1.3)
fig.savefig(OUT + "pic_geometry_to_labels_vertical.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("written", OUT + "pic_geometry_to_labels_vertical.png")
