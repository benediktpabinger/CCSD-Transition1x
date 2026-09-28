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
fig.text(0.5, 0.45, "no new" + chr(10) + "path search", ha="center", va="top", fontsize=10.5, color=INK, linespacing=1.3)
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
fig.text(0.5, 0.46, "no new" + chr(10) + "path search", ha="center", va="top", fontsize=10.5, color=INK, linespacing=1.3)
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
fig.text(0.5, 0.44, "no new" + chr(10) + "path search", ha="center", va="top", fontsize=10.5, color=INK, linespacing=1.3)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.14, wspace=0.05)
fig.savefig(OUT + "pic_relabelling_3d.png", dpi=220, transparent=True)
plt.close(fig)
print("written", OUT + "pic_relabelling_3d.png")
