#!/usr/bin/env python3
"""Show why another random tow does not fit: lid already touched, voids remain."""

from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "outputs"
NAVY = "#002D4C"
GREEN = "#009682"
RED = "#C0392B"
GOLD = "#C4A35A"
GREY = "#4D4D4D"


def render_slice():
    mesh = pv.read(OUT / "cuboid_01_90x33x9.vtk")
    slab = mesh.clip_box([-20.0, 20.0, -1.2, 1.2, -0.2, 9.4], invert=False)
    if not isinstance(slab, pv.PolyData):
        slab = slab.extract_geometry()
    if "diameter" in slab.cell_data:
        slab = slab.cell_data_to_point_data()
    if "diameter" in slab.array_names:
        slab["radius"] = np.asarray(slab["diameter"], dtype=float) * 0.5
        drawn = slab.tube(scalars="radius", absolute=True, n_sides=6, capping=True)
    else:
        drawn = slab.tube(radius=0.05, n_sides=6, capping=True)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=[1800, 700])
    plotter.set_background("white")
    plotter.add_mesh(
        drawn,
        scalars="angle" if "angle" in drawn.array_names else None,
        cmap="viridis",
        clim=[0.0, 180.0],
        show_scalar_bar=False,
        smooth_shading=True,
    )
    lid = pv.Line((-20.0, 0.0, 9.0), (20.0, 0.0, 9.0))
    plotter.add_mesh(lid, color=RED, line_width=8)
    floor = pv.Line((-20.0, 0.0, 0.0), (20.0, 0.0, 0.0))
    plotter.add_mesh(floor, color=NAVY, line_width=4)
    plotter.camera_position = [
        (0.0, 38.0, 4.5),
        (0.0, 0.0, 4.5),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.35)
    path = OUT / "fvc_why_not_slice.png"
    plotter.show(screenshot=str(path), auto_close=True)
    plotter.close()
    from PIL import Image
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    ink = (arr < 250).any(axis=2)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) and len(cols):
        pad = 12
        r0, r1 = max(0, rows[0] - pad), min(arr.shape[0], rows[-1] + pad)
        c0, c1 = max(0, cols[0] - pad), min(arr.shape[1], cols[-1] + pad)
        im.crop((c0, r0, c1, r1)).save(path)
    print("wrote", path)
    return path


def render_reject():
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.plot([0.6, 9.4], [9.0, 9.0], color=RED, lw=3.0, zorder=5)
    ax.text(9.55, 9.0, "Deckel 9 mm", color=RED, fontsize=11, va="center", fontweight="bold")
    ax.plot([0.6, 9.4], [0.3, 0.3], color=NAVY, lw=2.0)

    # existing tows stacked to the lid
    ax.add_patch(Rectangle((1.0, 0.4), 5.2, 1.15, angle=6, facecolor="#3B6B9A",
                           edgecolor=NAVY, lw=0.7, zorder=2))
    ax.add_patch(Rectangle((3.4, 1.7), 5.0, 1.15, angle=-8, facecolor="#2A9D8F",
                           edgecolor=NAVY, lw=0.7, zorder=2))
    ax.add_patch(Rectangle((1.2, 3.2), 5.4, 1.15, angle=10, facecolor="#E9C46A",
                           edgecolor=NAVY, lw=0.7, zorder=2))
    ax.add_patch(Rectangle((3.0, 4.9), 5.2, 1.15, angle=-6, facecolor="#264653",
                           edgecolor=NAVY, lw=0.7, zorder=2))
    ax.add_patch(Rectangle((1.1, 6.5), 5.5, 1.15, angle=7, facecolor="#3B6B9A",
                           edgecolor=NAVY, lw=0.7, zorder=2))
    # peak already at lid
    ax.add_patch(Rectangle((2.4, 7.85), 5.0, 1.15, angle=-4, facecolor="#2A9D8F",
                           edgecolor=NAVY, lw=0.8, zorder=3))

    # rejected tow sticking through the lid
    ax.add_patch(Rectangle((0.9, 8.55), 7.6, 1.15, angle=3, facecolor="#E74C3C",
                           edgecolor=RED, lw=1.2, linestyle="--", alpha=0.55, zorder=4))
    ax.text(5.0, 10.55, "nächster Roving passt nicht", ha="center",
            color=RED, fontsize=12, fontweight="bold")
    ax.annotate(
        "",
        xy=(5.4, 9.7), xytext=(5.4, 10.35),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.6),
    )
    ax.text(5.0, -0.15, "Der Stapel berührt den Deckel.\nUnten bleiben Lücken – aber kein Roving passt mehr hindurch.",
            ha="center", va="top", color=GREY, fontsize=10)

    path = OUT / "fvc_why_not_reject.png"
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


def montage():
    from matplotlib.image import imread
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), facecolor="white")
    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.98, bottom=0.02)
    for ax, name in zip(axes, ["fvc_why_not_slice.png", "fvc_why_not_reject.png"]):
        ax.imshow(imread(OUT / name))
        ax.set_axis_off()
    path = OUT / "fvc_why_not.png"
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    render_slice()
    render_reject()
    montage()
